from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"gymnasium\.envs\.registration",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Overriding environment .* already in registry.*",
    category=UserWarning,
)

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


XML_PATH = Path(__file__).resolve().parents[1] / "flexiv_panda_world.xml"
HEADLESS_ARG = "--headless"
STEPS_ARG = "--steps"
COLORS = ["white", "green", "blue", "yellow", "red"]
SLOT_TEXTS = [
    "front left",
    "front center",
    "front right",
    "back left",
    "back center",
]

ALL_INSTRUCTIONS = [
    f"pick and place the {color} box to the {slot_text} slot"
    for color in COLORS
    for slot_text in SLOT_TEXTS
] + [
    "pick and place all boxes",
]

# ── helpers ──────────────────────────────────────────────────────────────────

def joint_qpos_index(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid == -1:
        raise ValueError(f"Joint not found: {name}")
    return int(model.jnt_qposadr[jid])


def joint_qvel_index(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid == -1:
        raise ValueError(f"Joint not found: {name}")
    return int(model.jnt_dofadr[jid])


def mat2quat_mj(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q


def quat_error_mj(q_des: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    def sci(q):
        return np.array([q[1], q[2], q[3], q[0]], dtype=float)

    return (Rotation.from_quat(sci(q_des)) * Rotation.from_quat(sci(q_cur)).inv()).as_rotvec()


def _cubic_ease(s: float) -> float:
    return 3.0 * s * s - 2.0 * s * s * s


def compute_gravity_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_dof: np.ndarray,
) -> np.ndarray:
    qvel_backup = data.qvel.copy()
    qacc_backup = data.qacc.copy()
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_fwdPosition(model, data)
    gravity_torque = data.qfrc_bias[arm_dof].copy()
    data.qvel[:] = qvel_backup
    data.qacc[:] = qacc_backup
    return gravity_torque


def compute_delta(current_ee: np.ndarray, target_ee: np.ndarray, max_step=10, min_step=5):
    delta = np.asarray(target_ee, dtype=np.float32) - np.asarray(current_ee, dtype=np.float32)
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return np.zeros(3, dtype=np.float32), dist
    direction = delta / dist
    step_size = float(np.clip(dist, min_step, max_step))
    # print("step_size: ", step_size)
    delta = direction * step_size
    # print("delta: ", delta)
    return delta.astype(np.float32), dist


# ── trajectory ───────────────────────────────────────────────────────────────

class Waypoint:
    __slots__ = ("pos", "quat", "hold_steps", "label")

    def __init__(self, pos, quat, hold_steps=0, label=""):
        self.pos = np.asarray(pos, dtype=float)
        self.quat = np.asarray(quat, dtype=float)
        self.hold_steps = int(hold_steps)
        self.label = str(label)


class TrajectoryPlanner:
    def __init__(self, waypoints: list[Waypoint], steps_per_segment: int = 400):
        assert len(waypoints) >= 2
        self.waypoints = waypoints
        self.steps_per_segment = int(steps_per_segment)
        self._seg = 0
        self._t = 0
        self._holding = 0
        self._done = False

    @property
    def done(self):
        return self._done

    def current_label(self):
        return self.waypoints[min(self._seg + 1, len(self.waypoints) - 1)].label

    def target(self):
        if self._seg >= len(self.waypoints) - 1:
            wp = self.waypoints[-1]
            return wp.pos.copy(), wp.quat.copy()

        wp0, wp1 = self.waypoints[self._seg], self.waypoints[self._seg + 1]
        alpha = _cubic_ease(self._t / self.steps_per_segment)
        pos = wp0.pos + alpha * (wp1.pos - wp0.pos)

        def sci(q):
            return np.array([q[1], q[2], q[3], q[0]], dtype=float)

        xyzw = Slerp(
            [0.0, 1.0],
            Rotation.concatenate(
                [
                    Rotation.from_quat(sci(wp0.quat)),
                    Rotation.from_quat(sci(wp1.quat)),
                ]
            ),
        )(float(alpha)).as_quat()
        quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)
        return pos, quat

    def step(self, reached: bool = False):
        if self._done:
            return
        if self._seg >= len(self.waypoints) - 1:
            self._done = True
            return
        if self._t < self.steps_per_segment:
            self._t += 1
            return
        if not reached:
            return
        if self._holding < self.waypoints[self._seg + 1].hold_steps:
            self._holding += 1
            return
        self._seg += 1
        self._t = 0
        self._holding = 0
        if self._seg >= len(self.waypoints) - 1:
            self._done = True


# ── 6-DOF IK ─────────────────────────────────────────────────────────────────

class TaskSpaceIK:
    def __init__(
        self,
        model,
        ee_site_id,
        arm_dof,
        arm_qidx,
        home_q,
        task_gain=0.6,
        nullspace_gain=0.02,
        damping=0.08,
        max_pos_err=0.08,
        max_rot_err=0.16,
        max_dq=0.35,
        freeze_jnt_err_thr=None,
    ):
        self.model = model
        self.ee_site_id = ee_site_id
        self.arm_dof = arm_dof
        self.arm_qidx = arm_qidx
        self.home_q = home_q.copy()
        self.task_gain = float(task_gain)
        self.nullspace_gain = float(nullspace_gain)
        self.damping = float(damping)
        self.max_pos_err = float(max_pos_err)
        self.max_rot_err = float(max_rot_err)
        self.max_dq = float(max_dq)
        self.freeze_jnt_err_thr = freeze_jnt_err_thr

        self._q_lo = np.array(
            [
                model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"_joint{i}")][0]
                for i in range(1, 8)
            ]
        )
        self._q_hi = np.array(
            [
                model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"_joint{i}")][1]
                for i in range(1, 8)
            ]
        )

        self.last_frozen = False
        self.last_dq = np.zeros(7)
        self.last_dq_task = np.zeros(7)
        self.last_dq_null = np.zeros(7)
        self.last_clip_mask = np.zeros(7, dtype=bool)

    def compute(self, data, target_pos, target_quat, q_cmd):
        q_cur = data.qpos[self.arm_qidx].copy()
        jnt_track_err = float(np.linalg.norm(q_cur - q_cmd))

        if self.freeze_jnt_err_thr is not None and jnt_track_err > self.freeze_jnt_err_thr:
            self.last_frozen = True
            self.last_dq[:] = 0.0
            self.last_dq_task[:] = 0.0
            self.last_dq_null[:] = 0.0
            self.last_clip_mask[:] = False
            return q_cmd.copy()

        self.last_frozen = False
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ee_site_id)
        J = np.vstack([jacp[:, self.arm_dof], jacr[:, self.arm_dof]])

        pos_err = np.clip(target_pos - data.site_xpos[self.ee_site_id], -self.max_pos_err, self.max_pos_err)
        rot_err = np.clip(
            quat_error_mj(target_quat, mat2quat_mj(data.site_xmat[self.ee_site_id])),
            -self.max_rot_err,
            self.max_rot_err,
        )
        e6 = np.concatenate([pos_err, rot_err])

        lam2 = self.damping**2
        JJT = J @ J.T
        solver = np.linalg.solve(JJT + lam2 * np.eye(6), self.task_gain * e6)
        dq_task = J.T @ solver

        J_pinv = J.T @ np.linalg.inv(JJT + lam2 * np.eye(6))
        N = np.eye(7) - J_pinv @ J
        dq_null = self.nullspace_gain * (N @ (self.home_q - q_cur))

        dq_raw = dq_task + dq_null
        dq = np.clip(dq_raw, -self.max_dq, self.max_dq)
        self.last_dq = dq.copy()
        self.last_dq_task = dq_task.copy()
        self.last_dq_null = dq_null.copy()
        self.last_clip_mask = np.abs(dq_raw) > self.max_dq

        q_next = np.clip(q_cur + dq, self._q_lo, self._q_hi)
        return q_next


# ── joint controller ─────────────────────────────────────────────────────────

class JointPositionCtrl:
    GRAV_SCALE = np.array([0.95, 1.22, 0.95, 1.22, 0.95, 1.16, 0.95])

    def __init__(self, model, arm_qidx, arm_didx, act_ids):
        self.model = model
        self.arm_qidx = arm_qidx
        self.arm_didx = arm_didx
        self.act_ids = act_ids
        self.outer_kp = np.array([0.50, 1.90, 0.50, 1.35, 0.45, 1.10, 0.70])
        self.outer_kd = np.array([0.03, 0.14, 0.03, 0.11, 0.03, 0.08, 0.05])
        self.outer_ki = np.array([0.00, 0.04, 0.00, 0.03, 0.01, 0.03, 0.01])
        self.integral_clip = np.array([0.00, 0.45, 0.00, 0.24, 0.15, 0.12, 0.30])
        self.integral_error = np.zeros(7)
        self.ctrl_offset = np.zeros(7)

    def reset_integral(self):
        self.integral_error[:] = 0.0

    def compute(self, data, desired_q, gravity_torque, profile=None):
        q = data.qpos[self.arm_qidx]
        qd = data.qvel[self.arm_didx]
        err = desired_q - q

        if profile is None:
            grav_scale = self.GRAV_SCALE
            kp_scale = np.ones(7)
            kd_scale = np.ones(7)
            ki_scale = np.ones(7)
        else:
            grav_scale = self.GRAV_SCALE * profile["grav_scale"]
            kp_scale = profile["kp_scale"]
            kd_scale = profile["kd_scale"]
            ki_scale = profile["ki_scale"]

        self.integral_error = np.clip(self.integral_error + err, -self.integral_clip, self.integral_clip)

        xml_kp = np.array([200.0, 500.0, 180.0, 300.0, 120.0, 100.0, 150.0])
        grav_offset = grav_scale * gravity_torque / xml_kp
        ctrl = (
            desired_q
            + self.ctrl_offset
            + grav_offset
            + (self.outer_kp * kp_scale) * err
            + (self.outer_ki * ki_scale) * self.integral_error
            - (self.outer_kd * kd_scale) * qd
        )

        sat_mask = np.zeros(7, dtype=bool)
        for i, aid in enumerate(self.act_ids):
            lo, hi = self.model.actuator_ctrlrange[aid]
            unclipped = ctrl[i]
            ctrl[i] = float(np.clip(unclipped, lo, hi))
            sat_mask[i] = abs(ctrl[i] - unclipped) > 1e-9
        return ctrl, grav_offset, sat_mask

    def calibrate(self, data, home_q, gripper_act_id, open_width, warm_steps=800, settle_steps=2000):
        print(f"[CALIB] warm={warm_steps} settle={settle_steps} ...")
        for k in range(warm_steps):
            data.ctrl[:7] = home_q
            if gripper_act_id != -1:
                data.ctrl[gripper_act_id] = open_width
            mujoco.mj_step(self.model, data)
            if k % 200 == 0:
                q_now = data.qpos[self.arm_qidx].copy()
                print(f"  [warm {k:4d}] q={np.round(q_now, 3)}")

        self.ctrl_offset = home_q - data.qpos[self.arm_qidx].copy()
        print(f"[CALIB] initial offset = {np.round(self.ctrl_offset, 4)}")

        for k in range(settle_steps):
            g_torque = compute_gravity_torque(
                self.model,
                data,
                np.array([joint_qvel_index(self.model, f"_joint{i}") for i in range(1, 8)]),
            )
            ctrl_q, _, _ = self.compute(data, home_q, g_torque)
            data.ctrl[:7] = ctrl_q
            if gripper_act_id != -1:
                data.ctrl[gripper_act_id] = open_width
            mujoco.mj_step(self.model, data)
            if k % 400 == 0:
                q_now = data.qpos[self.arm_qidx].copy()
                err = home_q - q_now
                print(f"  [settle {k:4d}] q={np.round(q_now, 3)}  err={np.round(err, 4)}")

        calibrated = data.qpos[self.arm_qidx].copy()
        self.ctrl_offset = home_q - calibrated
        self.reset_integral()
        print(f"[CALIB] done.  calibrated_home_q = {np.round(calibrated, 4)}")
        print(f"[CALIB] final ctrl_offset        = {np.round(self.ctrl_offset, 4)}")
        return calibrated


# ── main demos ───────────────────────────────────────────────────────────────

class RizonTaskSpaceDemo:
    ARM_JOINT_NAMES = [f"_joint{i}" for i in range(1, 8)]
    ARM_ACT_NAMES = [f"act_joint{i}" for i in range(1, 8)]
    GRIPPER_ACT_NAME = "act_gripper"
    GRIPPER_JOINT_NAME = "_finger_width_joint"
    BOX_BODY_NAMES = ("box", "green_box", "blue_box", "yellow_box", "red_box")
    EE_SITE_NAME = "ee_site"
    TCP_SITE_NAME = "tcp_ft_site"
    HOME_Q = np.array([0.0, -0.6981, 0.0, 1.5708, 0.0, 0.6981, 0.0])
    OPEN_WIDTH = 0.10
    INIT_HOLD_STEPS = 1
    ENABLE_SIM_SAFETY = True
    ENABLE_FREEZE_GUARD = True
    PICK_APPROACH_Z = 0.18
    PICK_GRASP_Z = 0.185
    PICK_LIFT_Z = 0.35
    PLACE_APPROACH_Z = 0.35
    PLACE_DROP_Z = 0.185
    PICK_HOLD_STEPS = 1
    GRASP_HOLD_STEPS = 1
    PLACE_HOLD_STEPS = 1
    REACH_POS_THR = 0.03
    REACH_QD_THR = 6.0
    GRIPPER_REACH_THR = 0.1
    SAFETY_MAX_DIST = 0.8
    SAFETY_MAX_Z_SAG = 0.30
    GRAVITY_UPDATE_EVERY = 3
    PLACE_SLOTS = (
        np.array([0.54, 0.00, 0.03]),
        np.array([0.66, 0.00, 0.03]),
        np.array([0.78, 0.00, 0.03]),
        np.array([0.54, 0.14, 0.03]),
        np.array([0.66, 0.14, 0.03]),
    )
    OSC_ACTION_DIM = 7
    OSC_MAX_DPOS = 0.4
    OSC_MAX_DROT = 0.20

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)

        self.arm_qidx = [joint_qpos_index(self.model, n) for n in self.ARM_JOINT_NAMES]
        self.arm_didx = [joint_qvel_index(self.model, n) for n in self.ARM_JOINT_NAMES]
        self.arm_dof = np.array(self.arm_didx, dtype=int)
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.ARM_ACT_NAMES]
        self.gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.GRIPPER_ACT_NAME)
        self.has_gripper = self.gripper_act_id != -1
        self.gripper_qidx = joint_qpos_index(self.model, self.GRIPPER_JOINT_NAME) if self.has_gripper else -1

        self.box_body_ids = []
        for name in self.BOX_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                raise ValueError(f"Body '{name}' not found")
            self.box_body_ids.append(bid)

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.EE_SITE_NAME)
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{self.EE_SITE_NAME}' not found")
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.TCP_SITE_NAME)

        self.data.qpos[self.arm_qidx] = self.HOME_Q
        self.data.ctrl[:7] = self.HOME_Q
        if self.has_gripper:
            self.data.ctrl[self.gripper_act_id] = self.OPEN_WIDTH
        mujoco.mj_forward(self.model, self.data)

        self.joint_ctrl = JointPositionCtrl(self.model, self.arm_qidx, self.arm_didx, self.act_ids)
        self.command_home_q = self.joint_ctrl.calibrate(self.data, self.HOME_Q, self.gripper_act_id, self.OPEN_WIDTH)

        mujoco.mj_forward(self.model, self.data)
        self.home_pos = self.data.site_xpos[self.ee_site_id].copy()
        self.home_quat = mat2quat_mj(self.data.site_xmat[self.ee_site_id])

        self.ik = TaskSpaceIK(
            self.model,
            self.ee_site_id,
            self.arm_dof,
            np.array(self.arm_qidx),
            self.command_home_q,
            freeze_jnt_err_thr=(0.08 if self.ENABLE_FREEZE_GUARD else None),
        )

        self.q_cmd = self.command_home_q.copy()
        self.planner = self._build_trajectory()
        self.final_hold_active = False
        self.final_hold_target_pos = self.home_pos.copy()
        self.final_hold_target_quat = self.home_quat.copy()
        self.final_hold_q_cmd = self.command_home_q.copy()
        self.final_hold_phase = "hold_home_final"
        self.osc_target_pos = self.home_pos.copy()
        self.osc_target_quat = self.home_quat.copy()
        self.gripper_width = self.OPEN_WIDTH
        self._gravity_torque_cache = np.zeros(7)
        self._gravity_cache_step = -1

        print("=" * 68)
        print("  Rizon Task-Space Controller  –  6-DOF  +  Gravity Feedforward")
        print("  control   : robosuite-like OSC_POSE action interface")
        print(f"  XML       : {XML_PATH.name}")
        print(f"  home_pos  : {np.round(self.home_pos, 4)}")
        print(f"  home_quat : {np.round(self.home_quat, 4)}  [w x y z]")
        print(
            f"  pick_z    : approach={self.PICK_APPROACH_Z:.3f}  grasp={self.PICK_GRASP_Z:.3f}  lift={self.PICK_LIFT_Z:.3f} m"
        )
        print(f"  place_z   : approach={self.PLACE_APPROACH_Z:.3f}  drop={self.PLACE_DROP_Z:.3f} m")
        print(
            f"  sim_guard : safety={'on' if self.ENABLE_SIM_SAFETY else 'off'}  freeze={'on' if self.ENABLE_FREEZE_GUARD else 'off'}"
        )
        print("=" * 68)

    def _build_trajectory(self):
        hp = self.home_pos
        hq = self.home_quat
        wps = [Waypoint(hp, hq, 1, "home_init"), Waypoint(hp, hq, 1, "go_home_surely")]

        for idx, body_id in enumerate(self.box_body_ids):
            box_pos = self.data.xpos[body_id].copy()
            place_pos = self.PLACE_SLOTS[min(idx, len(self.PLACE_SLOTS) - 1)].copy()
            pre_approach_box = np.array([box_pos[0], box_pos[1], hp[2] - 0.015])
            above_box = np.array([box_pos[0], box_pos[1], box_pos[2] + self.PICK_APPROACH_Z])
            grasp_box = np.array([box_pos[0], box_pos[1], box_pos[2] + self.PICK_GRASP_Z])
            lift_box = np.array([box_pos[0], box_pos[1], box_pos[2] + self.PICK_LIFT_Z])
            place_above = np.array([place_pos[0], place_pos[1], place_pos[2] + self.PLACE_APPROACH_Z])
            place_drop = np.array([place_pos[0], place_pos[1], place_pos[2] + self.PLACE_DROP_Z])
            place_retreat = np.array([place_pos[0], place_pos[1], place_pos[2] + self.PLACE_APPROACH_Z])
            tag = f"_{idx + 1}"
            wps.extend(
                [
                    Waypoint(pre_approach_box, hq, self.PICK_HOLD_STEPS, f"pre_approach_box{tag}"),
                    Waypoint(above_box, hq, self.PICK_HOLD_STEPS, f"approach_box{tag}"),
                    Waypoint(grasp_box, hq, self.GRASP_HOLD_STEPS, f"gripper_close_pick{tag}"),
                    Waypoint(lift_box, hq, self.PICK_HOLD_STEPS, f"lift_box{tag}"),
                    Waypoint(lift_box, hq, self.PICK_HOLD_STEPS, f"hold_lift_final{tag}"),
                    Waypoint(place_above, hq, self.PICK_HOLD_STEPS, f"move_to_place_above{tag}"),
                    Waypoint(place_drop, hq, self.PICK_HOLD_STEPS, f"descend_place{tag}"),
                    Waypoint(place_drop, hq, self.PLACE_HOLD_STEPS, f"gripper_open_place{tag}"),
                    Waypoint(place_retreat, hq, self.PICK_HOLD_STEPS, f"retreat_after_place{tag}"),
                ]
            )

        wps.append(Waypoint(hp, hq, self.PICK_HOLD_STEPS, "return_home_final"))
        return TrajectoryPlanner(wps, steps_per_segment=5)

    def _safety_ok(self):
        if not self.ENABLE_SIM_SAFETY:
            return True, "ok"
        pos = self.data.site_xpos[self.ee_site_id]
        dz = float(pos[2] - self.home_pos[2])
        dist = float(np.linalg.norm(pos - self.home_pos))
        if dz < -self.SAFETY_MAX_Z_SAG:
            return False, "z_sag"
        if dist > self.SAFETY_MAX_DIST:
            return False, "too_far"
        return True, "ok"

    def _status_reason(self, target_pos):
        cur_pos = self.data.site_xpos[self.ee_site_id].copy()
        cur_q = self.data.qpos[self.arm_qidx].copy()
        pos_err = float(np.linalg.norm(cur_pos - target_pos))
        jnt_err = float(np.linalg.norm(cur_q - self.q_cmd))
        dz = float(cur_pos[2] - self.home_pos[2])
        z_err = float(target_pos[2] - cur_pos[2])
        if dz < -self.SAFETY_MAX_Z_SAG:
            return "no_works_z_sag", pos_err, jnt_err, z_err
        if pos_err < 0.02 and jnt_err < 0.08:
            return "works_tracking_good", pos_err, jnt_err, z_err
        if pos_err < 0.05:
            return "works_but_not_precise", pos_err, jnt_err, z_err
        return "no_works_tracking_weak", pos_err, jnt_err, z_err

    def _target_reached(self, target_pos):
        cur_pos = self.data.site_xpos[self.ee_site_id].copy()
        cur_qd = self.data.qvel[self.arm_didx].copy()
        pos_err = float(np.linalg.norm(cur_pos - target_pos))
        qd_norm = float(np.linalg.norm(cur_qd))
        return pos_err < self.REACH_POS_THR and qd_norm < self.REACH_QD_THR

    def _gripper_reached(self, phase):
        cur = self._current_gripper_width()
        if phase.startswith("gripper_close_pick"):
            return 0.04 <= cur <= 0.07
        if phase.startswith("gripper_open"):
            return abs(cur - self.OPEN_WIDTH) < self.GRIPPER_REACH_THR
        if phase.startswith("gripper_close"):
            return abs(cur - 0.0) < self.GRIPPER_REACH_THR
        return True

    def _phase_reached(self, phase: str, target_pos: np.ndarray):
        arm_ok = self._target_reached(target_pos)
        if phase.startswith("gripper_"):
            return self._gripper_reached(phase)
        return arm_ok

    def _current_gripper_width(self):
        if not self.has_gripper or self.gripper_qidx == -1:
            return 0.0
        return float(self.data.qpos[self.gripper_qidx])

    def _set_osc_target(self, pos: np.ndarray, quat: np.ndarray):
        self.osc_target_pos = np.asarray(pos, dtype=float).copy()
        self.osc_target_quat = np.asarray(quat, dtype=float).copy()

    def apply_osc_pose_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).reshape(self.OSC_ACTION_DIM)
        action = np.clip(action, -1.0, 1.0)
        dpos = action[:3] * self.OSC_MAX_DPOS
        drot = action[3:6] * self.OSC_MAX_DROT
        self.osc_target_pos = self.osc_target_pos + dpos

        cur_rot = Rotation.from_quat(
            [self.osc_target_quat[1], self.osc_target_quat[2], self.osc_target_quat[3], self.osc_target_quat[0]]
        )
        next_rot = Rotation.from_rotvec(drot) * cur_rot
        xyzw = next_rot.as_quat()
        self.osc_target_quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

        if self.has_gripper:
            open_close = 0.5 * (action[6] + 1.0)
            self.gripper_width = float(np.clip(open_close * self.OPEN_WIDTH, 0.0, self.OPEN_WIDTH))

    def _goal_to_osc_pose_action(self, goal_pos: np.ndarray, goal_quat: np.ndarray) -> np.ndarray:
        pos_delta = np.clip(goal_pos - self.osc_target_pos, -self.OSC_MAX_DPOS, self.OSC_MAX_DPOS)
        rot_delta = quat_error_mj(goal_quat, self.osc_target_quat)
        rot_delta = np.clip(rot_delta, -self.OSC_MAX_DROT, self.OSC_MAX_DROT)
        action = np.zeros(self.OSC_ACTION_DIM)
        action[:3] = pos_delta / self.OSC_MAX_DPOS
        action[3:6] = rot_delta / self.OSC_MAX_DROT
        action[6] = 1.0
        return np.clip(action, -1.0, 1.0)

    def _read_ft(self):
        if self.tcp_site_id == -1:
            return np.zeros(3), np.zeros(3)
        return self.data.sensor("tcp_force").data.copy(), self.data.sensor("tcp_torque").data.copy()

    def _apply_phase_gripper(self, phase: str):
        if not self.has_gripper:
            return
        if phase.startswith("gripper_open_") or phase.startswith(("retreat_after_place", "return_home_final")):
            self.gripper_width = self.OPEN_WIDTH
        elif phase.startswith("gripper_close_") or phase.startswith(("lift_", "hold_lift_", "move_to_place_", "descend_place")):
            self.gripper_width = 0.0

    def _phase_control_profile(self, phase: str):
        ones = np.ones(7)
        if phase.startswith(("approach_", "descend_", "lift_", "hold_lift_", "move_to_place_", "retreat_after_place")):
            return {
                "grav_scale": np.array([1.00, 1.12, 1.00, 1.16, 1.00, 1.12, 1.00]),
                "kp_scale": np.array([1.00, 1.14, 1.00, 1.14, 1.00, 1.10, 1.00]),
                "kd_scale": np.array([1.00, 1.20, 1.00, 1.18, 1.00, 1.14, 1.00]),
                "ki_scale": np.array([1.00, 0.90, 1.00, 0.90, 1.00, 0.90, 1.00]),
            }
        return {"grav_scale": ones, "kp_scale": ones, "kd_scale": ones, "ki_scale": ones}

    def _get_gravity_torque(self, step_count: int):
        if self._gravity_cache_step < 0 or step_count - self._gravity_cache_step >= self.GRAVITY_UPDATE_EVERY:
            self._gravity_torque_cache = compute_gravity_torque(self.model, self.data, self.arm_dof)
            self._gravity_cache_step = step_count
        return self._gravity_torque_cache

    def _controller_step(self, step_count: int):
        g_torque = self._get_gravity_torque(step_count)
        if step_count < self.INIT_HOLD_STEPS:
            target_pos = self.home_pos.copy()
            target_quat = self.home_quat.copy()
            self._set_osc_target(target_pos, target_quat)
            self.q_cmd = self.command_home_q.copy()
            phase = "init_hold"
            run_ik = False
            self.final_hold_active = False
        else:
            safe, reason = self._safety_ok()
            if not safe:
                target_pos = self.home_pos.copy()
                target_quat = self.home_quat.copy()
                self._set_osc_target(target_pos, target_quat)
                self.q_cmd = self.command_home_q.copy()
                self.joint_ctrl.reset_integral()
                phase = f"recover_{reason}"
                run_ik = False
                self.final_hold_active = False
            elif self.final_hold_active:
                target_pos = self.final_hold_target_pos.copy()
                target_quat = self.final_hold_target_quat.copy()
                self._set_osc_target(target_pos, target_quat)
                self.q_cmd = self.final_hold_q_cmd.copy()
                phase = self.final_hold_phase
                run_ik = False
            else:
                goal_pos, goal_quat = self.planner.target()
                phase = self.planner.current_label()
                if phase == "hold_home_final":
                    target_pos = self.home_pos.copy()
                    target_quat = self.home_quat.copy()
                    self._set_osc_target(target_pos, target_quat)
                    self.q_cmd = self.command_home_q.copy()
                    self.joint_ctrl.reset_integral()
                    self.final_hold_target_pos = target_pos.copy()
                    self.final_hold_target_quat = target_quat.copy()
                    self.final_hold_q_cmd = self.q_cmd.copy()
                    self.final_hold_phase = phase
                    self.final_hold_active = True
                    run_ik = False
                else:
                    osc_action = self._goal_to_osc_pose_action(goal_pos, goal_quat)
                    self.apply_osc_pose_action(osc_action)
                    target_pos = self.osc_target_pos.copy()
                    target_quat = self.osc_target_quat.copy()
                    self._apply_phase_gripper(phase)
                    run_ik = True
                    self.planner.step(reached=self._phase_reached(phase, target_pos))
                if self.planner.done:
                    self.final_hold_target_pos = target_pos.copy()
                    self.final_hold_target_quat = target_quat.copy()
                    self._set_osc_target(target_pos, target_quat)
                    self.final_hold_q_cmd = self.q_cmd.copy()
                    self.final_hold_phase = phase
                    self.joint_ctrl.reset_integral()
                    run_ik = False
                    self.final_hold_active = True

        if run_ik:
            self.q_cmd = self.ik.compute(self.data, target_pos, target_quat, self.q_cmd)

        ctrl_q, grav_offset, sat_mask = self.joint_ctrl.compute(
            self.data, self.q_cmd, g_torque, profile=self._phase_control_profile(phase)
        )
        self.data.ctrl[:7] = ctrl_q
        if self.has_gripper:
            self.data.ctrl[self.gripper_act_id] = self.gripper_width
        mujoco.mj_step(self.model, self.data)

        cur_pos = self.data.site_xpos[self.ee_site_id].copy()
        cur_quat = mat2quat_mj(self.data.site_xmat[self.ee_site_id])
        cur_q = self.data.qpos[self.arm_qidx].copy()
        cur_qd = self.data.qvel[self.arm_didx].copy()
        force, torque = self._read_ft()
        pos_err_xyz = target_pos - cur_pos
        rot_err = float(np.linalg.norm(quat_error_mj(target_quat, cur_quat)))
        q_err = self.q_cmd - cur_q
        qd_norm = float(np.linalg.norm(cur_qd))
        dz = float(cur_pos[2] - self.home_pos[2])
        gripper_cur = self._current_gripper_width()
        status_reason, pos_err, jnt_err, z_err = self._status_reason(target_pos)

        return {
            "phase": phase,
            "target_pos": target_pos,
            "ctrl_q": ctrl_q,
            "g_torque": g_torque,
            "cur_pos": cur_pos,
            "cur_quat": cur_quat,
            "force": force,
            "torque": torque,
            "pos_err": pos_err,
            "pos_err_xyz": pos_err_xyz,
            "rot_err": rot_err,
            "jnt_err": jnt_err,
            "q_err": q_err,
            "qd_norm": qd_norm,
            "cur_qd": cur_qd,
            "dz": dz,
            "z_err": z_err,
            "gripper_cur": gripper_cur,
            "gripper_cmd": float(self.gripper_width),
            "status_reason": status_reason,
            "grav_offset": grav_offset,
            "sat_mask": sat_mask,
            "dq_task": self.ik.last_dq_task.copy(),
            "dq_null": self.ik.last_dq_null.copy(),
            "dq_cmd": self.ik.last_dq.copy(),
            "dq_clip_mask": self.ik.last_clip_mask.copy(),
        }


class RizonFastPickPlaceDemo(RizonTaskSpaceDemo):
    ENABLE_SIM_SAFETY = False
    ENABLE_FREEZE_GUARD = False
    TARGET_CONTROL_HZ = 100.0
    PICK_APPROACH_Z = 0.20
    PICK_GRASP_Z = 0.180
    PICK_LIFT_Z = 0.35
    PLACE_APPROACH_Z = 0.35
    PLACE_DROP_Z = 0.19
    PICK_HOLD_STEPS = 1
    GRASP_HOLD_STEPS = 1
    PLACE_HOLD_STEPS = 1
    REACH_POS_THR = 0.040
    REACH_QD_THR = 0.18
    GRIPPER_REACH_THR = 0.08
    GRAVITY_UPDATE_EVERY = 6
    OSC_MAX_DPOS = 0.022
    OSC_MAX_DROT = 0.11
    BOX_COLOR_NAMES = ("white", "green", "blue", "yellow", "red")
    SLOT_LABELS = ("front_left", "front_center", "front_right", "back_left", "back_center")
    RANDOM_X_RANGE = (0.38, 0.82)
    RANDOM_Y_RANGE = (-0.34, -0.10)
    RANDOM_BOX_Z = 0.03
    RANDOM_MIN_DIST = 0.14

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.current_place_slots = tuple(slot.copy() for slot in self.PLACE_SLOTS)
        self.current_instruction = "pick and place all boxes"
        self.current_box_setup = []
        self.active_box_index = 0
        self.box_free_qidx = {}
        self.box_free_didx = {}
        super().__init__()
        for body_name in self.BOX_BODY_NAMES:
            joint_name = f"{body_name}_free"
            self.box_free_qidx[body_name] = joint_qpos_index(self.model, joint_name)
            self.box_free_didx[body_name] = joint_qvel_index(self.model, joint_name)
        sim_dt = float(self.model.opt.timestep)
        self.control_repeat = max(1, int(round((1.0 / self.TARGET_CONTROL_HZ) / sim_dt)))
        self._cached_ctrl = self.data.ctrl.copy()
        self._cached_step = None
        self.reset_random_task()
        print(
            f"  fast_mode  : control_repeat={self.control_repeat}  sim_dt={sim_dt:.4f} s  control_hz~{1.0 / (sim_dt * self.control_repeat):.2f}"
        )
        print(f"  instruction: {self.current_instruction}")
        print("=" * 68)

    def _sample_box_positions(self) -> list[np.ndarray]:
        base_min_dist = float(self.RANDOM_MIN_DIST)
        relaxed_dists = (base_min_dist, max(0.12, base_min_dist - 0.01), max(0.10, base_min_dist - 0.02))
        last_failed_dist = base_min_dist
        for min_dist in relaxed_dists:
            positions = []
            success = True
            for _ in self.BOX_BODY_NAMES:
                for _attempt in range(300):
                    pos = np.array(
                        [
                            self.rng.uniform(*self.RANDOM_X_RANGE),
                            self.rng.uniform(*self.RANDOM_Y_RANGE),
                            self.RANDOM_BOX_Z,
                        ],
                        dtype=float,
                    )
                    if all(np.linalg.norm(pos[:2] - other[:2]) >= min_dist for other in positions):
                        positions.append(pos)
                        break
                else:
                    success = False
                    last_failed_dist = min_dist
                    break
            if success:
                if min_dist < base_min_dist:
                    print(f"  [random_task] relaxed box spacing {base_min_dist:.3f} -> {min_dist:.3f}")
                return positions
        raise RuntimeError(
            "Failed to sample non-overlapping random box positions "
            f"(min_dist down to {last_failed_dist:.3f})"
        )

    def _slot_label_to_text(self, slot_label: str) -> str:
        return str(slot_label).replace("_", " ")

    def _instruction_from_assignments(self) -> str:
        setup = self.current_box_setup[self.active_box_index]
        slot_text = self._slot_label_to_text(setup["slot_label"])
        return f"pick and place the {setup['color']} box to the {slot_text} slot"

    def reset_random_task(self):
        sampled_positions = self._sample_box_positions()

        self.current_box_setup = []
        self.current_place_slots = tuple(slot.copy() for slot in self.PLACE_SLOTS)

        self.active_box_index = int(self.rng.integers(len(self.BOX_BODY_NAMES)))
        target_slot_index = int(self.rng.integers(len(self.PLACE_SLOTS)))

        for idx, body_name in enumerate(self.BOX_BODY_NAMES):
            qidx = self.box_free_qidx[body_name]
            didx = self.box_free_didx[body_name]
            pos = sampled_positions[idx]

            self.data.qpos[qidx : qidx + 7] = np.array(
                [pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0],
                dtype=float,
            )
            self.data.qvel[didx : didx + 6] = 0.0

            if idx == self.active_box_index:
                place_pos = self.PLACE_SLOTS[target_slot_index].copy()
                slot_label = self.SLOT_LABELS[target_slot_index]
            else:
                place_pos = None
                slot_label = None

            self.current_box_setup.append(
                {
                    "body_name": body_name,
                    "color": self.BOX_COLOR_NAMES[idx],
                    "pick_pos": pos.copy(),
                    "place_pos": place_pos,
                    "slot_label": slot_label,
                }
            )

        mujoco.mj_forward(self.model, self.data)
        self.planner = self._build_trajectory()
        self.current_instruction = self._instruction_from_assignments()
        self.final_hold_active = False
        self.final_hold_target_pos = self.home_pos.copy()
        self.final_hold_target_quat = self.home_quat.copy()
        self.final_hold_q_cmd = self.command_home_q.copy()
        self.final_hold_phase = "hold_home_final"
        self.osc_target_pos = self.home_pos.copy()
        self.osc_target_quat = self.home_quat.copy()
        self.q_cmd = self.command_home_q.copy()
        self.gripper_width = self.OPEN_WIDTH
        self._cached_step = None
        if self.has_gripper:
            self.data.ctrl[self.gripper_act_id] = self.OPEN_WIDTH
        if self.gripper_qidx != -1:
            self.data.qpos[self.gripper_qidx] = self.OPEN_WIDTH
            self.data.qvel[joint_qvel_index(self.model, self.GRIPPER_JOINT_NAME)] = 0.0
        mujoco.mj_forward(self.model, self.data)
        print("  random_task :")
        for idx, setup in enumerate(self.current_box_setup):
            marker = "*" if idx == self.active_box_index else " "

            pick = np.round(setup["pick_pos"], 3).tolist()

            place = (
                np.round(setup["place_pos"], 3).tolist()
                if setup["place_pos"] is not None
                else None
            )

            slot = setup["slot_label"]

            print(f"  {marker} {setup['color']:<6s} pick={pick} -> place={place} ({slot})")

    def reset_episode(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        self.data.qpos[self.arm_qidx] = self.command_home_q
        self.data.qvel[self.arm_didx] = 0.0
        self.data.ctrl[:7] = self.command_home_q
        if self.has_gripper:
            self.data.ctrl[self.gripper_act_id] = self.OPEN_WIDTH
        mujoco.mj_forward(self.model, self.data)
        self.reset_random_task()
        self._cached_ctrl = self.data.ctrl.copy()
        self._cached_step = None
        self.gripper_width = self.OPEN_WIDTH
        if self.has_gripper:
            self.data.ctrl[self.gripper_act_id] = self.OPEN_WIDTH
            if self.gripper_qidx != -1:
                self.data.qpos[self.gripper_qidx] = self.OPEN_WIDTH
                self.data.qvel[joint_qvel_index(self.model, self.GRIPPER_JOINT_NAME)] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _build_trajectory(self):
        hp = self.home_pos
        hq = self.home_quat
        wps = []

        idx = int(getattr(self, "active_box_index", 0))

        # Safe fallback during super().__init__() before reset_random_task()
        if (
            not hasattr(self, "current_box_setup")
            or self.current_box_setup is None
            or len(self.current_box_setup) == 0
            or idx >= len(self.current_box_setup)
            or self.current_box_setup[idx].get("place_pos", None) is None
        ):
            # simple dummy hold trajectory so base init can finish
            wps.extend([
                Waypoint(hp.copy(), hq.copy(), 1, "home_init"),
                Waypoint(hp.copy(), hq.copy(), 1, "go_home_surely"),
            ])
            return TrajectoryPlanner(wps, steps_per_segment=5)

        setup = self.current_box_setup[idx]

        body_id = self.box_body_ids[idx]
        box_pos = self.data.xpos[body_id].copy()
        place_pos = np.asarray(setup["place_pos"], dtype=float).copy()

        pre_approach_box = np.array([box_pos[0], box_pos[1], hp[2] - 0.1], dtype=float)
        above_box = np.array([box_pos[0], box_pos[1], box_pos[2] + self.PICK_APPROACH_Z + 0.05], dtype=float)
        above_box_v1 = np.array([box_pos[0], box_pos[1], box_pos[2] + self.PICK_APPROACH_Z - 0.01], dtype=float)
        grasp_box = np.array([box_pos[0], box_pos[1], box_pos[2] + self.PICK_GRASP_Z], dtype=float)
        lift_box = np.array([box_pos[0], box_pos[1], box_pos[2] + self.PICK_LIFT_Z], dtype=float)
        place_above = np.array([place_pos[0], place_pos[1], place_pos[2] + self.PLACE_APPROACH_Z], dtype=float)
        place_drop = np.array([place_pos[0], place_pos[1], place_pos[2] + self.PLACE_DROP_Z], dtype=float)
        place_retreat = np.array([place_pos[0], place_pos[1], place_pos[2] + self.PLACE_APPROACH_Z], dtype=float)

        tag = f"_{idx + 1}"
        wps.extend([
            Waypoint(pre_approach_box, hq, self.PICK_HOLD_STEPS, f"pre_approach_box{tag}"),
            Waypoint(above_box, hq, self.PICK_HOLD_STEPS, f"approach_box{tag}"),
            Waypoint(above_box_v1, hq, self.PICK_HOLD_STEPS, f"approach_box_v1{tag}"),
            Waypoint(grasp_box, hq, self.GRASP_HOLD_STEPS, f"gripper_close_pick{tag}"),
            Waypoint(lift_box, hq, self.PICK_HOLD_STEPS, f"lift_box{tag}"),
            Waypoint(place_above, hq, self.PICK_HOLD_STEPS, f"move_to_place_above{tag}"),
            Waypoint(place_drop, hq, self.PICK_HOLD_STEPS, f"descend_place{tag}"),
            Waypoint(place_drop, hq, self.PLACE_HOLD_STEPS, f"gripper_open_place{tag}"),
            Waypoint(place_retreat, hq, self.PICK_HOLD_STEPS, f"retreat_after_place{tag}"),
        ])
        return TrajectoryPlanner(wps, steps_per_segment=5)

    def _gripper_reached(self, phase):
        cur = self._current_gripper_width()
        if phase.startswith("gripper_close_pick"):
            return 0.03 <= cur <= 0.07
        if phase.startswith("gripper_open"):
            return abs(cur - self.OPEN_WIDTH) < self.GRIPPER_REACH_THR
        if phase.startswith("gripper_close"):
            return abs(cur - 0.0) < self.GRIPPER_REACH_THR
        return True

    def _controller_step(self, step_count: int):
        if step_count % self.control_repeat == 0 or self._cached_step is None:
            step = super()._controller_step(step_count)
            self._cached_ctrl = self.data.ctrl.copy()
            self._cached_step = step
            return step
        self.data.ctrl[:] = self._cached_ctrl
        mujoco.mj_step(self.model, self.data)
        return self._cached_step


# ── wrapper ──────────────────────────────────────────────────────────────────

class MujocoRizonWrapper:
    """
    Self-contained MuJoCo wrapper.

    No dependency on test_mujoco_fastPickPlace.py or test_mujoco_TaskCtrl.py.
    Everything needed for the Rizon fast pick-place environment is implemented
    directly in this file.
    """

    def __init__(
        self,
        env_name: str = "rizon-pick-place-v0",
        seed: int = 42,
        render_mode: str = "rgb_array",
        camera_name: str = "cam",
        image_size: int = 128,
        frame_skip: int = 1,
        max_episode_steps: int = 200,
        task_box_index: int = 0,
        place_slot_index: int | None = None,
    ):
        self.env_name = env_name
        self.seed = int(seed)
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.image_size = int(image_size)
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.task_box_index = int(task_box_index)
        self.place_slot_index = int(task_box_index if place_slot_index is None else place_slot_index)

        self.demo = RizonFastPickPlaceDemo(seed=self.seed)
        self.model = self.demo.model
        self.data = self.demo.data

        self.renderer = None
        self._render_error = None
        if self.render_mode == "rgb_array":
            try:
                self.renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
            except Exception as exc:
                self._render_error = str(exc)
                self.renderer = None

        self._base_qpos = self.data.qpos.copy()
        self._base_qvel = self.data.qvel.copy()
        self._base_act = self.data.act.copy() if self.model.na > 0 else None
        self._base_ctrl = self.data.ctrl.copy()

        self._step_count = 0
        self._sim_step_count = 0
        self._teacher_step_count = 0
        self.last_instruction = getattr(self.demo, "current_instruction", "pick and place all boxes")

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        state = self._get_state()
        image = self._get_image()
        self.state_dim = int(state.shape[0])
        self.action_dim = int(self.action_space.shape[0])
        self.obs_shape = image.shape

    def _reset_sim_state(self):
        self.data.qpos[:] = self._base_qpos
        self.data.qvel[:] = self._base_qvel
        if self._base_act is not None:
            self.data.act[:] = self._base_act
        self.data.ctrl[:] = self._base_ctrl
        mujoco.mj_forward(self.model, self.data)
        self.demo.q_cmd = self.demo.command_home_q.copy()
        self.demo.gripper_width = self.demo.OPEN_WIDTH
        self.demo.joint_ctrl.reset_integral()
        self.demo._gravity_cache_step = -1
        self.demo.planner = self.demo._build_trajectory()
        self.demo.final_hold_active = False
        self.demo.final_hold_target_pos = self.demo.home_pos.copy()
        self.demo.final_hold_target_quat = self.demo.home_quat.copy()
        self.demo.final_hold_q_cmd = self.demo.command_home_q.copy()
        self.demo.final_hold_phase = "hold_home_final"
        ee_pos = self.data.site_xpos[self.demo.ee_site_id].copy()
        ee_quat = mat2quat_mj(self.data.site_xmat[self.demo.ee_site_id])
        self.demo._set_osc_target(ee_pos, ee_quat)
        self._step_count = 0
        self._sim_step_count = 0
        self._teacher_step_count = 0
        self.fixed_quat = self.demo.home_quat.copy()
        self.demo.gripper_width = self.demo.OPEN_WIDTH

    def _get_box_pos(self, index: int) -> np.ndarray:
        return self.data.xpos[self.demo.box_body_ids[index]].copy()
    def _get_target_place_pos(self) -> np.ndarray:
        if hasattr(self.demo, "current_box_setup") and self.demo.current_box_setup:
            idx = int(min(self.task_box_index, len(self.demo.current_box_setup) - 1))
            place_pos = self.demo.current_box_setup[idx].get("place_pos", None)
            if place_pos is not None:
                return np.asarray(place_pos, dtype=np.float32).copy()

        slot_idx = int(min(max(self.place_slot_index, 0), len(self.demo.PLACE_SLOTS) - 1))
        slot = self.demo.PLACE_SLOTS[slot_idx]
        return np.asarray(slot, dtype=np.float32).copy()
    def _get_state(self) -> np.ndarray:
        """
        Minimal proprioceptive state:
        [arm_q(7),
        ee_pos(3),
        grip(1)]
        total = 11
        """
        arm_q = self.data.qpos[self.demo.arm_qidx].copy().astype(np.float32)          # 7
        ee_pos = self.data.site_xpos[self.demo.ee_site_id].copy().astype(np.float32)  # 3
        grip = np.array([self.demo._current_gripper_width()], dtype=np.float32)        # 1

        state = np.concatenate(
            [arm_q, ee_pos, grip],
            axis=0
        )
        return state.astype(np.float32)

    def _get_image(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")
        if self.renderer is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.renderer.update_scene(self.data, camera=self.camera_name)
        return np.asarray(self.renderer.render(), dtype=np.uint8)

    def _compute_reward_done(self):
        box_pos = self._get_box_pos(self.task_box_index)
        place_pos = self._get_target_place_pos()
        ee_pos = self.data.site_xpos[self.demo.ee_site_id].copy()
        box_to_place = float(np.linalg.norm(box_pos - place_pos))
        ee_to_box = float(np.linalg.norm(ee_pos - box_pos))
        grip_open = self.demo._current_gripper_width()
        success = (
            abs(box_pos[0] - place_pos[0]) < 0.04
            and abs(box_pos[1] - place_pos[1]) < 0.04
            and abs(box_pos[2] - place_pos[2]) < 0.05
            and grip_open > 0.08
        )
        reward = -(box_to_place + 0.1 * ee_to_box)
        if success:
            reward += 1.0
        done = success or (self._step_count >= self.max_episode_steps)
        info = {
            "success": int(success),
            "task_box_index": self.task_box_index,
            "place_slot_index": self.place_slot_index,
            "box_to_place_dist": box_to_place,
            "ee_to_box_dist": ee_to_box,
        }
        return float(reward), bool(done), info

    def _low_level_step(self):
        g_torque = self.demo._get_gravity_torque(self._sim_step_count)
        self.demo.q_cmd = self.demo.ik.compute(
            self.data,
            self.demo.osc_target_pos,
            self.demo.osc_target_quat,
            self.demo.q_cmd,
        )
        ctrl_q, _, _ = self.demo.joint_ctrl.compute(
            self.data,
            self.demo.q_cmd,
            g_torque,
            profile=self.demo._phase_control_profile("vla_env"),
        )
        self.data.ctrl[:7] = ctrl_q
        if self.demo.has_gripper:
            self.data.ctrl[self.demo.gripper_act_id] = self.demo.gripper_width
        mujoco.mj_step(self.model, self.data)
        self._sim_step_count += 1

    def _set_task_from_instruction(self, instruction: str) -> bool:
        if instruction is None:
            return False

        text = str(instruction).lower().strip()

        color_names = ["white", "green", "blue", "yellow", "red"]
        slot_map = {
            "front left": "front_left",
            "front center": "front_center",
            "front right": "front_right",
            "back left": "back_left",
            "back center": "back_center",
        }

        found_idx = None
        for i, c in enumerate(color_names):
            if c in text:
                found_idx = i
                break
        if found_idx is None:
            return False

        found_slot_label = None
        for slot_text, slot_label in slot_map.items():
            if slot_text in text:
                found_slot_label = slot_label
                break
        if found_slot_label is None:
            return False

        slot_idx = self.demo.SLOT_LABELS.index(found_slot_label)

        self.task_box_index = int(found_idx)
        self.place_slot_index = int(slot_idx)

        if hasattr(self.demo, "active_box_index"):
            self.demo.active_box_index = int(found_idx)

        if hasattr(self.demo, "current_box_setup") and self.demo.current_box_setup:
            for i, setup in enumerate(self.demo.current_box_setup):
                if i == found_idx:
                    setup["place_pos"] = self.demo.PLACE_SLOTS[slot_idx].copy()
                    setup["slot_label"] = found_slot_label

            self.demo.current_instruction = (
                f"pick and place the {color_names[found_idx]} box "
                f"to the {found_slot_label.replace('_', ' ')} slot"
            )
            self.demo.planner = self.demo._build_trajectory()

        return True

    def reset(self, seed=None, instruction=None):
        if seed is not None:
            self.seed = int(seed)
        self._reset_sim_state()
        if hasattr(self.demo, "reset_episode"):
            self.demo.reset_episode(seed=self.seed)
        used_instruction_task = False
        if instruction is not None and str(instruction).lower().strip() != "auto":
            used_instruction_task = self._set_task_from_instruction(instruction)
        if not used_instruction_task:
            if hasattr(self.demo, "active_box_index"):
                self.task_box_index = int(self.demo.active_box_index)
                if (
                    hasattr(self.demo, "current_box_setup")
                    and self.demo.current_box_setup
                    and self.task_box_index < len(self.demo.current_box_setup)
                ):
                    slot_label = self.demo.current_box_setup[self.task_box_index]["slot_label"]
                    if slot_label in self.demo.SLOT_LABELS:
                        self.place_slot_index = int(self.demo.SLOT_LABELS.index(slot_label))
                    else:
                        self.place_slot_index = -1
        self.last_instruction = getattr(self.demo, "current_instruction", instruction or "")
        image = self._get_image()
        state = self._get_state()
        info = {
            "task_box_index": self.task_box_index,
            "place_slot_index": self.place_slot_index,
            "instruction": self.last_instruction,
        }
        if hasattr(self.demo, "current_box_setup") and self.demo.current_box_setup:
            setup = self.demo.current_box_setup[self.task_box_index]
            info["task_color"] = setup["color"]
            info["target_slot"] = setup["slot_label"]
        if self._render_error is not None:
            info["render_error"] = self._render_error
        return image, state, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        action = np.clip(action, -1.0, 1.0)
        action7 = np.array([action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]], dtype=np.float32)
        self.demo.apply_osc_pose_action(action7)
        if self.demo.has_gripper:
            open_close = 0.5 * (float(action[3]) + 1.0)
            self.demo.gripper_width = float(np.clip(open_close * self.demo.OPEN_WIDTH, 0.0, self.demo.OPEN_WIDTH))
        for _ in range(self.frame_skip):
            self._low_level_step()
        self._step_count += 1
        image = self._get_image()
        state = self._get_state()
        reward, done, info = self._compute_reward_done()
        if self._render_error is not None:
            info["render_error"] = self._render_error
        return image, state, reward, done, info

    def get_instruction(self) -> str:
        return self.last_instruction

    def check_current_task_success(self) -> bool:
        idx = int(self.task_box_index)
        box_pos = self._get_box_pos(idx)
        target = self._get_target_place_pos()
        xy_err = float(np.linalg.norm(box_pos[:2] - target[:2]))
        z_err = float(abs(box_pos[2] - target[2]))

        if xy_err < 0.05 and z_err < 0.06:
            print("\n[SUCCESS]")
            print(f"  idx     : {idx}")
            print(f"  box_pos : {box_pos}")
            print(f"  target  : {target}")
            print(f"  xy_err  : {xy_err:.4f}")
            print(f"  z_err   : {z_err:.4f}")

        return xy_err < 0.05 and z_err < 0.06

    def teacher_step(self):
        ee_pos = self.data.site_xpos[self.demo.ee_site_id].copy()
        step = self.demo._controller_step(self._teacher_step_count)
        target_pos = step["target_pos"]
        delta, dist = compute_delta(ee_pos, target_pos, max_step=getattr(self.demo, "OSC_MAX_DPOS", 0.5))
        grip_action = 1.0 if step["gripper_cmd"] > self.demo.OPEN_WIDTH * 0.5 else -1.0
        action = np.zeros(4, dtype=np.float32)
        action[:3] = delta
        action[3] = grip_action
        self._teacher_step_count += 1
        info = {
            "phase": step["phase"],
            "target_pos": target_pos.copy(),
            "dist": float(dist),
            "planner_done": bool(self.demo.planner.done),
            "success": bool(self.check_current_task_success()),
            "instruction": self.get_instruction(),
        }
        return action, info

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None