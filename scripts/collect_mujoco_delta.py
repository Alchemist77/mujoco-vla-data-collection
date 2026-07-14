from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.mujoco_rizon_env import ALL_INSTRUCTIONS, MujocoRizonWrapper


HEADLESS_ARG = "--headless"
PRINT_EVERY_ARG = "--print-every"
EPISODES_ARG = "--episodes"
MAX_STEPS_ARG = "--max-steps"
OUTPUT_ARG = "--output"
LOG_ARG = "--log-actions"


class SimpleTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.vocab = [self.pad_token, self.unk_token]
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1}

    def build_from_texts(self, texts):
        for text in texts:
            for tok in self._tokenize(text):
                if tok not in self.token_to_id:
                    self.token_to_id[tok] = len(self.vocab)
                    self.vocab.append(tok)
        return self

    def _tokenize(self, text: str):
        return str(text).lower().strip().split()

    def encode(self, text: str):
        return [self.token_to_id.get(tok, 1) for tok in self._tokenize(text)]


def build_tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer().build_from_texts(ALL_INSTRUCTIONS)


def _parse_int_arg(flag: str, default: int) -> int:
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 >= len(sys.argv):
            raise ValueError(f"Missing value for {flag}")
        return int(sys.argv[idx + 1])
    return int(default)


def _parse_str_arg(flag: str, default: str) -> str:
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 >= len(sys.argv):
            raise ValueError(f"Missing value for {flag}")
        return str(sys.argv[idx + 1])
    return str(default)


def _format_vec(vec) -> str:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return np.array2string(arr, precision=5, floatmode="fixed", suppress_small=False)


def _append_action_log(log_path: str | None, lines: list[str]):
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def run_episode(
    env: MujocoRizonWrapper,
    tokenizer: SimpleTokenizer,
    success_count: int,
    attempts: int,
    max_steps: int,
    print_every: int,
    log_path: str | None,
):
    images = []
    states = []
    actions = []
    text_ids_list = []

    _, _, reset_info = env.reset(seed=42 + attempts, instruction="auto")
    instruction = reset_info.get("instruction", env.get_instruction())
    text_ids = np.array(tokenizer.encode(instruction), dtype=np.int64)
    episode_seed = 42 + attempts

    print(f"\n===== EP {success_count} (attempt {attempts}) =====")
    print(f"instruction: {instruction}")
    print(f"episode_seed: {episode_seed}")
    if "task_color" in reset_info:
        print(
            f"task_color: {reset_info['task_color']} | "
            f"target_slot: {reset_info.get('target_slot', 'n/a')} | "
            f"task_box_index: {reset_info.get('task_box_index')}"
        )

    episode_log = [
        f"===== EP {success_count} (attempt {attempts}) =====",
        f"instruction: {instruction}",
        f"episode_seed: {episode_seed}",
    ]
    _append_action_log(log_path, episode_log)

    for sim_step in range(max_steps):
        image = env._get_image()
        state = env._get_state()

        # 1. teacher가 action 계산
        action, info = env.teacher_step()

        # 2. 그 action을 실제 env에 넣는다 (중요)
        next_image, next_state, reward, done, step_info = env.step(action)

        # 3. 저장 (state_t, action_t, next_state_t+1 구조)
        images.append(np.asarray(image, dtype=np.uint8))
        states.append(np.asarray(state, dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        text_ids_list.append(text_ids)

        # 4. 다음 스텝 준비
        image = next_image
        state = next_state

        should_print = (sim_step < 20) or (print_every > 0 and sim_step % print_every == 0) or bool(info["planner_done"])
        if should_print:
            ee_pos = state[7:10]
            target_pos = info["target_pos"]
            raw_delta = np.asarray(target_pos, dtype=np.float32) - np.asarray(ee_pos, dtype=np.float32)
            lines = [
                f"step={sim_step:05d} phase={info['phase']:<20s}",
                f"  ee_pos        = {_format_vec(ee_pos)}",
                f"  target_pos    = {_format_vec(target_pos)}",
                f"  raw_delta_xyz = {_format_vec(raw_delta)}",
                f"  action_delta  = {_format_vec(action[:3])}",
                f"  grip_action   = {action[3]:+.1f}",
                f"  dist          = {info['dist']:.6f}",
            ]
            print("\n".join(lines))
            _append_action_log(log_path, lines)

        if info["planner_done"]:
            if info["success"]:
                print(f"✅ SUCCESS EP={success_count} attempt={attempts}")
                _append_action_log(log_path, [f"SUCCESS EP={success_count} attempt={attempts}", ""])
                episode_meta = {
                    "seed": episode_seed,
                    "instruction": instruction,
                    "task_box_index": int(reset_info.get("task_box_index", -1)),
                    "place_slot_index": int(reset_info.get("place_slot_index", -1)),
                    "task_color": str(reset_info.get("task_color", "")),
                    "target_slot": str(reset_info.get("target_slot", "")),
                    "steps": len(actions),
                    "success": 1,
                }
                return images, states, actions, text_ids_list, episode_meta, True
            print(f"❌ FAIL EP={success_count} attempt={attempts} (planner done but fail)")
            _append_action_log(log_path, [f"FAIL EP={success_count} attempt={attempts} planner_done_no_success", ""])
            episode_meta = {
                "seed": episode_seed,
                "instruction": instruction,
                "task_box_index": int(reset_info.get("task_box_index", -1)),
                "place_slot_index": int(reset_info.get("place_slot_index", -1)),
                "task_color": str(reset_info.get("task_color", "")),
                "target_slot": str(reset_info.get("target_slot", "")),
                "steps": len(actions),
                "success": 0,
            }
            return images, states, actions, text_ids_list, episode_meta, False

    print(f"❌ FAIL EP={success_count} attempt={attempts} (max_steps reached)")
    _append_action_log(log_path, [f"FAIL EP={success_count} attempt={attempts} max_steps_reached", ""])
    episode_meta = {
        "seed": episode_seed,
        "instruction": instruction,
        "task_box_index": int(reset_info.get("task_box_index", -1)),
        "place_slot_index": int(reset_info.get("place_slot_index", -1)),
        "task_color": str(reset_info.get("task_color", "")),
        "target_slot": str(reset_info.get("target_slot", "")),
        "steps": len(actions),
        "success": 0,
    }
    return images, states, actions, text_ids_list, episode_meta, False


def main_headless(episodes=30, max_steps=10000, print_every=500, output_path="data/mujoco_delta_dataset.npz", log_path=None):
    tokenizer = build_tokenizer()
    env = MujocoRizonWrapper(seed=42, image_size=224, max_episode_steps=max_steps)

    all_images, all_states, all_actions, all_text_ids = [], [], [], []
    episode_meta_list = []
    success_count = 0
    attempts = 0

    while success_count < episodes:
        attempts += 1
        images, states, actions, text_ids_list, episode_meta, success = run_episode(
            env, tokenizer, success_count, attempts, max_steps, print_every, log_path
        )
        if success:
            all_images.extend(images)
            all_states.extend(states)
            all_actions.extend(actions)
            all_text_ids.extend(text_ids_list)
            episode_meta_list.append(episode_meta)
            success_count += 1

    _save(all_images, all_states, all_actions, all_text_ids, tokenizer, episode_meta_list, output_path)
    env.close()


def main_viewer(episodes=20, max_steps=10000, print_every=1, output_path="data/mujoco_delta_dataset.npz", log_path=None):
    tokenizer = build_tokenizer()
    env = MujocoRizonWrapper(seed=42, image_size=224, max_episode_steps=max_steps)

    all_images, all_states, all_actions, all_text_ids = [], [], [], []
    episode_meta_list = []
    success_count = 0
    attempts = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 1.9
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -24
        viewer.cam.lookat[:] = [0.50, -0.05, 0.42]

        while success_count < episodes and viewer.is_running():
            attempts += 1
            _, _, reset_info = env.reset(seed=42 + attempts, instruction="auto")
            instruction = reset_info.get("instruction", env.get_instruction())
            text_ids = np.array(tokenizer.encode(instruction), dtype=np.int64)
            episode_seed = 42 + attempts

            print(f"\n===== EP {success_count} (attempt {attempts}) =====")
            print(f"instruction: {instruction}")
            print(f"episode_seed: {episode_seed}")
            if "task_color" in reset_info:
                print(
                    f"task_color: {reset_info['task_color']} | "
                    f"target_slot: {reset_info.get('target_slot', 'n/a')} | "
                    f"task_box_index: {reset_info.get('task_box_index')}"
                )

            ep_images, ep_states, ep_actions, ep_text_ids = [], [], [], []
            for sim_step in range(max_steps):
                if not viewer.is_running():
                    break

                image = env._get_image()
                state = env._get_state()
                action, info = env.teacher_step()

                ep_images.append(image)
                ep_states.append(state)
                ep_actions.append(action)
                ep_text_ids.append(text_ids)

                ee_pos = state[7:10]
                target_pos = info["target_pos"]
                raw_delta = np.asarray(target_pos, dtype=np.float32) - np.asarray(ee_pos, dtype=np.float32)
                should_print = (sim_step < 20) or (print_every > 0 and sim_step % print_every == 0) or bool(info["planner_done"])
                if should_print:
                    lines = [
                        f"step={sim_step:05d} phase={info['phase']:<20s}",
                        f"  ee_pos        = {_format_vec(ee_pos)}",
                        f"  target_pos    = {_format_vec(target_pos)}",
                        f"  raw_delta_xyz = {_format_vec(raw_delta)}",
                        f"  action_delta  = {_format_vec(action[:3])}",
                        f"  grip_action   = {action[3]:+.1f}",
                        f"  dist          = {info['dist']:.6f}",
                    ]
                    print("\n".join(lines))
                    _append_action_log(log_path, lines)
                viewer.sync()

                if info["planner_done"]:
                    if info["success"]:
                        success_count += 1
                        print(f"✅ SUCCESS EP={success_count} attempt={attempts}")
                        all_images.extend(ep_images)
                        all_states.extend(ep_states)
                        all_actions.extend(ep_actions)
                        all_text_ids.extend(ep_text_ids)
                        episode_meta_list.append(
                            {
                                "seed": episode_seed,
                                "instruction": instruction,
                                "task_box_index": int(reset_info.get("task_box_index", -1)),
                                "place_slot_index": int(reset_info.get("place_slot_index", -1)),
                                "task_color": str(reset_info.get("task_color", "")),
                                "target_slot": str(reset_info.get("target_slot", "")),
                                "steps": len(ep_actions),
                                "success": 1,
                            }
                        )
                    else:
                        print(f"❌ FAIL EP={success_count} attempt={attempts} (planner done but fail)")
                    break

    _save(all_images, all_states, all_actions, all_text_ids, tokenizer, episode_meta_list, output_path)
    env.close()


def _save(images, states, actions, text_ids_list, tokenizer, episode_meta_list, output_path):
    images = np.stack(images, axis=0)
    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)

    max_len = max(t.shape[0] for t in text_ids_list)
    padded = np.zeros((len(text_ids_list), max_len), dtype=np.int64)
    for i, t in enumerate(text_ids_list):
        padded[i, : t.shape[0]] = t
    text_ids = padded

    episode_starts = []
    episode_lengths = []
    cursor = 0
    episode_seeds = []
    episode_instructions = []
    episode_task_box_index = []
    episode_place_slot_index = []
    episode_task_color = []
    episode_target_slot = []

    for meta in episode_meta_list:
        episode_starts.append(cursor)
        episode_lengths.append(int(meta["steps"]))
        cursor += int(meta["steps"])
        episode_seeds.append(int(meta["seed"]))
        episode_instructions.append(str(meta["instruction"]))
        episode_task_box_index.append(int(meta["task_box_index"]))
        episode_place_slot_index.append(int(meta["place_slot_index"]))
        episode_task_color.append(str(meta["task_color"]))
        episode_target_slot.append(str(meta["target_slot"]))

    output_path = str(output_path)
    os.makedirs(str(Path(output_path).resolve().parent), exist_ok=True)
    np.savez_compressed(
        output_path,
        images=images,
        states=states,
        actions=actions,
        text_ids=text_ids,
        vocab=np.array(tokenizer.vocab, dtype=object),
        episode_starts=np.asarray(episode_starts, dtype=np.int64),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
        episode_seeds=np.asarray(episode_seeds, dtype=np.int64),
        episode_instructions=np.asarray(episode_instructions, dtype=object),
        episode_task_box_index=np.asarray(episode_task_box_index, dtype=np.int64),
        episode_place_slot_index=np.asarray(episode_place_slot_index, dtype=np.int64),
        episode_task_color=np.asarray(episode_task_color, dtype=object),
        episode_target_slot=np.asarray(episode_target_slot, dtype=object),
    )

    print("\n===== DONE =====")
    print("save_path:", output_path)
    print("images   :", images.shape)
    print("states   :", states.shape)
    print("actions  :", actions.shape)
    print("text_ids :", text_ids.shape)
    print("episodes :", len(episode_meta_list))
    print("vocab    :", len(tokenizer.vocab), "tokens")


if __name__ == "__main__":
    episodes = _parse_int_arg(EPISODES_ARG, 20)
    max_steps = _parse_int_arg(MAX_STEPS_ARG, 10000)
    print_every = _parse_int_arg(PRINT_EVERY_ARG, 1 if HEADLESS_ARG not in sys.argv else 500)
    output_path = _parse_str_arg(OUTPUT_ARG, "data/mujoco_delta_dataset.npz")
    log_path = _parse_str_arg(LOG_ARG, "") or None

    if HEADLESS_ARG in sys.argv:
        print("=== HEADLESS MODE ===")
        main_headless(episodes=episodes, max_steps=max_steps, print_every=print_every, output_path=output_path, log_path=log_path)
    else:
        print("=== VIEWER MODE ===")
        main_viewer(episodes=episodes, max_steps=max_steps, print_every=print_every, output_path=output_path, log_path=log_path)
