"""Minimal keyboard harness for UR5P2PEnv using absolute position targets.

Use number keys 1-6 to move joints positive, Q-W-E-R-T-Y to move negative,
space to hold position, and X to exit. Runs with optional rendering."""

import argparse
import select
import sys
import termios
import tty
from contextlib import contextmanager

import numpy as np

from env import UR5P2PEnv


@contextmanager
def raw_stdin():
    """Switch stdin to raw mode so we can capture single key presses."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key(timeout=0.05):
    """Return a single character if available within timeout, else None."""
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x03":  # Ctrl+C
        raise KeyboardInterrupt
    return ch


def build_keymap(step):
    """Create mapping from key to (joint index, delta)."""
    pos_keys = "123456"
    neg_keys = "qwerty"
    keymap = {}
    for idx, key in enumerate(pos_keys):
        keymap[key] = (idx, step)
    for idx, key in enumerate(neg_keys):
        keymap[key] = (idx, -step)
    return keymap


def main():
    parser = argparse.ArgumentParser(description="Interactive test for UR5P2PEnv")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo viewer")
    parser.add_argument("--step", type=float, default=0.02, help="Per-key joint delta (rad)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = UR5P2PEnv(render_mode=render_mode, seed=args.seed, target_mode="random")
    keymap = build_keymap(args.step)

    print("Keyboard control ready: 1-6 increase joints, Q-W-E-R-T-Y decrease, space hold, X exit")

    obs, info = env.reset()
    print("Initial q pos:", env.data.qpos[:env.nq])
    print("Start info:", info)

    try:
        with raw_stdin():
            q_target = env.q_target.copy() if getattr(env, "q_target", None) is not None else env.data.qpos[:env.nq].copy()
            last_action = np.zeros(env.action_space.shape, dtype=np.float32)
            while True:
                key = read_key()
                if key is None:
                    env.step(q_target - env.data.qpos[:env.nq])
                    continue

                if key.lower() == "x":
                    break
                if key == " ":
                    q_target = env.data.qpos[:env.nq].copy()
                    obs, reward, terminated, truncated, info = env.step(np.zeros_like(last_action))
                    print("Hold. reward=%.4f terminated=%s truncated=%s" % (reward, terminated, truncated))
                    if terminated or truncated:
                        obs, info = env.reset()
                        q_target = env.data.qpos[:env.nq].copy()
                    continue

                if key in keymap:
                    idx, delta = keymap[key]
                    q_target = q_target.copy()
                    q_target[idx] += delta
                    obs, reward, terminated, truncated, info = env.step(q_target - env.data.qpos[:env.nq])
                    q = env.data.qpos[:env.nq].copy()
                    print("Key %s -> joint %d target %.3f rad | reward %.4f" % (key, idx, q_target[idx], reward))
                    if terminated or truncated:
                        print("Episode finished. Resetting...")
                        obs, info = env.reset()
                        q_target = env.data.qpos[:env.nq].copy()
                    continue

                # Unmapped key: keep previous action but print help
                print("Key '%s' ignored. Valid: 1-6, Q-W-E-R-T-Y, space, X" % key)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
