import numpy as np
import torch
from ppo_continuous_action import Agent, Args, make_env
from env import UR5P2PEnv  # 또는 import UR5 후 gym.make 사용
import gymnasium as gym

args = Args()
args.env_id = "UR5-v0"
args.target_mode = "random"
args.target_pos = None
device = torch.device("cpu")

# GUI가 필요한 경우 직접 환경 생성
env = UR5P2PEnv(render_mode="human", seed=None, target_mode=args.target_mode, target_pos=args.target_pos)
obs, info = env.reset()


def augment(obs_array):
    """Match training observation by appending goal position."""
    goal = env.goal_pos if getattr(env, "goal_pos", None) is not None else np.zeros(3, dtype=np.float32)
    return np.concatenate([obs_array.astype(np.float32), goal.astype(np.float32)]).astype(np.float32)


obs_aug = augment(obs)

# Agent 초기화 및 가중치 로드
dummy_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, False, "eval-gui", args.gamma, args=args)])
agent = Agent(dummy_envs)
agent.load_state_dict(torch.load("runs/UR5-v0__ppo_continuous_action__1__1759933101/ppo_continuous_action.cleanrl_model", map_location=device))
agent.eval()

done = False

while not done:
    obs_t = torch.as_tensor(obs_aug, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action, _, _, _ = agent.get_action_and_value(obs_t)
    obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
    obs_aug = augment(obs)
    done = terminated or truncated
    env.render()  # MuJoCo GUI 업데이트

env.close()
