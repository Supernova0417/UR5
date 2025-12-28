# UR5 Reinforcement Learning Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WandB](https://img.shields.io/badge/WandB-Experiments-orange)](https://wandb.ai/supernova0417-korea-university/ur5)

Reinforcement Learning for UR5 Robotic Arm Control using PPO and SAC algorithms. This project implements and compares on-policy (PPO) and off-policy (SAC) deep reinforcement learning methods for a 6-DOF robotic manipulation task.

## 🎯 Project Overview

This project trains RL agents to control a UR5 robotic arm to reach random target positions in 3D space. Key achievements:

| Algorithm | Success Rate | Episodic Return |
|-----------|--------------|-----------------|
| PPO Baseline | 5.7% | 37.16 |
| **PPO Tuned** | **61.95%** ✨ | **340.65** |
| SAC Baseline | 0% | 0.176 |
| SAC Tuned | 0% | 3.03 |

**Key Finding:** Reward shaping is critical! A 10.9x improvement was achieved by tuning the reward function.

## 📁 Repository Structure

```
UR5/
├── env.py                          # Original UR5 Gymnasium environment
├── env_tuned.py                    # Tuned environment with modified rewards
├── __init__.py                     # Package initialization, registers UR5-v0
├── test_env.py                     # Environment testing script
├── ik.py                           # Inverse kinematics implementation
├── ik_solver.py                    # IK solver utilities
│
├── assets/                         # MuJoCo model files
│   ├── scene.xml                   # Main scene with UR5 + environment
│   ├── ur5e.xml                    # UR5e robot MJCF model
│   └── meshes/                     # Robot mesh files (.stl)
│
├── cleanrl/                        # CleanRL-based training scripts
│   └── cleanrl/
│       ├── ppo_continuous_action.py         # PPO baseline
│       ├── ppo_continuous_action_tuned.py   # PPO with tuned hyperparameters
│       ├── sac_continuous_action.py         # Original SAC
│       ├── sac_continuous_action_modified.py # SAC adapted for UR5
│       ├── sac_continuous_action_tuned.py   # SAC with tuned hyperparameters
│       ├── ddpg_continuous_action.py        # DDPG implementation
│       └── td3_continuous_action.py         # TD3 implementation
│   └── cleanrl_utils/
│       ├── buffers.py              # Replay buffer implementations
│       └── evals/                  # Evaluation utilities
│
├── runs/                           # TensorBoard training logs
│   ├── UR5-v0__ppo_continuous_action__*/
│   ├── UR5-v0__sac_continuous_action_modified__*/
│   ├── UR5-tuned-v0__ppo_continuous_action_tuned__*/
│   └── UR5-tuned-v0__sac_continuous_action_tuned__*/
│
├── wandb/                          # Weights & Biases experiment logs
│
├── videos/                         # Recorded episode videos
│
└── reference papers/               # Research papers (PPO, SAC, TD3)
```

### Key Files Description

| File | Description |
|------|-------------|
| `env.py` | Custom Gymnasium environment for UR5 P2P reaching task. Includes reward function, MuJoCo physics, collision detection. |
| `env_tuned.py` | Modified environment with relaxed success threshold (10cm), reduced time penalty, increased progress reward. |
| `ppo_continuous_action.py` | PPO implementation from CleanRL, adapted for UR5 with WandB logging. |
| `sac_continuous_action_modified.py` | SAC implementation adapted for UR5 environment with success rate tracking. |
| `*_tuned.py` | Tuned versions using `env_tuned.py` with optimized hyperparameters. |

## 🖥️ Test Environment

| Component | Specification |
|-----------|---------------|
| OS | Windows 10 |
| CPU | Intel Core i7-7700HQ @ 2.80GHz |
| GPU | NVIDIA GeForce GTX 1060 (6GB) |
| Python | 3.10.19 |
| PyTorch | 2.4.1 (CUDA) |

## 📦 Dependencies

### Core Requirements
```
gymnasium>=0.29.0
mujoco>=3.0.0
dm_control>=1.0.0
torch>=2.0.0
numpy>=1.24.0
wandb>=0.23.0
tyro>=0.5.0
tensorboard>=2.14.0
```

### Full Installation
```bash
# Create conda environment
conda create -n ur5 python=3.10 -y
conda activate ur5

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install MuJoCo and Gymnasium
pip install mujoco gymnasium[mujoco] dm_control

# Install RL utilities
pip install wandb tyro tensorboard

# Install CleanRL utilities (optional)
pip install stable-baselines3
```

## 🚀 Installation & Usage (Windows 10)

### 1. Clone the Repository
```bash
git clone https://github.com/Supernova0417/UR5.git
cd UR5
```

### 2. Setup Environment
```bash
# Create and activate conda environment
conda create -n ur5 python=3.10 -y
conda activate ur5

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mujoco gymnasium dm_control wandb tyro tensorboard
```

### 3. Test Environment
```bash
python test_env.py
```
If successful, you should see the MuJoCo viewer with the UR5 robot.

### 4. Run Training

**PPO Training (Baseline):**
```bash
python cleanrl/cleanrl/ppo_continuous_action.py --total-timesteps 1000000 --track
```

**PPO Training (Tuned - Recommended):**
```bash
python cleanrl/cleanrl/ppo_continuous_action_tuned.py --total-timesteps 500000 --track
```

**SAC Training (Tuned):**
```bash
python cleanrl/cleanrl/sac_continuous_action_tuned.py --total-timesteps 500000 --track
```

### 5. Monitor Training
- **TensorBoard:** `tensorboard --logdir runs/`
- **WandB:** https://wandb.ai/supernova0417-korea-university/ur5

## ⚙️ Reward Function Configuration

### Original (Baseline)
```python
reward_cfg = {
    "dist_success_thresh": 0.03,  # 3cm - too strict
    "time_penalty": 0.01,
    "success_bonus": 100.0,
    "progress_scale": 5.0,
}
```

### Tuned (Recommended)
```python
reward_cfg = {
    "dist_success_thresh": 0.10,  # 10cm - achievable
    "time_penalty": 0.001,        # 10x reduction
    "success_bonus": 300.0,       # 3x increase
    "progress_scale": 15.0,       # 3x increase
}
```

## 📊 Experiment Results

All experiments are logged to WandB:
- **Dashboard:** https://wandb.ai/supernova0417-korea-university/ur5
- **PPO Baseline:** [Run a91933jf](https://wandb.ai/supernova0417-korea-university/ur5/runs/a91933jf)
- **PPO Tuned:** [Run ggqikjdg](https://wandb.ai/supernova0417-korea-university/ur5/runs/ggqikjdg)
- **SAC Baseline:** [Run yyz7lo7a](https://wandb.ai/supernova0417-korea-university/ur5/runs/yyz7lo7a)
- **SAC Tuned:** [Run jxhds5u9](https://wandb.ai/supernova0417-korea-university/ur5/runs/jxhds5u9)

## 🔮 Future Work

- [ ] Achieve higher success rate with PPO (target: >80%)
- [ ] Get SAC to achieve meaningful success rate (target: >50%)
- [ ] Implement curriculum learning for tighter thresholds
- [ ] Add orientation control for full 6-DOF pose reaching
- [ ] Sim-to-real transfer to physical UR5

## 📚 References

1. Schulman et al. (2017). Proximal Policy Optimization Algorithms.
2. Haarnoja et al. (2018). Soft Actor-Critic.
3. [CleanRL](https://github.com/vwxyzjn/cleanrl) - Clean implementation of RL algorithms.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Korea University - MECH485 Intelligent Robotics Course
- [CleanRL](https://github.com/vwxyzjn/cleanrl) for the excellent RL implementations
- [MuJoCo](https://mujoco.org/) for the physics simulation

---

**Author:** Jinkwon Park (2020170644)  
**Course:** MECH485 - Intelligent Robotics, Korea University  
**Date:** December 2025
