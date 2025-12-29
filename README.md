# UR5 Reinforcement Learning Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WandB](https://img.shields.io/badge/WandB-Experiments-orange)](https://wandb.ai/supernova0417-korea-university/ur5)

Reinforcement Learning for UR5 Robotic Arm Control using PPO, TD3, and SAC algorithms. This project implements and compares on-policy (PPO) and off-policy (TD3, SAC) deep reinforcement learning methods for a 6-DOF robotic manipulation task.

## 🎯 Project Overview

This project trains RL agents to control a UR5 robotic arm to reach random target positions in 3D space.

### 🏆 Best Results

| Algorithm | Success Rate | Episodic Return | Training Steps |
|-----------|--------------|-----------------|----------------|
| **TD3 Baseline** | **92.3%** 🔥 | 337.76 | 1M |
| **PPO v2.1** | **74.9%** | 337.26 | 2M |
| PPO v2 | 62.8% | 336.2 | 2M |
| PPO Tuned | 61.9% | 340.65 | 500k |
| PPO Baseline | 5.7% | 37.16 | 1M |
| SAC v3 | 0.05% ❌ | -10.5 | 1M |

**Key Findings:**
- **TD3 outperforms all other algorithms** with 92.3% success rate!
- PPO achieved ~75% after extensive hyperparameter tuning
- SAC failed to learn in this environment (unstable Q-function)
- **Reward shaping is critical** - 13x improvement from baseline to tuned PPO

## 📁 Repository Structure

```
UR5/
├── env.py                          # Original UR5 Gymnasium environment
├── env_tuned.py                    # Tuned environment (10cm threshold)
├── env_tuned_v3.py                 # PPO v3 environment (8cm threshold)
├── env_tuned_sac.py                # SAC-specific environment
├── env_tuned_sac_v3.py             # SAC v3 environment (simplified rewards)
├── __init__.py                     # Package initialization
├── test_env.py                     # Environment testing script
├── ik.py                           # Inverse kinematics implementation
│
├── assets/                         # MuJoCo model files
│   ├── scene.xml                   # Main scene with UR5 + environment
│   ├── ur5e.xml                    # UR5e robot MJCF model
│   └── meshes/                     # Robot mesh files (.stl)
│
├── cleanrl/cleanrl/                # Training scripts
│   ├── ppo_continuous_action.py           # PPO baseline
│   ├── ppo_continuous_action_tuned.py     # PPO tuned (61.9%)
│   ├── ppo_continuous_action_v2.py        # PPO v2 (62.8%)
│   ├── ppo_continuous_action_v2_1.py      # PPO v2.1 (74.9%) ✨
│   ├── ppo_continuous_action_v3.py        # PPO v3 (8cm, 41.3%)
│   ├── td3_continuous_action.py           # TD3 original
│   ├── td3_continuous_action_ur5.py       # TD3 UR5 (92.3%) 🔥
│   ├── sac_continuous_action.py           # SAC original
│   ├── sac_continuous_action_modified.py  # SAC adapted for UR5
│   ├── sac_continuous_action_tuned.py     # SAC tuned
│   ├── sac_continuous_action_v2.py        # SAC v2
│   ├── sac_continuous_action_v3.py        # SAC v3
│   └── ddpg_continuous_action.py          # DDPG implementation
│
├── runs/                           # TensorBoard training logs
├── wandb/                          # Weights & Biases experiment logs
└── videos/                         # Recorded episode videos
```

## 🖥️ Test Environment

| Component | Specification |
|-----------|---------------|
| OS | Windows 10 |
| CPU | Intel Core i7-7700HQ @ 2.80GHz |
| GPU | NVIDIA GeForce GTX 1060 (6GB) |
| Python | 3.10.19 |
| PyTorch | 2.4.1 (CUDA) |

## 📦 Installation

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
```

## 🚀 Usage

### Test Environment
```bash
python test_env.py
```

### Training

**TD3 (Best Performance - 92.3%):**
```bash
python cleanrl/cleanrl/td3_continuous_action_ur5.py
```

**PPO v2.1 (74.9%):**
```bash
python cleanrl/cleanrl/ppo_continuous_action_v2_1.py
```

**PPO Tuned (61.9%):**
```bash
python cleanrl/cleanrl/ppo_continuous_action_tuned.py
```

### Monitoring
- **TensorBoard:** `tensorboard --logdir runs/`
- **WandB:** https://wandb.ai/supernova0417-korea-university/ur5

## ⚙️ Hyperparameter Configurations

### TD3 Baseline (Best)
```python
learning_rate = 3e-4
learning_starts = 25000
policy_noise = 0.2
exploration_noise = 0.1
tau = 0.005
batch_size = 256
total_timesteps = 1000000
```

### PPO v2.1 (Best PPO)
```python
learning_rate = 5e-5
ent_coef = 0.003  # Key tuning point!
num_envs = 8
num_steps = 2048
total_timesteps = 2000000
```

### Reward Function (env_tuned.py)
```python
reward_cfg = {
    "dist_success_thresh": 0.10,  # 10cm
    "time_penalty": 0.001,
    "success_bonus": 300.0,
    "progress_scale": 15.0,
}
```

## 📊 Experiment Results Summary

### Algorithm Comparison

| Algorithm | Version | Success Rate | Key Changes |
|-----------|---------|--------------|-------------|
| **TD3** | Baseline | **92.3%** | Twin Q, delayed policy |
| PPO | v2.1 | 74.9% | ent_coef=0.003 |
| PPO | v2 | 62.8% | 2M steps, 8 envs |
| PPO | Tuned | 61.9% | Reward shaping |
| PPO | v3 | 41.3% | 8cm threshold (too hard) |
| PPO | Baseline | 5.7% | Original settings |
| SAC | v3 | 0.05% | Failed (Q divergence) |
| SAC | Baseline | 0% | Failed |

### Key Insights

1. **TD3 is the best algorithm for this task**
   - Twin Q-networks prevent overestimation
   - Delayed policy updates ensure stability
   - SAC's entropy term caused instability

2. **PPO converges around ~75%**
   - `ent_coef=0.003` is optimal (between 0.005 and 0.001)
   - Further tuning shows diminishing returns

3. **SAC fundamentally fails in this environment**
   - Q-function diverges to infinity
   - Entropy regularization seems problematic

4. **Reward shaping matters significantly**
   - 10cm threshold is achievable
   - 8cm threshold causes major performance drop
   - Progress-based rewards are essential

## 📚 References

1. Fujimoto et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods. (TD3)
2. Schulman et al. (2017). Proximal Policy Optimization Algorithms. (PPO)
3. Haarnoja et al. (2018). Soft Actor-Critic. (SAC)
4. [CleanRL](https://github.com/vwxyzjn/cleanrl) - Clean implementation of RL algorithms.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Korea University - MECH485 Intelligent Robotics Course
- [CleanRL](https://github.com/vwxyzjn/cleanrl) for the excellent RL implementations
- [MuJoCo](https://mujoco.org/) for the physics simulation

---

**Author:** Jinkwon Park  
**Course:** MECH485 - Intelligent Robotics, Korea University  
**Date:** 28, December, 2025
