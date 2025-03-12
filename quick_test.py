import gymnasium as gym
import matplotlib.pyplot as plt
import custom_envs

# Create environment with fixed map seed
seed = 0
env = gym.make(
    "CarRacingFixedMap-v2", continuous=True, render_mode="rgb_array", map_seed=seed
)

# Store frames after reset
frames = []

for i in range(3):
    obs, _ = env.reset(seed=seed)  # Reset with the same seed to ensure reproducibility
    frames.append(obs)
    print(f"Collected frame {i+1}, shape: {obs.shape}")

env.close()

env = gym.make(
    "CarRacingFixedMap-v2", continuous=True, render_mode="rgb_array", map_seed=seed
)

for i in range(3):
    obs, _ = env.reset(seed=seed)  # Reset with the same seed to ensure reproducibility
    frames.append(obs)
    print(f"Collected frame {i+1}, shape: {obs.shape}")

env.close()

# Plot all frames in one figure
fig, axes = plt.subplots(1, 6, figsize=(15, 5))

for i, frame in enumerate(frames):
    axes[i].imshow(frame)
    axes[i].axis("off")
    axes[i].set_title(f"Reset {i+1}")

plt.tight_layout()
plt.show()
