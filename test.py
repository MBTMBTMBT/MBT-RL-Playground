import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid

# List of environments and seeds
seeds = [i for i in range(50)]
environments = [
    ("MiniGrid-Dynamic-Obstacles-5x5-v0", seeds, "rgb_array"),
    ("MiniGrid-Dynamic-Obstacles-6x6-v0", seeds, "rgb_array"),
    ("MiniGrid-Dynamic-Obstacles-8x8-v0", seeds, "rgb_array"),
    ("MiniGrid-Dynamic-Obstacles-16x16-v0", seeds, "rgb_array"),
    ("CarRacing-v2", seeds, "rgb_array"),
    ("BipedalWalker-v3", seeds, "rgb_array"),
]

# Create a large canvas for plotting
fig, axes = plt.subplots(nrows=len(environments), ncols=len(seeds), figsize=(256, 60))
fig.suptitle("States of Different Environments and Seeds", fontsize=12)

for row, (env_name, seeds, render_mode) in enumerate(environments):
    for col, seed in enumerate(seeds):
        # Create environment with the correct render mode

        if "BipedalWalker" in env_name:
            env = gym.make(env_name, render_mode=render_mode, hardcore=True)
        else:
            env = gym.make(env_name, render_mode=render_mode)

        # Reset environment with the specific init_seed
        obs, _ = env.reset(seed=seed)

        # Capture the rendered image
        img = env.render()

        # Plot the image
        ax = axes[row][col]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{env_name}\nSeed: {seed}")

        env.close()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to leave space for the title
plt.savefig("environments_with_different_seeds.png")  # Save to a file
plt.close()
