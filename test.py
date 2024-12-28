import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
env = FlatObsWrapper(env)
observation, info = env.reset(seed=42)
print(observation.shape)
