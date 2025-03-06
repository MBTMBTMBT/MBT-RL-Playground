import gymnasium as gym
import numpy as np
import json
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj


def encode_state(env):
    env = env.unwrapped  # Unwrap to access internal attributes

    # Extract grid encoding (3D list)
    grid_array = env.grid.encode()  # numpy array shape=(W,H,3)
    grid_list = grid_array.tolist()  # Convert to list for JSON serialization

    # Extract agent position and direction
    agent_pos = tuple(int(x) for x in env.agent_pos)
    agent_dir = int(env.agent_dir)

    # Extract carried object (if any)
    carrying = None
    if env.carrying is not None:
        carrying = tuple(env.carrying.encode())  # Object type, color, state

    # Extract random state (optional)
    rng_state = None
    try:
        rng_state = env.np_random.bit_generator.state
    except AttributeError:
        rng_state = env.np_random.get_state()

    # Extract step count (optional)
    step_count = env.step_count if hasattr(env, 'step_count') else None

    # Assemble dictionary and serialize to JSON
    state = {
        "grid": grid_list,
        "agent_pos": agent_pos,
        "agent_dir": agent_dir,
        "carrying": carrying,
        "step_count": step_count,
        "rng_state": rng_state
    }
    return json.dumps(state)


def decode_state(env, state_str):
    env = env.unwrapped  # Unwrap the environment

    data = json.loads(state_str)

    # Reconstruct the grid
    grid_array = np.array(data["grid"], dtype=np.uint8)
    grid, _ = Grid.decode(grid_array)
    env.grid = grid

    # Restore agent position and direction
    env.agent_pos = np.array(data["agent_pos"])
    env.agent_dir = data["agent_dir"]

    # Restore carried object
    env.carrying = None
    if data.get("carrying") is not None:
        t, c, s = data["carrying"]
        carry_obj = WorldObj.decode(t, c, s)
        carry_obj.cur_pos = np.array([-1, -1])  # Mark object as carried
        env.carrying = carry_obj

    # Restore step count
    if data.get("step_count") is not None:
        env.step_count = data["step_count"]

    # Restore random state
    if data.get("rng_state") is not None:
        try:
            env.np_random.bit_generator.state = data["rng_state"]
        except AttributeError:
            env.np_random.set_state(data["rng_state"])


# Example usage:
env = gym.make("MiniGrid-MultiRoom-N6-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)  # Set seed for reproducibility

# Perform some actions to modify state
env.step(2)  # Move forward
env.step(5)  # Open door or interact

# Encode current state
state_str = encode_state(env)
print("Encoded length:", len(state_str))
print("State hash:", __import__("hashlib").sha256(state_str.encode()).hexdigest()[:16])

# Create a new environment instance and restore state
env2 = gym.make("MiniGrid-MultiRoom-N6-v0", render_mode="rgb_array")
obs2, info2 = env2.reset()  # Initialize new environment
decode_state(env2, state_str)  # Restore previous state

# Validate restoration
assert env2.unwrapped.agent_dir == env.unwrapped.agent_dir
assert np.array_equal(env2.unwrapped.agent_pos, env.unwrapped.agent_pos)
assert env2.unwrapped.carrying.encode() == env.unwrapped.carrying.encode() if env.unwrapped.carrying else env2.unwrapped.carrying is env.unwrapped.carrying
assert env2.unwrapped.grid == env.unwrapped.grid  # Use Grid.__eq__ for comparison

print("State successfully restored!")
