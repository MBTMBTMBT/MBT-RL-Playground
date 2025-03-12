import gymnasium as gym
import numpy as np
import json
import imageio
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from collections import defaultdict


# Encoding and Decoding functions for state storage and retrieval
def encode_state(env):
    """Encode the current environment state into a hashable JSON string."""
    env = env.unwrapped

    # Ensure deterministic ordering
    grid_array = env.grid.encode()
    grid_list = np.array(grid_array, dtype=np.uint8).tolist()  # Standardize format

    # Ensure integer position (no float conversion issues)
    agent_pos = tuple(map(int, env.agent_pos))
    agent_dir = int(env.agent_dir)

    # Standardize carried object encoding
    carrying = None
    if env.carrying is not None:
        carrying = tuple(map(int, env.carrying.encode()))  # Standardized integer tuple

    # Ensure JSON ordering is stable
    state = {
        "grid": grid_list,
        "agent_pos": agent_pos,
        "agent_dir": agent_dir,
        "carrying": carrying,
    }
    return json.dumps(state, sort_keys=True)  # Ensure stable JSON ordering


def decode_state(env, state_str):
    """Restore environment from encoded JSON string."""
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


# MDP Exploration with Correct Reset Mechanism
def explore_environment(env_name, seed=0):
    """Fully explore the environment and build the transition table."""
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    transition_table = defaultdict(lambda: defaultdict(str))
    rewards = defaultdict(lambda: defaultdict(float))

    explored_states = set()  # Fully explored states
    new_states = set([encode_state(env)])  # Newly discovered states

    while new_states:
        state_str = new_states.pop()
        if state_str in explored_states:
            continue

        decode_state(env, state_str)  # Restore environment to this state

        unvisited_actions = 0  # Track number of unexplored actions

        for action in range(env.action_space.n):
            prev_state_str = encode_state(env)  # Save the current state
            obs, reward, done, truncated, _ = env.step(action)
            next_state_str = encode_state(env)

            transition_table[prev_state_str][action] = next_state_str
            rewards[prev_state_str][action] = reward

            # If the new state is not explored and is not terminal, add to new_states
            if next_state_str not in explored_states and not done and not truncated:
                if next_state_str not in new_states:
                    new_states.add(next_state_str)
                    unvisited_actions += (
                        1  # Count how many actions lead to unknown states
                    )

            decode_state(env, prev_state_str)  # Restore the original state

        # Only move to `explored_states` if all actions have been fully explored
        if unvisited_actions == 0:
            explored_states.add(state_str)

        print(f"New states: {len(new_states)}, Explored states: {len(explored_states)}")

    env.close()
    return transition_table, rewards


# Value Iteration for MDP Solving
def value_iteration(
    transition_table, rewards, gamma=0.95, theta=1e-5, max_iterations=10000000
):
    """Solve the MDP using Value Iteration algorithm."""
    states = list(transition_table.keys())
    V = {s: 0 for s in states}  # Initialize values to zero

    for iteration in range(max_iterations):  # Limit iteration count
        delta = 0
        for s in states:
            if s not in transition_table or not transition_table[s]:
                continue  # Skip terminal states

            max_value = float("-inf")
            for a in transition_table[s]:
                next_s = transition_table[s][a]
                reward = rewards[s][a]
                value = reward + gamma * V.get(next_s, 0)

                max_value = max(max_value, value)

            delta = max(delta, abs(V[s] - max_value))
            V[s] = max_value

        if delta < theta:
            break

    # Extract policy
    policy = {}
    for s in states:
        if s not in transition_table or not transition_table[s]:
            continue  # Skip terminal states

        best_action = max(
            transition_table[s],
            key=lambda a: rewards[s][a] + gamma * V.get(transition_table[s][a], 0),
        )
        policy[s] = best_action

    return policy


# Execute the Optimal Policy and Save as GIF
def execute_policy(env_name, policy, seed=0, output_gif="optimal_policy.gif"):
    """Execute the optimal policy and record the trajectory."""
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    state_str = encode_state(env)
    frames = [env.render()]

    while state_str in policy:
        action = policy[state_str]
        obs, reward, done, truncated, _ = env.step(action)
        frames.append(env.render())

        state_str = encode_state(env)
        if done or truncated:
            break

    env.close()

    # Save as GIF
    imageio.mimsave(output_gif, frames, duration=0.2)


# Solve MiniGrid-DoorKey-5x5-v0
transition_table_door, rewards_door = explore_environment(
    "MiniGrid-DoorKey-5x5-v0", seed=0
)
policy_door = value_iteration(transition_table_door, rewards_door)
execute_policy(
    "MiniGrid-DoorKey-5x5-v0", policy_door, seed=0, output_gif="doorkey_solution.gif"
)

# Solve MiniGrid-Dynamic-Obstacles-5x5-v0
transition_table_obstacles, rewards_obstacles = explore_environment(
    "MiniGrid-Dynamic-Obstacles-5x5-v0", seed=0
)
policy_obstacles = value_iteration(transition_table_obstacles, rewards_obstacles)
execute_policy(
    "MiniGrid-Dynamic-Obstacles-5x5-v0",
    policy_obstacles,
    seed=0,
    output_gif="dynamic_obstacles_solution.gif",
)
