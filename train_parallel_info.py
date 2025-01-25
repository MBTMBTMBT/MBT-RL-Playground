import gc
import os
from multiprocessing import Pool
from typing import List, Dict

import numpy as np
from tqdm import tqdm
import plotly.graph_objs as go

from dyna_q_modernized import Agent
from configs_info import get_envs_discretizers_and_configs
from parallel_training import generate_test_gif


def run_training(task_name: str, env_idx: int, run_id: int,):
    env, test_env, env_desc, state_discretizer, action_discretizer, configs = get_envs_discretizers_and_configs(task_name, env_idx)
    dir_path = os.path.dirname(configs["save_path"])
    os.makedirs(dir_path, exist_ok=True)
    save_path = configs["save_path"] + f"-{env_desc}-id-{run_id}"

    agent = Agent(
        state_discretizer,
        action_discretizer,
        env,
        configs["use_deep_agent"],
        configs["train_max_num_steps_per_episode"],
        configs["initialization_distribution"],
        configs["exploit_agent_lr"],
        configs["exploit_value_decay"],
        configs["exploit_policy_reward_rate"],
    )

    pbar = tqdm(
        total=configs["exploit_policy_training_steps"],
        desc=f"[{run_id}-{env_desc}]",
        unit="step",
        leave=False,
        dynamic_ncols=True,
        smoothing=1.0,
        mininterval=1.0,
        maxinterval=30.0,
    )
    sample_step_count = 0
    test_results = []
    test_steps = []
    final_test_rewards = 0.0

    first_test = True
    while sample_step_count < configs["exploit_policy_training_steps"]:
        if not first_test:
            agent.learn(configs["exploit_policy_test_per_num_steps"], False)
            sample_step_count += configs["exploit_policy_test_per_num_steps"]

        periodic_test_rewards = []
        frames = []
        for t in range(configs["exploit_policy_test_episodes"]):
            test_state, _ = test_env.reset()
            test_total_reward = 0
            test_done = False
            while not test_done:
                test_action = agent.choose_action(test_state, greedy=True)
                test_next_state, test_reward, test_done, test_truncated, _ = test_env.step(test_action)
                if t == 0:
                    frames.append(test_env.render())
                test_state = test_next_state
                test_total_reward += test_reward
                if test_done or test_truncated:
                    break
            periodic_test_rewards.append(test_total_reward)

        avg_test_reward = np.mean(periodic_test_rewards)
        test_results.append(avg_test_reward)
        test_steps.append(sample_step_count)

        # Save GIF for the first test episode
        gif_path = save_path + f"_latest.gif"
        generate_test_gif(frames, gif_path, to_print=configs["print_training_info"])

        first_test = False

        if sample_step_count % configs["save_per_num_steps"] == 0 and sample_step_count > 0:
            agent.save_agent(save_path + f"_{sample_step_count}")

        pbar.set_postfix({
            "Test Rwd": f"{avg_test_reward:04.3f}" if len(f"{int(avg_test_reward)}") <= 6 else f"{avg_test_reward:.3f}",
            "Found Trans": f"{len(agent.transition_table_env.forward_dict):.2e}",
        })
        pbar.update(configs["exploit_policy_test_per_num_steps"])

    agent.save_agent(save_path + f"_final")

    pbar.close()

    del agent, pbar
    del env, test_env, state_discretizer, action_discretizer, configs
    gc.collect()

    return task_name, run_id, env_idx, test_results, test_steps, final_test_rewards


# A wrapper function for unpacking arguments
def run_training_unpack(args):
    return run_training(**args)  # Unpack the dictionary into keyword arguments


# Aggregating results for consistent step-based plotting
def run_all_experiments_and_plot(task_names_and_num_experiments: Dict[str, int], max_workers):
    tasks = []
    run_id = 0
    for task_name, runs in task_names_and_num_experiments.items():
        # Shuffle the sequence just for monitoring more possible cases simultaneously
        num_envs = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)["num_envs"]
        for env_idx in range(num_envs):
            for _ in range(runs):
                tasks.append({
                    "task_name": task_name,
                    "env_idx": env_idx,
                    "run_id": run_id,
                })
                run_id += 1

    print(f"Total tasks: {run_id}.")

    # Execute tasks in parallel
    if max_workers > 1:
        with Pool(processes=max_workers, maxtasksperchild=1) as pool:
            all_results = pool.map(run_training_unpack, tasks)
    else:
        all_results = [run_training_unpack(task) for task in tasks]

    # Aggregate results for each group
    aggregated_results = {}

    # Group results by task_name and init_group
    grouped_results = {}
    for task_name, run_id, env_idx, test_results, test_steps, final_test_rewards in all_results:
        if task_name not in grouped_results:
            grouped_results[task_name] = {}
        if env_idx not in grouped_results[task_name]:
            grouped_results[task_name][env_idx] = {"test_results": [], "test_steps": [], "final_test_rewards": []}

        # Append results to the corresponding group
        grouped_results[task_name][env_idx]["test_results"].append(test_results)
        grouped_results[task_name][env_idx]["test_steps"].append(test_steps)
        grouped_results[task_name][env_idx]["final_test_rewards"].append(final_test_rewards)

    # Aggregate data for each task_name and init_group
    for task_name, env_idxs in grouped_results.items():
        aggregated_results[task_name] = {}
        for env_idx, data in env_idxs.items():
            test_results_array = np.array(data["test_results"])  # Shape: (runs, steps)
            test_steps = data["test_steps"][0]  # Assume all runs share the same test_steps
            final_test_rewards_array = np.array(data["final_test_rewards"])  # Shape: (runs,)

            # Compute mean and std for test_results
            mean_test_results = test_results_array.mean(axis=0)
            std_test_results = test_results_array.std(axis=0)

            # Compute mean and std for final_test_rewards
            mean_final_rewards = final_test_rewards_array.mean()
            std_final_rewards = final_test_rewards_array.std()

            # Store aggregated results
            aggregated_results[task_name][env_idx] = {
                "mean_test_results": mean_test_results.tolist(),
                "std_test_results": std_test_results.tolist(),
                "test_steps": test_steps,
                "mean_final_rewards": mean_final_rewards,
                "std_final_rewards": std_final_rewards,
            }

    # Plot results
    for task_name, env_idxs in aggregated_results.items():
        fig = go.Figure()

        # Use hex color codes instead of names
        colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']  # Hex color codes
        line_styles = ['dash', 'dot', 'longdash', 'dashdot']  # Line styles for final results
        color_idx = 0

        for env_idx in sorted(env_idxs.keys()):  # Sort subtasks alphabetically
            subtask_data = env_idxs[env_idx]
            # Extract aggregated data
            mean_test_results = subtask_data["mean_test_results"]
            std_test_results = subtask_data["std_test_results"]
            test_steps = subtask_data["test_steps"]
            mean_final_rewards = subtask_data["mean_final_rewards"]

            _, _, env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, env_idx)

            # Add mean test results curve
            fig.add_trace(go.Scatter(
                x=test_steps,
                y=mean_test_results,
                mode='lines',
                name=f"{env_desc} Mean Test Results",
                line=dict(color=colors[color_idx], width=2),
            ))

            # Add shaded area for std
            fig.add_trace(go.Scatter(
                x=test_steps + test_steps[::-1],  # Create a filled region
                y=(np.array(mean_test_results) + np.array(std_test_results)).tolist() +
                  (np.array(mean_test_results) - np.array(std_test_results))[::-1].tolist(),
                fill='toself',
                fillcolor=f"rgba({int(colors[color_idx][1:3], 16)}, "
                          f"{int(colors[color_idx][3:5], 16)}, "
                          f"{int(colors[color_idx][5:], 16)}, 0.2)",  # Match line color
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{env_desc} Std Dev",
            ))

            # Add final mean result as a horizontal line
            fig.add_trace(go.Scatter(
                x=[test_steps[0], test_steps[-1]],
                y=[mean_final_rewards, mean_final_rewards],
                mode='lines',
                name=f"{env_desc} Final Mean",
                line=dict(color='black', width=2, dash=line_styles[color_idx % len(line_styles)]),
            ))

            # Add annotation for the final mean result
            fig.add_trace(go.Scatter(
                x=[test_steps[-1]],  # Position at the end of the line
                y=[mean_final_rewards],
                mode='text',
                text=[f"{mean_final_rewards:.2f}"],  # Format the number with two decimals
                textposition="top right",
                showlegend=False  # Do not show in legend
            ))

            color_idx = (color_idx + 1) % len(colors)

        # Customize layout
        fig.update_layout(
            title=f"Results for {task_name}",
            xaxis_title="Steps",
            yaxis_title="Test Results",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,  # High resolution width
            height=800,  # High resolution height
        )

        # Save plot to file using your specified path
        plot_path = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)["save_path"] + ".png"
        fig.write_image(plot_path, scale=2)  # High-resolution PNG
        print(f"Saved plot for {task_name} at {plot_path}")

    return aggregated_results


if __name__ == '__main__':
    run_all_experiments_and_plot(
        task_names_and_num_experiments={
            "frozen_lake": 16,
        },
        max_workers=16,
    )
