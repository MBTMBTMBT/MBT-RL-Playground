import gc
import os
from multiprocessing import Pool
from typing import List, Dict

import numpy as np
from gymnasium import spaces
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
    frames = []
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
        final_test_rewards = avg_test_reward
        test_results.append(avg_test_reward)
        test_steps.append(sample_step_count)

        first_test = False

        if sample_step_count % configs["save_per_num_steps"] == 0 and sample_step_count > 0:
            agent.save_agent(save_path + f"_{sample_step_count}")

        pbar.set_postfix({
            "Test Rwd": f"{avg_test_reward:04.3f}" if len(f"{int(avg_test_reward)}") <= 6 else f"{avg_test_reward:.3f}",
            "Found Trans": f"{len(agent.transition_table_env.forward_dict):.2e}",
        })
        pbar.update(configs["exploit_policy_test_per_num_steps"])

    agent.save_agent(save_path + f"_final")

    # Save GIF for the first test episode
    if len(frames) > 0:
        gif_path = save_path + f"_final.gif"
        try:
            generate_test_gif(frames, gif_path, to_print=configs["print_training_info"])
        except Exception as e:
            print(f"Error generating GIF: {e}")

    pbar.close()

    del agent, pbar
    del env, test_env, state_discretizer, action_discretizer, configs
    gc.collect()

    return task_name, run_id, env_idx, test_results, test_steps, final_test_rewards


def compute_discrete_kl_divergence(weighted_distribution: np.ndarray, default_distribution: np.ndarray) -> float:
    """
    Compute the KL divergence between two distributions for a discrete action space.

    :param weighted_distribution: Weighted action distribution (e.g., greedy-weighted).
    :param default_distribution: Default action distribution (e.g., uniform or another baseline).
    :return: KL divergence value.
    """
    assert weighted_distribution.shape == default_distribution.shape, \
        "Distributions must have the same shape."

    # Avoid division by zero and log(0) by adding a small constant
    epsilon = 1e-4
    weighted_distribution = np.clip(weighted_distribution, epsilon, 1.0)
    default_distribution = np.clip(default_distribution, epsilon, 1.0)

    # Compute KL divergence
    kl_divergence = np.sum(weighted_distribution * np.log2(weighted_distribution / default_distribution))
    return kl_divergence


def run_eval(task_name: str, env_idx: int, run_id: int):
    env, test_env, env_desc, state_discretizer, action_discretizer, configs = get_envs_discretizers_and_configs(
        task_name, env_idx)
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
        configs["use_balanced_random_init"],
    )

    agent.load_agent(save_path + f"_final")

    if configs["use_deep_agent"]:
        agent.exploit_agent.policy.to("cpu")

    kl_weight = 1.0
    action_space = agent.action_discretizer.get_gym_space()
    if isinstance(action_space, spaces.Discrete):
        kl_weight = 1.0 / np.log2(action_space.n)
    else:
        pass  # not implemented yet

    weights = np.linspace(0, 1, 100)

    pbar = tqdm(
        total=len(weights),
        desc=f"[{run_id}-{env_desc}]",
        leave=False,
        dynamic_ncols=True,
    )

    test_results = []
    test_weights = []
    test_control_infos = []
    test_free_energies = []
    for p in weights:
        periodic_test_rewards = []
        periodic_test_control_infos = []
        periodic_test_free_energies = []
        for t in range(configs["exploit_policy_eval_episodes"]):
            test_state, _ = test_env.reset()
            test_total_reward = 0
            test_done = False
            trajectory = [test_state]
            while not test_done:
                test_action = agent.choose_action_by_weight(test_state, p=p)
                test_next_state, test_reward, test_done, test_truncated, _ = test_env.step(test_action)
                trajectory.append(test_next_state)
                test_state = test_next_state
                test_total_reward += test_reward
                if test_done or test_truncated:
                    break
            periodic_test_rewards.append(test_total_reward)
            trajectory.reverse()
            # value = test_total_reward
            # free_energy = 0.0
            control_info = 0.0
            if isinstance(action_space, spaces.Discrete):
                for state in trajectory:
                    weighted_action_distribution = agent.get_greedy_weighted_action_distribution(state, p)
                    default_action_distribution = agent.get_default_policy_distribution(state)
                    kl_divergence = compute_discrete_kl_divergence(
                        weighted_action_distribution, default_action_distribution
                    )
                    control_info += kl_weight * kl_divergence
                    # free_energy += kl_weight * kl_divergence - value
                    # value *= configs["exploit_value_decay"]
            else:
                pass
            # free_energy = control_info - len(trajectory) * value
            periodic_test_control_infos.append(control_info)
            # periodic_test_free_energies.append(free_energy)

        avg_test_reward = np.mean(periodic_test_rewards)
        avg_test_control_info = np.mean(periodic_test_control_infos)
        for control_info, test_total_reward in zip(periodic_test_control_infos, periodic_test_rewards):
            periodic_test_free_energies.append(
                (control_info - configs["train_max_num_steps_per_episode"] * test_total_reward)  / configs["train_max_num_steps_per_episode"]
            )
        avg_test_free_energy = np.mean(periodic_test_free_energies)
        test_results.append(avg_test_reward)
        test_control_infos.append(avg_test_control_info)
        test_free_energies.append(avg_test_free_energy)
        test_weights.append(p)

        pbar.set_postfix({
            "Weight": f"{p:.2f}",
            "Rwd": f"{avg_test_reward:05.3f}",
            "CI": f"{avg_test_control_info:05.1f}",
            "FE": f"{avg_test_free_energy:05.1f}",
        })
        pbar.update(1)

    pbar.close()

    return task_name, run_id, env_idx, test_results, test_control_infos, test_free_energies, test_weights


# A wrapper function for unpacking arguments
def run_training_unpack(args):
    return run_training(**args)  # Unpack the dictionary into keyword arguments

def run_eval_unpack(args):
    return run_eval(**args)


# Aggregating results for consistent step-based plotting
def run_all_trainings_and_plot(task_names_and_num_experiments: Dict[str, int], max_workers):
    tasks = []
    run_id = 0
    for task_name, runs in task_names_and_num_experiments.items():
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
        colors = [
            '#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        ]  # Hex color codes
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


def run_all_evals_and_plot(task_names_and_num_experiments: Dict[str, int], max_workers):
    tasks = []
    run_id = 0
    for task_name, runs in task_names_and_num_experiments.items():
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
            all_results = pool.map(run_eval_unpack, tasks)
    else:
        all_results = [run_eval_unpack(task) for task in tasks]

    # Aggregate results for each group
    aggregated_results = {}

    # Group results by task_name and env_idx
    grouped_results = {}
    for task_name, run_id, env_idx, test_results, test_control_infos, test_free_energies, test_weights in all_results:
        if task_name not in grouped_results:
            grouped_results[task_name] = {}
        if env_idx not in grouped_results[task_name]:
            grouped_results[task_name][env_idx] = {
                "test_results": [],
                "test_control_infos": [],
                "test_free_energies": [],
                "test_weights": []
            }

        # Append results to the corresponding group
        grouped_results[task_name][env_idx]["test_results"].append(test_results)
        grouped_results[task_name][env_idx]["test_control_infos"].append(test_control_infos)
        grouped_results[task_name][env_idx]["test_free_energies"].append(test_free_energies)
        grouped_results[task_name][env_idx]["test_weights"].append(test_weights)

    # Aggregate data for each task_name and env_idx
    for task_name, env_idxs in grouped_results.items():
        aggregated_results[task_name] = {}
        for env_idx, data in env_idxs.items():
            test_results_array = np.array(data["test_results"])  # Shape: (runs, weights)
            test_control_infos_array = np.array(data["test_control_infos"])  # Shape: (runs, weights)
            test_free_energies_array = np.array(data["test_free_energies"])  # Shape: (runs, weights)
            test_weights = data["test_weights"][0]  # Assume all runs share the same weights

            # Compute mean and std for test_results and test_free_energies
            mean_test_results = test_results_array.mean(axis=0)
            std_test_results = test_results_array.std(axis=0)

            mean_test_control_infos = test_control_infos_array.mean(axis=0)
            std_test_control_infos = test_control_infos_array.std(axis=0)

            mean_test_free_energies = test_free_energies_array.mean(axis=0)
            std_test_free_energies = test_free_energies_array.std(axis=0)

            # Store aggregated results
            aggregated_results[task_name][env_idx] = {
                "mean_test_results": mean_test_results.tolist(),
                "std_test_results": std_test_results.tolist(),
                "mean_test_control_infos": mean_test_control_infos.tolist(),
                "std_test_control_infos": std_test_control_infos.tolist(),
                "mean_test_free_energies": mean_test_free_energies.tolist(),
                "std_test_free_energies": std_test_free_energies.tolist(),
                "test_weights": test_weights,
            }

    # Plot results
    for task_name, env_idxs in aggregated_results.items():
        fig = go.Figure()

        # Use hex color codes for consistency
        colors = [
            '#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        ]  # Hex color codes  # Hex color codes
        color_idx = 0

        for env_idx in sorted(env_idxs.keys()):  # Sort subtasks alphabetically
            subtask_data = env_idxs[env_idx]
            # Extract aggregated data
            mean_test_results = subtask_data["mean_test_results"]
            std_test_results = subtask_data["std_test_results"]
            test_weights = subtask_data["test_weights"]

            _, _, env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, env_idx)

            # Add mean test results curve
            fig.add_trace(go.Scatter(
                x=test_weights,
                y=mean_test_results,
                mode='lines',
                name=f"{env_desc} Mean Test Results",
                line=dict(color=colors[color_idx], width=2),
            ))

            # Add shaded area for std
            fig.add_trace(go.Scatter(
                x=test_weights + test_weights[::-1],  # Create a filled region
                y=(np.array(mean_test_results) + np.array(std_test_results)).tolist() +
                  (np.array(mean_test_results) - np.array(std_test_results))[::-1].tolist(),
                fill='toself',
                fillcolor=f"rgba({int(colors[color_idx][1:3], 16)}, "
                          f"{int(colors[color_idx][3:5], 16)}, "
                          f"{int(colors[color_idx][5:], 16)}, 0.2)",  # Match line color
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{env_desc} Std Dev",
            ))

            color_idx = (color_idx + 1) % len(colors)

        # Customize layout
        fig.update_layout(
            title=f"Evaluation Results for {task_name}",
            xaxis_title="Weight (p)",
            yaxis_title="Average Test Reward",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,  # High resolution width
            height=800,  # High resolution height
        )

        # Save plot to file
        plot_path = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)[
                        "save_path"] + "_eval.png"
        fig.write_image(plot_path, scale=2)  # High-resolution PNG
        print(f"Saved evaluation plot for {task_name} at {plot_path}")

        # Plot Control Information
        fig_control_info = go.Figure()
        color_idx = 0

        for env_idx in sorted(env_idxs.keys()):
            subtask_data = env_idxs[env_idx]
            mean_test_control_infos = subtask_data["mean_test_control_infos"]
            std_test_control_infos = subtask_data["std_test_control_infos"]
            test_weights = subtask_data["test_weights"]

            _, _, env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, env_idx)

            fig_control_info.add_trace(go.Scatter(
                x=test_weights,
                y=mean_test_control_infos,
                mode='lines',
                name=f"{env_desc} Mean Control Information",
                line=dict(color=colors[color_idx], width=2),
            ))

            fig_control_info.add_trace(go.Scatter(
                x=test_weights + test_weights[::-1],
                y=(np.array(mean_test_control_infos) + np.array(std_test_control_infos)).tolist() +
                  (np.array(mean_test_control_infos) - np.array(std_test_control_infos))[::-1].tolist(),
                fill='toself',
                fillcolor=f"rgba({int(colors[color_idx][1:3], 16)}, "
                          f"{int(colors[color_idx][3:5], 16)}, "
                          f"{int(colors[color_idx][5:], 16)}, 0.2)",
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{env_desc} Control Information Std Dev",
            ))

            color_idx = (color_idx + 1) % len(colors)

        fig_control_info.update_layout(
            title=f"Control Information for {task_name}",
            xaxis_title="Weight (p)",
            yaxis_title="Average Control Information",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,
            height=800,
        )

        plot_path_control_info = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)[
                                     "save_path"] + "_control_info.png"
        fig_control_info.write_image(plot_path_control_info, scale=2)

        print(f"Saved evaluation plot for control information: {plot_path_control_info}")

        # Plot results for free energy
        fig_free_energy = go.Figure()

        color_idx = 0
        for env_idx in sorted(env_idxs.keys()):
            subtask_data = env_idxs[env_idx]
            mean_test_free_energies = subtask_data["mean_test_free_energies"]
            std_test_free_energies = subtask_data["std_test_free_energies"]
            test_weights = subtask_data["test_weights"]

            _, _, env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, env_idx)

            # Add mean free energy curve
            fig_free_energy.add_trace(go.Scatter(
                x=test_weights,
                y=mean_test_free_energies,
                mode='lines',
                name=f"{env_desc} Mean Free Energy",
                line=dict(color=colors[color_idx], width=2),
            ))

            # Add shaded area for std
            fig_free_energy.add_trace(go.Scatter(
                x=test_weights + test_weights[::-1],
                y=(np.array(mean_test_free_energies) + np.array(std_test_free_energies)).tolist() +
                  (np.array(mean_test_free_energies) - np.array(std_test_free_energies))[::-1].tolist(),
                fill='toself',
                fillcolor=f"rgba({int(colors[color_idx][1:3], 16)}, "
                          f"{int(colors[color_idx][3:5], 16)}, "
                          f"{int(colors[color_idx][5:], 16)}, 0.2)",
                line=dict(color='rgba(255,255,255,0)'),
                name=f"{env_desc} Free Energy Std Dev",
            ))

            color_idx = (color_idx + 1) % len(colors)

        # Customize layout for free energy
        fig_free_energy.update_layout(
            title=f"Free Energy for {task_name}",
            xaxis_title="Weight (p)",
            yaxis_title="Average Free Energy",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,
            height=800,
        )

        # Save plot for free energy
        plot_path_free_energy = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)[
                                    "save_path"] + "_free_energy.png"
        fig_free_energy.write_image(plot_path_free_energy, scale=2)
        print(f"Saved evaluation plot for free energy: {plot_path_free_energy}")

    return aggregated_results


if __name__ == '__main__':
    # run_all_trainings_and_plot(
    #     task_names_and_num_experiments={"frozen_lake-44": 16,},
    #     max_workers=16,
    # )
    # run_all_evals_and_plot(
    #     task_names_and_num_experiments={"frozen_lake-44": 16,},
    #     max_workers=16,
    # )
    # run_all_trainings_and_plot(
    #     task_names_and_num_experiments={"frozen_lake-88": 16,},
    #     max_workers=16,
    # )
    # run_all_evals_and_plot(
    #     task_names_and_num_experiments={"frozen_lake-88": 16,},
    #     max_workers=16,
    # )
    # run_all_trainings_and_plot(
    #     task_names_and_num_experiments={"frozen_lake-custom": 16, },
    #     max_workers=24,
    # )
    # run_all_evals_and_plot(
    #     task_names_and_num_experiments={"frozen_lake-custom": 16, },
    #     max_workers=24,
    # )
    run_all_trainings_and_plot(
        task_names_and_num_experiments={"mountaincar-custom": 6, },
        max_workers=6,
    )
    run_all_evals_and_plot(
        task_names_and_num_experiments={"mountaincar-custom": 6, },
        max_workers=9,
    )
