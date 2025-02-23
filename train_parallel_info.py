import gc
import os
from multiprocessing import Pool
from typing import List, Dict, Tuple

import numpy as np
import torch
from gymnasium import spaces
from tqdm import tqdm
import plotly.graph_objs as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

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

    # if configs["use_deep_agent"]:
    #     agent.exploit_agent.policy.to("cpu")

    if "fast_exploit_policy_training_steps" in configs.keys():
        exploit_policy_training_steps = configs["fast_exploit_policy_training_steps"]
    else:
        exploit_policy_training_steps = configs["exploit_policy_training_steps"]
    # exploit_policy_training_steps = configs["exploit_policy_training_steps"]

    pbar = tqdm(
        total=exploit_policy_training_steps,
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
    while sample_step_count < exploit_policy_training_steps:
        if not first_test:
            agent.learn(configs["exploit_policy_test_per_num_steps"], False)
            # baseline_agent.learn(configs["exploit_policy_test_per_num_steps"], False)
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
                    try:
                        frames.append(test_env.render())
                    except Exception as e:
                        print(e)
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

        # if sample_step_count % configs["save_per_num_steps"] == 0 and sample_step_count > 0:
        #     agent.save_agent(save_path + f"_{sample_step_count}")

        pbar.set_postfix({
            "Test Rwd": f"{avg_test_reward:04.3f}" if len(f"{int(avg_test_reward)}") <= 6 else f"{avg_test_reward:.3f}",
            "Found Trans": f"{len(agent.transition_table_env.forward_dict):.2e}",
        })
        pbar.update(configs["exploit_policy_test_per_num_steps"])

    agent.save_agent(save_path + f"_final")
    # baseline_agent.save_agent(save_path + f"_baseline")

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


def run_cl_eval(task_name: str, prior_env_idx: int, target_env_idx: int, prior_run_id: int, target_run_id: int, final_target_env_idx: int,):
    prior_env, prior_test_env, prior_env_desc, _, _, _ \
        = get_envs_discretizers_and_configs(task_name, prior_env_idx)
    if prior_env_idx == -1 and prior_run_id == -1:
        prior_env_desc = "scratch"
    target_env, target_test_env, target_env_desc, state_discretizer, action_discretizer, configs \
        = get_envs_discretizers_and_configs(task_name, target_env_idx)
    if target_env_idx == -1 and target_run_id == -1:
        target_env_desc = "scratch"
    _, final_target_test_env, _, _, _, _ \
        = get_envs_discretizers_and_configs(task_name, final_target_env_idx)
    dir_path = os.path.dirname(configs["save_path"])
    os.makedirs(dir_path, exist_ok=True)
    prior_save_path = configs["save_path"] + f"-{prior_env_desc}-id-{prior_run_id}"
    target_save_path = configs["save_path"] + f"-{target_env_desc}-id-{target_run_id}"

    prior_agent = Agent(
        state_discretizer,
        action_discretizer,
        prior_env,
        configs["use_deep_agent"],
        configs["train_max_num_steps_per_episode"],
        configs["initialization_distribution"],
        configs["exploit_agent_lr"],
        configs["exploit_value_decay"],
        configs["exploit_policy_reward_rate"],
        configs["use_balanced_random_init"],
    )
    if prior_env_idx != -1 and prior_run_id != -1:
        prior_agent.load_agent(prior_save_path + f"_final", load_transition_table=False)

    target_agent = Agent(
        state_discretizer,
        action_discretizer,
        target_env,
        configs["use_deep_agent"],
        configs["train_max_num_steps_per_episode"],
        configs["initialization_distribution"],
        configs["exploit_agent_lr"],
        configs["exploit_value_decay"],
        configs["exploit_policy_reward_rate"],
        configs["use_balanced_random_init"],
    )
    if target_env_idx != -1 and target_run_id != -1:
        target_agent.load_agent(target_save_path + f"_final", load_transition_table=False)

    # if configs["use_deep_agent"]:
    #     prior_agent.exploit_agent.policy.to("cpu")
    #     target_agent.exploit_agent.policy.to("cpu")

    if prior_env_idx != -1 and prior_run_id != -1 and target_env_idx != -1 and target_run_id != -1:
        if "quick_test_threshold" in configs.keys() and "quick_test_num_episodes" in configs.keys():
            quick_test_episodes = configs["quick_test_num_episodes"]
            test_total_rewards = []
            for test_env, agent in zip([prior_test_env, target_test_env], [prior_agent, target_agent]):
                for t in range(quick_test_episodes):
                    test_state, _ = test_env.reset()
                    test_total_reward = 0
                    test_done = False
                    while not test_done:
                        test_action = agent.choose_action_by_weight(test_state, p=1.0)
                        test_next_state, test_reward, test_done, test_truncated, _ = test_env.step(test_action)
                        test_state = test_next_state
                        test_total_reward += test_reward
                        if test_done or test_truncated:
                            test_total_rewards.append(test_total_reward)
                            break
                test_avg_reward = np.mean(test_total_rewards)
                if test_avg_reward < configs["quick_test_threshold"]:
                    return task_name, prior_run_id, target_run_id, prior_env_idx, target_env_idx, [], 0.0, 0.0, 0.0, 0.0, []

    kl_weight = 1.0
    action_space = target_agent.action_discretizer.get_gym_space()
    if isinstance(action_space, spaces.Discrete):
        kl_weight = 1.0 / np.log2(action_space.n)
    else:
        pass  # not implemented yet

    weights = np.linspace(0, 1, 20)

    pbar = tqdm(
        total=len(weights),
        desc=f"[{prior_run_id}-{target_run_id}-{prior_env_desc}-{target_env_desc}]",
        leave=False,
        dynamic_ncols=True,
    )

    test_results = []
    test_weights = []
    for p in weights:
        periodic_test_rewards = []
        for t in range(configs["exploit_policy_test_episodes"]):
        # for t in range(1):
            test_state, _ = target_test_env.reset()
            test_total_reward = 0
            test_done = False
            trajectory = [test_state]
            while not test_done:
                if prior_env_idx != -1 and prior_run_id != -1:
                    test_action = target_agent.choose_action_by_weight(
                        test_state, p=p, default_policy_func=prior_agent.get_greedy_weighted_action_distribution,
                    )
                    # test_action = target_agent.choose_action(test_state, greedy=True)
                else:
                    test_action = target_agent.choose_action_by_weight(
                        test_state, p=p, default_policy_func=prior_agent.get_default_policy_distribution,
                    )
                    # test_action = target_agent.choose_action(test_state, greedy=True)
                test_next_state, test_reward, test_done, test_truncated, _ = target_test_env.step(test_action)
                trajectory.append(test_next_state)
                test_state = test_next_state
                test_total_reward += test_reward
                if test_done or test_truncated:
                    break
            periodic_test_rewards.append(test_total_reward)

        avg_test_reward = np.mean(periodic_test_rewards)
        test_results.append(avg_test_reward)
        test_weights.append(p)

        pbar.set_postfix({
            "Weight": f"{p:.2f}",
            "Rwd": f"{avg_test_reward:05.3f}",
        })
        pbar.update(1)
    pbar.close()

    periodic_test_control_infos = []
    periodic_test_rewards_final_target = []
    periodic_test_control_infos_default = []
    for t in range(configs["exploit_policy_eval_episodes"]):
    # for t in range(1):
        test_state, _ = final_target_test_env.reset()
        test_total_reward = 0.0
        test_done = False
        trajectory = [test_state]
        while not test_done:
            test_action = target_agent.choose_action(
                test_state, temperature=1.5
            )
            test_next_state, test_reward, test_done, test_truncated, _ = final_target_test_env.step(test_action)
            trajectory.append(test_next_state)
            test_state = test_next_state
            test_total_reward += test_reward
            if test_done or test_truncated:
                break
        periodic_test_rewards_final_target.append(test_total_reward)

        # if p == weights[-1]:
        trajectory.reverse()
        control_info_target_prior = 0.0
        control_info_target_default = 0.0
        control_info_prior_default = 0.0
        if isinstance(action_space, spaces.Discrete):
            for state in trajectory:
                action_distribution_target = np.array(
                    target_agent.get_action_probabilities(state, temperature=1.5)).squeeze()
                action_distribution_prior = np.array(
                    prior_agent.get_action_probabilities(state, temperature=1.5)).squeeze()
                default_action_distribution = np.array(prior_agent.get_default_policy_distribution(state,)).squeeze()
                kl_divergence_target_prior = compute_discrete_kl_divergence(
                    action_distribution_target, action_distribution_prior
                )
                kl_divergence_target_default = compute_discrete_kl_divergence(
                    action_distribution_target, default_action_distribution
                )
                kl_divergence_prior_default = compute_discrete_kl_divergence(
                    action_distribution_prior, default_action_distribution
                )
                control_info_target_prior += kl_weight * kl_divergence_target_prior
                control_info_target_default += kl_weight * kl_divergence_target_default
                control_info_prior_default += kl_weight * kl_divergence_prior_default
        else:
            pass
        periodic_test_control_infos.append(control_info_target_prior)
        if prior_env_idx != -1 and prior_run_id != -1:
            periodic_test_control_infos_default.append(control_info_prior_default)  # / len(trajectory))
        else:
            periodic_test_control_infos_default.append(control_info_target_default)

    avg_test_reward_final_target = np.mean(periodic_test_rewards_final_target)
    avg_test_control_info = np.mean(periodic_test_control_infos)
    avg_test_control_info_default = np.mean(periodic_test_control_infos_default)

    test_free_energies = np.zeros(len(weights)).tolist()
    return task_name, prior_run_id, target_run_id, prior_env_idx, target_env_idx, test_results, avg_test_reward_final_target, avg_test_control_info, avg_test_control_info_default, test_free_energies, test_weights


# A wrapper function for unpacking arguments
def run_training_unpack(args):
    torch.set_num_threads(1)
    return run_training(**args)  # Unpack the dictionary into keyword arguments

def run_cl_eval_unpack(args):
    torch.set_num_threads(1)
    return run_cl_eval(**args)


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
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#fdae61', '#4daf4a', '#a65628', '#984ea3', '#e41a1c',
            '#377eb8', '#ff69b4', '#f781bf', '#66c2a5', '#ffcc00',
        ]

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
                name=f"{env_desc}",
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
                showlegend=False,
            ))

            # Add final mean result as a horizontal line
            fig.add_trace(go.Scatter(
                x=[test_steps[0], test_steps[-1]],
                y=[mean_final_rewards, mean_final_rewards],
                mode='lines',
                name=f"{env_desc} Final",
                line=dict(color='black', width=2, dash=line_styles[color_idx % len(line_styles)]),
            ))

            # Add annotation for the final mean result
            fig.add_trace(go.Scatter(
                x=[test_steps[-1]],  # Position at the end of the line
                y=[mean_final_rewards],
                mode='text',
                text=[f"{mean_final_rewards:.3f}"],  # Format the number with two decimals
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
            width=1200,
            height=800,
        )

        # Save plot to file using your specified path
        plot_path = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)["save_path"] + ".png"
        fig.write_image(plot_path, scale=2)  # High-resolution PNG
        print(f"Saved plot for {task_name} at {plot_path}")

    return aggregated_results


def run_all_cl_evals_and_plot(task_names_and_num_experiments: Dict[str, Tuple[int, int]], max_workers):
    run_id = 0

    # Store all run IDs by task_name and prior_env_idx
    all_task_runs = {}

    for task_name, (runs, target_env_idx) in task_names_and_num_experiments.items():
        num_envs = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)["num_envs"]

        if target_env_idx >= num_envs:
            raise ValueError(f"Invalid target_env_idx {target_env_idx} for task {task_name}, max is {num_envs - 1}")

        # Store run IDs for each prior_env_idx under the current task_name
        env_run_ids = {env_idx: [] for env_idx in range(num_envs)}

        for prior_env_idx in range(num_envs):
            for _ in range(runs):
                env_run_ids[prior_env_idx].append(run_id)
                run_id += 1

        # Save the environment run mappings for the current task
        all_task_runs[task_name] = {"env_run_ids": env_run_ids, "target_env_idx": target_env_idx}

    # Construct the paired task list for training comparison
    paired_tasks = []

    for task_name, task_data in all_task_runs.items():
        env_run_ids = task_data["env_run_ids"]
        target_env_idx = task_data["target_env_idx"]  # The designated target environment
        target_run_ids = env_run_ids[target_env_idx]  # All run IDs for the target environment

        for target_run_id in target_run_ids:
            paired_tasks.append({
                "task_name": task_name,
                "prior_env_idx": -1,
                "target_env_idx": -1,
                "prior_run_id": -1,
                "target_run_id": -1,
                "final_target_env_idx": target_env_idx,
            })
            paired_tasks.append({
                "task_name": task_name,
                "prior_env_idx": target_env_idx,
                "target_env_idx": target_env_idx,
                "prior_run_id": target_run_id,
                "target_run_id": target_run_id,
                "final_target_env_idx": target_env_idx,
            })
            paired_tasks.append({
                "task_name": task_name,
                "prior_env_idx": -1,
                "target_env_idx": target_env_idx,
                "prior_run_id": -1,
                "target_run_id": target_run_id,
                "final_target_env_idx": target_env_idx,
            })

        for prior_env_idx, prior_run_ids in env_run_ids.items():
            if prior_env_idx == target_env_idx:
                continue  # Skip the target environment itself

            # Pair each prior environment run_id with all target environment run_ids
            for prior_run_id, target_run_id in zip(prior_run_ids, target_run_ids):
                paired_tasks.append({
                    "task_name": task_name,
                    "prior_env_idx": -1,
                    "target_env_idx": prior_env_idx,
                    "prior_run_id": -1,
                    "target_run_id": prior_run_id,
                    "final_target_env_idx": target_env_idx,
                })
                paired_tasks.append({
                    "task_name": task_name,
                    "prior_env_idx": prior_env_idx,
                    "target_env_idx": target_env_idx,
                    "prior_run_id": prior_run_id,
                    "target_run_id": target_run_id,
                    "final_target_env_idx": target_env_idx,
                })

    print(f"Total tasks: {len(paired_tasks)}.")

    # Execute tasks in parallel
    if max_workers > 1:
        with Pool(processes=max_workers, maxtasksperchild=1) as pool:
            all_results = pool.map(run_cl_eval_unpack, paired_tasks)
    else:
        all_results = [run_cl_eval_unpack(task) for task in paired_tasks]

    # Group results by task_name, prior_env_idx, and target_env_idx
    grouped_results = {}

    for task_name, prior_run_id, target_run_id, prior_env_idx, target_env_idx, test_results, avg_test_reward_final_target, test_control_info, test_control_info_final_target, test_free_energies, test_weights in all_results:
        # Skip empty results
        if not (test_results and test_weights):
            print("run id: {}, task_name: {}, prior_env_idx: {} has no test results".format(prior_run_id, task_name,
                                                                                            prior_env_idx))
            continue

        # Initialize task_name in grouped_results
        if task_name not in grouped_results:
            grouped_results[task_name] = {}

        # Initialize prior_env_idx in grouped_results
        if prior_env_idx not in grouped_results[task_name]:
            grouped_results[task_name][prior_env_idx] = {}

        # Initialize target_env_idx within prior_env_idx
        if target_env_idx not in grouped_results[task_name][prior_env_idx]:
            grouped_results[task_name][prior_env_idx][target_env_idx] = {
                "test_results": [],
                "avg_test_reward_final_target": [],
                "test_control_info": [],
                "test_control_info_final_target": [],
                "test_free_energies": [],
                "test_weights": []
            }

        # Append results to the corresponding group
        grouped_results[task_name][prior_env_idx][target_env_idx]["test_results"].append(test_results)
        grouped_results[task_name][prior_env_idx][target_env_idx]["avg_test_reward_final_target"].append(avg_test_reward_final_target)
        grouped_results[task_name][prior_env_idx][target_env_idx]["test_control_info"].append(test_control_info)
        grouped_results[task_name][prior_env_idx][target_env_idx]["test_control_info_final_target"].append(test_control_info_final_target)
        grouped_results[task_name][prior_env_idx][target_env_idx]["test_free_energies"].append(test_free_energies)
        grouped_results[task_name][prior_env_idx][target_env_idx]["test_weights"].append(test_weights)

    # Aggregate data for each task_name, prior_env_idx, and target_env_idx
    aggregated_results = {}

    for task_name, prior_envs in grouped_results.items():
        aggregated_results[task_name] = {}

        for prior_env_idx, target_envs in prior_envs.items():
            aggregated_results[task_name][prior_env_idx] = {}

            for target_env_idx, data in target_envs.items():
                # Skip if there is no valid data
                if not any(data["test_results"]):
                    continue

                # Convert lists to numpy arrays
                test_results_array = np.array(data["test_results"])
                test_results_final_target_array = np.array(data["avg_test_reward_final_target"])
                test_control_info_array = np.array(data["test_control_info"])
                test_control_info_final_target_array = np.array(data["test_control_info_final_target"])
                test_free_energies_array = np.array(data["test_free_energies"])
                test_weights = data["test_weights"][0]

                # Compute mean and std
                mean_test_results = test_results_array.mean(axis=0)
                std_test_results = test_results_array.std(axis=0)

                mean_test_results_final_target = test_results_final_target_array.mean(axis=0)
                std_test_results_final_target = test_results_final_target_array.std(axis=0)

                mean_test_control_info = test_control_info_array.mean(axis=0)
                std_test_control_info = test_control_info_array.std(axis=0)

                mean_test_control_info_final_target = test_control_info_final_target_array.mean(axis=0)
                std_test_control_info_final_target = test_control_info_final_target_array.std(axis=0)

                mean_test_free_energies = test_free_energies_array.mean(axis=0)
                std_test_free_energies = test_free_energies_array.std(axis=0)

                # Store aggregated results
                aggregated_results[task_name][prior_env_idx][target_env_idx] = {
                    "mean_test_results": mean_test_results.tolist(),
                    "std_test_results": std_test_results.tolist(),
                    "mean_test_results_final_target": mean_test_results_final_target.tolist(),
                    "std_test_results_final_target": std_test_results_final_target.tolist(),
                    "mean_test_control_info": float(mean_test_control_info),
                    "std_test_control_info": float(std_test_control_info),
                    "mean_test_control_info_final_target": float(mean_test_control_info_final_target),
                    "std_test_control_info_final_target": float(std_test_control_info_final_target),
                    "mean_test_free_energies": mean_test_free_energies.tolist(),
                    "std_test_free_energies": std_test_free_energies.tolist(),
                    "test_weights": test_weights,
                }

            # Remove empty prior_env_idx
            if not aggregated_results[task_name][prior_env_idx]:
                del aggregated_results[task_name][prior_env_idx]

        # Remove empty task_name
        if not aggregated_results[task_name]:
            del aggregated_results[task_name]

    # Use hex color codes for consistency
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#fdae61', '#4daf4a', '#a65628', '#984ea3', '#e41a1c',
        '#377eb8', '#ff69b4', '#f781bf', '#66c2a5', '#ffcc00',
    ]

    # Iterate over task_name, prior_env_idx, and target_env_idx
    for task_name, prior_envs in aggregated_results.items():
        if not prior_envs:
            continue  # Skip empty results

        fig = go.Figure()
        color_idx = 0

        for prior_env_idx, target_envs in prior_envs.items():
            for target_env_idx, subtask_data in target_envs.items():
                if not subtask_data["mean_test_results"]:
                    continue  # Skip empty results

                mean_test_results = subtask_data["mean_test_results"]
                std_test_results = subtask_data["std_test_results"]
                test_weights = subtask_data["test_weights"]

                # Get environment description
                if prior_env_idx != -1:
                    _, _, prior_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, prior_env_idx)
                else:
                    prior_env_desc = "scratch"

                if target_env_idx != -1:
                    _, _, target_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, target_env_idx)
                else:
                    target_env_desc = "scratch"

                # Label format: "Env X - Env Y"
                label = f"{prior_env_desc} - {target_env_desc}"

                if prior_env_desc != "scratch" or target_env_desc == "scratch":
                    continue

                # Add mean test results curve
                fig.add_trace(go.Scatter(
                    x=test_weights,
                    y=mean_test_results,
                    mode='lines',
                    name=f"{label}",
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
                    # name=f"{label} Std Dev",
                    showlegend=False,
                ))

                color_idx = (color_idx + 1) % len(colors)

        # Customize layout
        fig.update_layout(
            title=f"Evaluation Results for {task_name} by Curriculum towards {target_env_desc}",
            xaxis_title="Weight (p)",
            yaxis_title="Average Test Reward",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,  # High resolution width
            height=800,  # High resolution height
        )

        # Save plot to file
        plot_path = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)[
                        "save_path"] + f"_eval_cl-{target_env_desc}.png"
        fig.write_image(plot_path, scale=2)  # High-resolution PNG
        print(f"Saved evaluation plot for {task_name} at {plot_path}")

        # Create a figure for the integral bar chart
        fig_integral = go.Figure()
        color_idx = 0

        # Store bar data
        bar_labels = []  # X-axis labels
        bar_means = []  # Heights of bars
        bar_errors = []  # Error bars

        for prior_env_idx, target_envs in prior_envs.items():
            for target_env_idx, subtask_data in target_envs.items():
                if not subtask_data["mean_test_results"]:
                    continue  # Skip empty results

                mean_test_results = np.array(subtask_data["mean_test_results"])
                std_test_results = np.array(subtask_data["std_test_results"])

                # Compute integral (or mean)
                integral_mean = np.mean(mean_test_results)  # Equivalent to integral divided by length
                integral_std = np.mean(std_test_results)  # Equivalent to integral error divided by length

                # Get environment description
                if prior_env_idx != -1:
                    _, _, prior_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, prior_env_idx)
                else:
                    prior_env_desc = "scratch"

                if target_env_idx != -1:
                    _, _, target_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, target_env_idx)
                else:
                    target_env_desc = "scratch"

                # Label format: "Env X - Env Y"
                label = f"{prior_env_desc} - {target_env_desc}"

                if prior_env_desc != "scratch" or target_env_desc == "scratch":
                    continue

                # Store data
                bar_labels.append(label)
                bar_means.append(integral_mean)
                bar_errors.append(integral_std)

        # Add bars to figure
        fig_integral.add_trace(go.Bar(
            x=bar_labels,
            y=bar_means,
            error_y=dict(type='data', array=bar_errors, visible=True),
            marker_color=[colors[i % len(colors)] for i in range(len(bar_labels))],  # Assign colors
            text=[f"{val:.3f}" for val in bar_means],  # Format values to 2 decimal places
            textposition='outside'  # Auto-adjust text position
        ))

        # Customize layout
        fig_integral.update_layout(
            title=f"Integrated Performance for {task_name} towards {target_env_desc}",
            xaxis_title="Environment Transition",
            yaxis_title="Mean Integrated Test Reward",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,
            height=800,
        )

        # Save plot
        plot_path_integral = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)[
                                 "save_path"] + f"_eval_integral_cl-{target_env_desc}.png"
        fig_integral.write_image(plot_path_integral, scale=2)

        print(f"Saved integral performance plot for {task_name} at {plot_path_integral}")

        # Initialize bar chart
        fig_control_info = go.Figure()
        color_idx = 0

        # Store bar data
        bar_labels = []  # X-axis labels
        bar_means = []  # Heights of bars
        bar_errors = []  # Error bars

        for prior_env_idx, target_envs in prior_envs.items():
            for target_env_idx, subtask_data in target_envs.items():
                if not subtask_data["mean_test_results"]:
                    continue  # Skip empty results

                mean_test_control_info = subtask_data["mean_test_control_info"]
                std_test_control_info = subtask_data["std_test_control_info"]

                # Get environment description
                if prior_env_idx != -1:
                    _, _, prior_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, prior_env_idx)
                else:
                    prior_env_desc = "scratch"

                if target_env_idx != -1:
                    _, _, target_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, target_env_idx)
                else:
                    target_env_desc = "scratch"

                # Label format: "Env X - Env Y"
                label = f"{prior_env_desc} - {target_env_desc}"

                if prior_env_desc == "scratch" or target_env_desc == "scratch":
                    continue

                # Store data
                bar_labels.append(label)
                bar_means.append(mean_test_control_info)
                bar_errors.append(std_test_control_info)

        # Add bars to figure
        fig_control_info.add_trace(go.Bar(
            x=bar_labels,
            y=bar_means,
            error_y=dict(type='data', array=bar_errors, visible=True),
            marker_color=[colors[i % len(colors)] for i in range(len(bar_labels))],  # Assign colors
            text=[f"{val:.3f}" for val in bar_means],  # Format values to 2 decimal places
            textposition='outside'  # Auto-adjust text position
        ))

        # Customize layout
        fig_control_info.update_layout(
            title=f"Evaluation Control Information for {task_name} towards {target_env_desc}",
            xaxis_title="Environment Transition",
            yaxis_title="Average Control Information by Curriculum",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,
            height=800,
        )

        # Save plot
        plot_path_control_info = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)[
                                     "save_path"] + f"_eval_control_info_cl-{target_env_desc}.png"
        fig_control_info.write_image(plot_path_control_info, scale=2)

        print(f"Saved evaluation plot for control information: {plot_path_control_info}")

        # Initialize bar chart
        fig_free_energy = go.Figure()
        color_idx = 0

        for prior_env_idx, target_envs in prior_envs.items():
            for target_env_idx, subtask_data in target_envs.items():
                if not subtask_data["mean_test_results"]:
                    continue  # Skip empty results

                mean_test_free_energies = subtask_data["mean_test_free_energies"]
                std_test_free_energies = subtask_data["std_test_free_energies"]
                test_weights = subtask_data["test_weights"]

                # Get environment description
                if prior_env_idx != -1:
                    _, _, prior_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, prior_env_idx)
                else:
                    prior_env_desc = "scratch"

                if target_env_idx != -1:
                    _, _, target_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, target_env_idx)
                else:
                    target_env_desc = "scratch"

                # Label format: "Env X - Env Y"
                label = f"{prior_env_desc} - {target_env_desc}"

                # Add mean test results curve
                fig_free_energy.add_trace(go.Scatter(
                    x=test_weights,
                    y=mean_test_free_energies,
                    mode='lines',
                    name=f"{label}",
                    line=dict(color=colors[color_idx], width=2),
                ))

                # Add shaded area for std
                fig_free_energy.add_trace(go.Scatter(
                    x=test_weights + test_weights[::-1],  # Create a filled region
                    y=(np.array(mean_test_free_energies) + np.array(std_test_free_energies)).tolist() +
                      (np.array(mean_test_free_energies) - np.array(std_test_free_energies))[::-1].tolist(),
                    fill='toself',
                    fillcolor=f"rgba({int(colors[color_idx][1:3], 16)}, "
                              f"{int(colors[color_idx][3:5], 16)}, "
                              f"{int(colors[color_idx][5:], 16)}, 0.2)",  # Match line color
                    line=dict(color='rgba(255,255,255,0)'),
                    # name=f"{label} Std Dev",
                    showlegend=False,
                ))

                color_idx = (color_idx + 1) % len(colors)

        # Customize layout
        fig_free_energy.update_layout(
            title=f"Free Energy for {task_name} by Curriculum towards {target_env_desc}",
            xaxis_title="Weight (p)",
            yaxis_title="Average Free Energy",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,  # High resolution width
            height=800,  # High resolution height
        )

        # Save plot to file
        plot_path = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)[
                        "save_path"] + f"_eval_free_energy_cl-{target_env_desc}.png"
        fig_free_energy.write_image(plot_path, scale=2)  # High-resolution PNG
        print(f"Saved free energy plot for {task_name} at {plot_path}")

        # Create a figure for the integral bar chart
        fig_free_energy_integral = go.Figure()
        color_idx = 0

        # Store bar data
        bar_labels = []  # X-axis labels
        bar_means = []  # Heights of bars
        bar_errors = []  # Error bars

        for prior_env_idx, target_envs in prior_envs.items():
            for target_env_idx, subtask_data in target_envs.items():
                if not subtask_data["mean_test_results"]:
                    continue  # Skip empty results

                mean_test_free_energies = subtask_data["mean_test_free_energies"]
                std_test_free_energies = subtask_data["std_test_free_energies"]

                # Compute integral (or mean)
                integral_mean = np.mean(mean_test_free_energies)  # Equivalent to integral divided by length
                integral_std = np.mean(std_test_free_energies)  # Equivalent to integral error divided by length

                # Get environment description
                if prior_env_idx != -1:
                    _, _, prior_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, prior_env_idx)
                else:
                    prior_env_desc = "scratch"

                if target_env_idx != -1:
                    _, _, target_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, target_env_idx)
                else:
                    target_env_desc = "scratch"

                # Label format: "Env X - Env Y"
                label = f"{prior_env_desc} - {target_env_desc}"

                # Store data
                bar_labels.append(label)
                bar_means.append(integral_mean)
                bar_errors.append(integral_std)

        # Add bars to figure
        fig_free_energy_integral.add_trace(go.Bar(
            x=bar_labels,
            y=bar_means,
            error_y=dict(type='data', array=bar_errors, visible=True),
            marker_color=[colors[i % len(colors)] for i in range(len(bar_labels))],  # Assign colors
            text=[f"{val:.3f}" for val in bar_means],  # Format values to 2 decimal places
            textposition='outside'  # Auto-adjust text position
        ))

        # Customize layout
        fig_free_energy_integral.update_layout(
            title=f"Integrated Free Energy for {task_name} towards {target_env_desc}",
            xaxis_title="Environment Transition",
            yaxis_title="Mean Integrated Free Energy",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,
            height=800,
        )

        # Save plot
        plot_path_integral = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)[
                                 "save_path"] + f"_eval_integral_free_energy_cl-{target_env_desc}.png"
        fig_free_energy_integral.write_image(plot_path_integral, scale=2)

        print(f"Saved free energy performance plot for {task_name} at {plot_path_integral}")

        # Store data
        scatter_x = []
        scatter_y = []
        scatter_labels = []
        scatter_colors = []
        final_target_env_idx = all_task_runs[task_name]["target_env_idx"]

        for task_name, prior_envs in aggregated_results.items():
            for prior_env_idx, target_envs in prior_envs.items():
                if prior_env_idx != -1:  # Only include "scratch" prior environments
                    continue  # Skip environments that are not "scratch"

                for target_env_idx, subtask_data in target_envs.items():
                    if not subtask_data["mean_test_results"]:
                        continue  # Skip empty results

                    # Get x (test_control_info_final_target) and y (result_final_target)
                    x_value = subtask_data["mean_test_control_info_final_target"]
                    y_value = subtask_data["mean_test_results_final_target"]

                    if target_env_idx != -1:
                        _, _, target_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name, target_env_idx)
                    else:
                        target_env_desc = "scratch"

                    # Use only target_env_desc as label
                    scatter_x.append(x_value)
                    scatter_y.append(y_value)
                    scatter_labels.append(target_env_desc)

                    # Highlight if target_env_idx is final_target_env_idx
                    scatter_colors.append(
                        'red' if target_env_idx == final_target_env_idx
                        else ('green' if target_env_idx == -1 else 'blue')
                    )

        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot scatter points with different colors
        ax.scatter(scatter_x, scatter_y, c=scatter_colors, alpha=0.7, s=50, label="Data Points")

        # ax.set_yscale('log')

        # Add grid for better visualization
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grid on both major & minor ticks

        # Add text labels with arrows
        texts = []
        for i, label in enumerate(scatter_labels):
            texts.append(ax.text(scatter_x[i], scatter_y[i], label, fontsize=12, color='black'))

        # Automatically adjust text to prevent overlapping, adding arrows if needed
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=1))

        # Set title and axis labels
        ax.set_title("Scatter Plot of Reward vs. Control Info", fontsize=16)
        ax.set_xlabel("Control Info with Uniform Prior", fontsize=14)
        ax.set_ylabel("Reward", fontsize=14)

        # Save figure
        plot_path_scatter = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)[
                                "save_path"] + "_scatter_result_vs_control_info.png"
        plt.savefig(plot_path_scatter, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Saved scatter plot for result vs. test control info at {plot_path_scatter}")

        # Store data
        scatter_x = []
        scatter_y = []
        scatter_labels = []
        scatter_colors = []

        for task_name, prior_envs in aggregated_results.items():
            for prior_env_idx, target_envs in prior_envs.items():
                for target_env_idx, subtask_data in target_envs.items():
                    if not subtask_data["mean_test_results"]:
                        continue  # Skip empty results
                    if target_env_idx != final_target_env_idx:
                        continue

                    # Get x (test_control_info_final_target) and y (result_final_target)
                    x_value = subtask_data["mean_test_control_info_final_target"]
                    y_value = subtask_data["mean_test_control_info"]

                    if prior_env_idx != -1:
                        _, _, prior_env_desc, _, _, _ = get_envs_discretizers_and_configs(task_name,
                                                                                           prior_env_idx)
                    else:
                        prior_env_desc = "scratch"

                    # Use only prior_env_desc as label
                    scatter_x.append(x_value)
                    scatter_y.append(-y_value)
                    scatter_labels.append(prior_env_desc)

                    # Highlight if target_env_idx is final_target_env_idx
                    scatter_colors.append(
                        'red' if prior_env_idx == final_target_env_idx
                        else ('green' if prior_env_idx == -1 else 'blue')
                    )

        # Create Matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot scatter points with different colors
        ax.scatter(scatter_x, scatter_y, c=scatter_colors, alpha=0.7, s=50, label="Data Points")

        # Add grid for better visualization
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grid on both major & minor ticks

        # Add text labels with arrows
        texts = []
        for i, label in enumerate(scatter_labels):
            texts.append(ax.text(scatter_x[i], scatter_y[i], label, fontsize=12, color='black'))

        # Automatically adjust text to prevent overlapping, adding arrows if needed
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=1))

        # Set title and axis labels
        ax.set_title("Scatter Plot of Reward vs. Control Info", fontsize=16)
        ax.set_xlabel("Control Info with Uniform Prior", fontsize=14)
        ax.set_ylabel("Minus Control with Optimal Prior of the Sub Env", fontsize=14)

        # Save figure
        plot_path_scatter = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)[
                                "save_path"] + "_scatter_learned_control_info_vs_control_info.png"
        plt.savefig(plot_path_scatter, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"Saved scatter plot for result vs. test control info at {plot_path_scatter}")

    return aggregated_results


if __name__ == '__main__':
    from cl_training import run_all_2_stage_cl_training_and_plot

    run_all_trainings_and_plot(
        task_names_and_num_experiments={"frozen_lake-4-times-4": 8, },
        max_workers=24,
    )
    run_all_cl_evals_and_plot(
        task_names_and_num_experiments={"frozen_lake-4-times-4": (8, 1), },
        max_workers=24,
    )
    run_all_2_stage_cl_training_and_plot(
        task_names_and_num_experiments={"frozen_lake-4-times-4": (8, 1), },
        max_workers=24,
    )

    # run_all_trainings_and_plot(
    #     task_names_and_num_experiments={"frozen_lake-custom": 8, },
    #     max_workers=24,
    # )
    # run_all_cl_evals_and_plot(
    #     task_names_and_num_experiments={"frozen_lake-custom": (8, 14), },
    #     max_workers=24,
    # )
    # run_all_2_stage_cl_training_and_plot(
    #     task_names_and_num_experiments={"frozen_lake-custom": (8, 14), },
    #     max_workers=24,
    # )
    #
    # run_all_trainings_and_plot(
    #     task_names_and_num_experiments={"acrobot-custom": 8, },
    #     max_workers=24,
    # )
    # run_all_cl_evals_and_plot(
    #     task_names_and_num_experiments={"acrobot-custom": (8, 0), },
    #     max_workers=24,
    # )
    # run_all_2_stage_cl_training_and_plot(
    #     task_names_and_num_experiments={"acrobot-custom": (8, 0), },
    #     max_workers=24,
    # )
