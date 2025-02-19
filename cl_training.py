import gc
import os
from multiprocessing import Pool
from typing import List, Dict, Tuple

import numpy as np
from gymnasium import spaces
from tqdm import tqdm
import plotly.graph_objs as go

from dyna_q_modernized import Agent
from configs_info import get_envs_discretizers_and_configs
from parallel_training import generate_test_gif


def run_2_stage_cl_training(task_name: str, prior_env_idx: int, target_env_idx: int, run_id: int,):
    prior_env, prior_test_env, prior_env_desc, _, _, _ \
        = get_envs_discretizers_and_configs(task_name, prior_env_idx)
    if prior_env_idx == -1:
        prior_env_desc = "scratch"
    assert target_env_idx != -1, "Target env must be specified for CL training."
    target_env, target_test_env, target_env_desc, state_discretizer, action_discretizer, configs \
        = get_envs_discretizers_and_configs(task_name, target_env_idx)
    dir_path = os.path.dirname(configs["save_path"])
    os.makedirs(dir_path, exist_ok=True)
    prior_save_path = configs["save_path"] + f"-{prior_env_desc}-id-{run_id}"
    target_save_path = configs["save_path"] + f"-{target_env_desc}-id-{run_id}"
    early_stop_counts = configs["early_stop_counts"]
    force_to_proceed_steps = int(configs["exploit_policy_training_steps"] // 2)

    agent = Agent(
        state_discretizer,
        action_discretizer,
        prior_env if not prior_env_idx == -1 else target_env,
        configs["use_deep_agent"],
        configs["train_max_num_steps_per_episode"],
        (1.0, 0.0),  # do not use re-initialization strategy
        configs["exploit_agent_lr"],
        configs["exploit_value_decay"],
        configs["exploit_policy_reward_rate"],
        configs["use_balanced_random_init"],
    )

    pbar = tqdm(
        total=configs["exploit_policy_training_steps"],
        desc=f"[{run_id}-{prior_env_desc}-{target_env_desc}]",
        unit="step",
        leave=False,
        dynamic_ncols=True,
        smoothing=1.0,
        mininterval=1.0,
        maxinterval=30.0,
    )
    sample_step_count = 0
    test_results = []
    target_test_results = []
    test_steps = []
    prior_end_step = 0  # this value tells when the env is changed (for training from scratch this value stays 0

    envs = [prior_env, target_env] if not prior_env_idx == -1 else [target_env]
    test_envs = [prior_test_env, target_test_env] if not prior_env_idx == -1 else [target_test_env]
    save_paths = [prior_save_path, target_save_path] if not prior_env_idx == -1 else [target_save_path]
    save_path = configs["save_path"] + f"-{prior_env_desc}-{target_env_desc}-id-{run_id}"

    first_test = True
    best_avg_reward = None
    no_improvement_count = 0  # Counter for early stopping based on test results

    success = False
    success_threshold = configs["success_threshold"]
    success_step = configs["exploit_policy_training_steps"]

    for idx, env in enumerate(envs):
        agent.exploit_agent.set_env(env)
        frames = []
        while sample_step_count < configs["exploit_policy_training_steps"]:
            # Train the agent for a fixed number of steps before evaluation
            if not first_test:
                agent.learn(configs["exploit_policy_test_per_num_steps"], False)
                sample_step_count += configs["exploit_policy_test_per_num_steps"]

            # Evaluate the agent after each training phase
            periodic_test_rewards = []

            frames = []
            for t in range(configs["exploit_policy_eval_episodes"]):
                test_state, _ = test_envs[idx].reset()
                test_total_reward = 0
                test_done = False
                trajectory = [test_state]

                while not test_done:
                    test_action = agent.choose_action(test_state, greedy=True)
                    test_next_state, test_reward, test_done, test_truncated, _ = test_envs[idx].step(test_action)
                    if t == 0:
                        frames.append(test_envs[idx].render())
                    trajectory.append(test_next_state)
                    test_state = test_next_state
                    test_total_reward += test_reward
                    if test_done or test_truncated:
                        break

                periodic_test_rewards.append(test_total_reward)

            first_test = False  # The first test is completed

            avg_test_reward = np.mean(periodic_test_rewards)
            test_results.append(avg_test_reward)
            test_steps.append(sample_step_count)

            if idx < len(envs) - 1:
                periodic_test_rewards = []
                for t in range(configs["exploit_policy_eval_episodes"]):
                    test_state, _ = test_envs[-1].reset()
                    test_total_reward = 0
                    test_done = False
                    trajectory = [test_state]

                    while not test_done:
                        test_action = agent.choose_action(test_state, greedy=True)
                        test_next_state, test_reward, test_done, test_truncated, _ = test_envs[idx].step(test_action)
                        trajectory.append(test_next_state)
                        test_state = test_next_state
                        test_total_reward += test_reward
                        if test_done or test_truncated:
                            break

                    periodic_test_rewards.append(test_total_reward)
                target_test_results.append(np.mean(periodic_test_rewards))
            else:
                target_test_results.append(avg_test_reward)

            pbar.set_postfix({
                "Rwd": f"{avg_test_reward:05.3f}",
            })
            pbar.update(configs["exploit_policy_test_per_num_steps"])

            if target_test_results[-1] >= success_threshold and not success:
                print(f"Success threshold reached at step {sample_step_count}")
                success_step = sample_step_count
                success = True

            if best_avg_reward is None:
                best_avg_reward = avg_test_reward

            # Ignore the first test result when considering early stopping
            if not first_test:
                if avg_test_reward < best_avg_reward or best_avg_reward >= 0.99:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0  # Reset counter if reward improves
                    best_avg_reward = avg_test_reward

                # Stop training if performance does not improve for 'early_stop_counts' evaluations
                if idx != len(envs) - 1 and no_improvement_count >= early_stop_counts:
                    print(f"Early stopping after {sample_step_count} steps")
                    break

                # if idx != len(envs) - 1 and best_avg_reward >= 0.99:
                #     print(f"Early stopping after {sample_step_count} steps")
                #     break

                if idx != len(envs) - 1 and sample_step_count >= force_to_proceed_steps:
                    print(f"Force to proceed after {sample_step_count} steps")
                    break

        # Save GIF for the first test episode
        if len(frames) > 0:
            gif_path = save_path + f".gif"
            try:
                generate_test_gif(frames, gif_path, to_print=configs["print_training_info"])
            except Exception as e:
                print(f"Error generating GIF: {e}")

        agent.save_agent(save_path)

        # Record when training switches from the prior environment to the target environment
        if idx != len(envs) - 1:
            prior_end_step = sample_step_count

    pbar.close()

    return task_name, run_id, prior_env_idx, target_env_idx, test_results, target_test_results, test_steps, prior_end_step, success_step


def run_2_stage_cl_training_unpack(args):
    return run_2_stage_cl_training(**args)


# Aggregating results for consistent step-based plotting
def run_all_2_stage_cl_training_and_plot(task_names_and_num_experiments: Dict[str, Tuple[int, int]], max_workers):
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
    run_id = 0
    for task_name, task_data in all_task_runs.items():
        env_run_ids = task_data["env_run_ids"]
        target_env_idx = task_data["target_env_idx"]  # The designated target environment
        target_run_ids = env_run_ids[target_env_idx]  # All run IDs for the target environment

        for target_run_id in target_run_ids:
            paired_tasks.append({
                "task_name": task_name,
                "prior_env_idx": -1,
                "target_env_idx": target_env_idx,
                "run_id": run_id,
            })
            run_id += 1

        for prior_env_idx, prior_run_ids in env_run_ids.items():
            if prior_env_idx == target_env_idx:
                continue  # Skip the target environment itself

            # Pair each prior environment run_id with all target environment run_ids
            for prior_run_id in prior_run_ids:
                paired_tasks.append({
                    "task_name": task_name,
                    "prior_env_idx": prior_env_idx,
                    "target_env_idx": target_env_idx,
                    "run_id": run_id,
                })
                run_id += 1

    print(f"Total tasks: {len(paired_tasks)}.")

    # Execute tasks in parallel
    if max_workers > 1:
        with Pool(processes=max_workers, maxtasksperchild=1) as pool:
            all_results = pool.map(run_2_stage_cl_training_unpack, paired_tasks)
    else:
        all_results = [run_2_stage_cl_training_unpack(task) for task in paired_tasks]

    # Group results by task_name and init_group
    grouped_results = {}
    for task_name, run_id, prior_env_idx, target_env_idx, test_results, target_test_results, test_steps, prior_end_step, success_step in all_results:
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
                "target_test_results": [],
                "test_steps": [],
                "prior_end_steps": [],
                "success_steps": []
            }

        # Append results to the corresponding group
        grouped_results[task_name][prior_env_idx][target_env_idx]["test_results"].append(test_results)
        grouped_results[task_name][prior_env_idx][target_env_idx]["target_test_results"].append(target_test_results)
        grouped_results[task_name][prior_env_idx][target_env_idx]["test_steps"].append(test_steps)
        grouped_results[task_name][prior_env_idx][target_env_idx]["prior_end_steps"].append(prior_end_step)  # Directly assign
        grouped_results[task_name][prior_env_idx][target_env_idx]["success_steps"].append(success_step)  # Directly assign

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
                target_test_results_array = np.array(data["target_test_results"])
                test_steps = data["test_steps"][0]
                prior_end_steps = data["prior_end_steps"]  # Retrieve as int
                success_steps = data["success_steps"]

                # Compute mean and std
                mean_test_results = test_results_array.mean(axis=0)
                std_test_results = test_results_array.std(axis=0)
                mean_target_test_results = target_test_results_array.mean(axis=0)
                std_target_test_results = target_test_results_array.std(axis=0)

                # Store aggregated results
                aggregated_results[task_name][prior_env_idx][target_env_idx] = {
                    "mean_test_results": mean_test_results.tolist(),
                    "std_test_results": std_test_results.tolist(),
                    "mean_target_test_results": mean_target_test_results.tolist(),
                    "std_target_test_results": std_target_test_results.tolist(),
                    "test_steps": test_steps,
                    "prior_end_steps": prior_end_steps,
                    "success_steps": success_steps,
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
    ]  # Hex color codes

    # Iterate over task_name, prior_env_idx, and target_env_idx
    for task_name, prior_envs in aggregated_results.items():
        if not prior_envs:
            continue  # Skip empty results

        fig_test = go.Figure()  # Test results figure
        fig_target_test = go.Figure()  # Target test results figure
        fig_integral = go.Figure()  # Integral (averaged) bar chart
        color_idx = 0

        # Store data for integral plot
        bar_labels = []
        bar_means_target_test = []
        bar_errors_target_test = []
        bar_means_success_steps = []
        bar_errors_success_steps = []

        for prior_env_idx, target_envs in prior_envs.items():
            for target_env_idx, subtask_data in target_envs.items():
                if not subtask_data["mean_test_results"]:
                    continue  # Skip empty results

                mean_test_results = np.array(subtask_data["mean_test_results"])
                std_test_results = np.array(subtask_data["std_test_results"])
                mean_target_test_results = np.array(subtask_data["mean_target_test_results"])
                std_target_test_results = np.array(subtask_data["std_target_test_results"])
                test_steps = subtask_data["test_steps"]
                prior_end_steps = subtask_data["prior_end_steps"]
                success_steps = np.array(subtask_data["success_steps"])
                success_mean = np.mean(success_steps)
                success_std = np.std(success_steps)

                # Compute integral (mean) for bar chart
                integral_mean_target_test = np.mean(mean_target_test_results)
                integral_std_target_test = np.mean(std_target_test_results)

                bar_means_success_steps.append(success_mean)
                bar_errors_success_steps.append(success_std)

                # Get environment description
                prior_env_desc = "scratch" if prior_env_idx == -1 else \
                get_envs_discretizers_and_configs(task_name, prior_env_idx)[2]
                target_env_desc = get_envs_discretizers_and_configs(task_name, target_env_idx)[2]

                # Label format: "Env X - Env Y"
                label = f"{prior_env_desc} - {target_env_desc}"

                # ---- Test results plot ----
                fig_test.add_trace(go.Scatter(
                    x=test_steps,
                    y=mean_test_results,
                    mode='lines',
                    name=f"{label}",
                    line=dict(color=colors[color_idx], width=2),
                ))

                fig_test.add_trace(go.Scatter(
                    x=test_steps + test_steps[::-1],
                    y=(mean_test_results + std_test_results).tolist() +
                      (mean_test_results - std_test_results)[::-1].tolist(),
                    fill='toself',
                    fillcolor=f"rgba({int(colors[color_idx][1:3], 16)}, {int(colors[color_idx][3:5], 16)}, {int(colors[color_idx][5:], 16)}, 0.2)",
                    line=dict(color='rgba(255,255,255,0)'),
                    # name=f"{label} Std Dev",
                    showlegend=False,
                ))

                # ---- Target test results plot ----
                fig_target_test.add_trace(go.Scatter(
                    x=test_steps,
                    y=mean_target_test_results,
                    mode='lines',
                    name=f"{label}",
                    line=dict(color=colors[color_idx], width=2),
                    # showlegend=False,
                ))

                fig_target_test.add_trace(go.Scatter(
                    x=test_steps + test_steps[::-1],
                    y=(mean_target_test_results + std_target_test_results).tolist() +
                      (mean_target_test_results - std_target_test_results)[::-1].tolist(),
                    fill='toself',
                    fillcolor=f"rgba({int(colors[color_idx][1:3], 16)}, {int(colors[color_idx][3:5], 16)}, {int(colors[color_idx][5:], 16)}, 0.2)",
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{label} Target Std Dev",
                    showlegend=False,
                ))

                # ---- Store data for integral bar chart ----
                bar_labels.append(label)
                bar_means_target_test.append(integral_mean_target_test)
                bar_errors_target_test.append(integral_std_target_test)

                # ---- Add environment transition lines to test plot ----
                if prior_env_idx != -1:
                    for idx, step in enumerate(prior_end_steps):
                        line_color = colors[color_idx]

                        fig_test.add_trace(go.Scatter(
                            x=[step, step],
                            y=[min(mean_test_results), max(mean_test_results)],
                            mode="lines",
                            line=dict(color=line_color, dash="dash"),
                            name=f"Env Transition (Step {step})",
                            showlegend=False,
                        ))

                color_idx = (color_idx + 1) % len(colors)

        # ---- Update layout for test results plot ----
        fig_test.update_layout(
            title=f"Training Results for {task_name} by Curriculum towards {target_env_desc}",
            xaxis_title="Steps",
            yaxis_title="Average Test Reward",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,
            height=800,
        )

        # ---- Update layout for target test results plot ----
        fig_target_test.update_layout(
            title=f"Target Training Results for {task_name} by Curriculum towards {target_env_desc}",
            xaxis_title="Steps",
            yaxis_title="Average Target Test Reward",
            legend_title="Subtasks",
            font=dict(size=14),
            width=1200,
            height=800,
        )

        # ---- Update layout for integral bar chart ----
        fig_integral.add_trace(go.Bar(
            x=bar_labels,
            y=bar_means_target_test,
            error_y=dict(type='data', array=bar_errors_target_test, visible=True),
            name="Target Test Results",
            marker_color=[colors[i % len(colors)] for i in range(len(bar_labels))],
            text=[f"{val:.2f}" for val in bar_means_target_test],  # Format values to 2 decimal places
            textposition='outside'  # Auto-adjust text position
        ))

        fig_integral.update_layout(
            title=f"Integrated Target Test Performance for {task_name} towards {target_env_desc}",
            xaxis_title="Environment Transition",
            yaxis_title="Mean Integrated Target Test Reward",
            legend_title="Metrics",
            font=dict(size=14),
            width=1000,
            height=750,
        )

        fig_success_steps = go.Figure()
        fig_success_steps.add_trace(go.Bar(
            x=bar_labels,
            y=bar_means_success_steps,
            error_y=dict(type='data', array=bar_errors_success_steps, visible=True),
            name="Success Steps",
            marker_color=[colors[i % len(colors)] for i in range(len(bar_labels))],
            text=[f"{val:d}" for val in bar_means_success_steps],
            textposition='outside'
        ))
        fig_success_steps.update_layout(
            title=f"Success Steps for {task_name} towards {target_env_desc}",
            xaxis_title="Environment Transition",
            yaxis_title="Mean Success Steps",
            legend_title="Metrics",
            font=dict(size=14),
            width=1000,
            height=750,
        )

        # ---- Save figures ----
        save_path = get_envs_discretizers_and_configs(task_name, env_idx=0, configs_only=True)["save_path"]

        plot_path_test = save_path + f"_training_cl-{target_env_desc}.png"
        fig_test.write_image(plot_path_test, scale=2)
        print(f"Saved evaluation plot for {task_name} at {plot_path_test}")

        plot_path_target_test = save_path + f"_training_target_cl-{target_env_desc}.png"
        fig_target_test.write_image(plot_path_target_test, scale=2)
        print(f"Saved target test evaluation plot for {task_name} at {plot_path_target_test}")

        plot_path_integral = save_path + f"_training_integral_target_cl-{target_env_desc}.png"
        fig_integral.write_image(plot_path_integral, scale=2)
        print(f"Saved integral target test performance plot for {task_name} at {plot_path_integral}")

        plot_path_success_steps = save_path + f"_training_success_steps_cl-{target_env_desc}.png"
        fig_success_steps.write_image(plot_path_success_steps, scale=2)
        print(f"Saved success steps performance plot for {task_name} at {plot_path_success_steps}")

    return aggregated_results


if __name__ == '__main__':
    run_all_2_stage_cl_training_and_plot(
        task_names_and_num_experiments={"frozen_lake-custom": (16, 7), },
        max_workers=1,
    )
