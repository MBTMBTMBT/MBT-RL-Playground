import gc
import os
import random
from multiprocessing import Pool
from typing import List, Dict

import numpy as np
from tqdm import tqdm
import plotly.graph_objs as go

from dyna_q_modernized import DeepPyramidDynaQAgent
from dyna_q_task_configs_modernized_pyramid import get_envs_discretizers_and_configs
from parallel_training import generate_test_gif


def run_experiment(task_name: str, run_id: int, init_group: str):
    (
        env,
        test_env,
        state_discretizers,
        action_discretizers,
        configs,
    ) = get_envs_discretizers_and_configs(task_name)
    dir_path = os.path.dirname(configs["save_path"])
    os.makedirs(dir_path, exist_ok=True)
    save_path = configs["save_path"] + f"-{run_id}"
    sample_strategies = ["explore_greedy", "greedy"]
    test_results = []
    test_steps = []
    final_test_rewards = 0.0

    group_save_path = save_path + f"-{init_group}"
    sample_steps = sorted([k for k in configs.keys() if isinstance(k, int)])

    agent = DeepPyramidDynaQAgent(
        state_discretizers,
        action_discretizers,
        max_steps=configs["train_max_num_steps_per_episode"],
        reward_resolution=configs["reward_resolution"],
        init_strategy_distribution=(0.5, 0.5),
        explore_lr=configs["explore_agent_lr"],
        exploit_lr=configs["exploit_agent_lr"],
        gamma=configs["exploit_value_decay"],
        bonus_decay=configs["explore_bonus_decay"],
        exploit_policy_reward_rate=configs["exploit_policy_reward_rate"],
        add_noise=True,
    )

    pbar = tqdm(
        total=sample_steps[-1],
        desc=f"[{init_group}] - Sampling",
        unit="step",
        leave=True,
        dynamic_ncols=True,
        smoothing=1.0,
        mininterval=1.0,
        maxinterval=30.0,
    )
    sample_step_count = 0
    avg_test_reward = 0.0
    exploit_policy_updates = 0

    was_testing_exploit_policy = False
    for sample_step in sample_steps:
        sample_strategy_distribution = configs[sample_step][
            "explore_policy_exploit_policy_ratio"
        ]
        num_steps_to_sample = sample_step - sample_step_count
        current_step = 0
        sample_strategy_step_count = {s: 0 for s in sample_strategies}

        stage = ",".join(
            f"[{step / 1e3:.0f}e3]" if step == sample_step else f"{step / 1e3:.0f}e3"
            for step in sample_steps
        )

        if init_group == "pyramid" and "pyramid_index" in configs[sample_step]:
            pbar.set_description(
                f"[{run_id}-{task_name}-{init_group}]-Stage-[{stage}-Idx-{configs[sample_step]['pyramid_index']}]"
            )
        else:
            pbar.set_description(f"[{run_id}-{task_name}-{init_group}]-Stage-[{stage}]")

        state, _ = env.reset()
        agent.pyramid_env.leader_env.add_start_state(state)
        agent.transition_table_env_e.add_start_state(state)

        init_sample_strategy = random.choices(
            sample_strategies, weights=sample_strategy_distribution, k=1
        )[0]
        while current_step < num_steps_to_sample:
            if random.random() < configs["explore_epsilon"]:
                action = env.action_space.sample()
            else:
                if init_sample_strategy == "explore_greedy":
                    action = agent.choose_action(
                        state, explore_action=True, greedy=True
                    )
                else:
                    action = agent.choose_action(
                        state, explore_action=False, greedy=True
                    )
            next_state, reward, done, truncated, _ = env.step(action)
            if init_group == "pyramid" or init_group == "dynatrans":
                agent.update_from_env(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )
            state = next_state

            current_step += 1
            sample_step_count += 1
            sample_strategy_step_count[init_sample_strategy] += 1

            if done or truncated:
                state, _ = env.reset()
                agent.pyramid_env.leader_env.add_start_state(state)
                agent.transition_table_env_e.add_start_state(state)
                strategy_selection_dict = {}
                for i, s in enumerate(sample_strategies):
                    if sample_strategy_distribution[i] != 0:
                        strategy_selection_dict[s] = (
                            sample_strategy_step_count[s]
                            / sample_strategy_distribution[i]
                        )
                    else:
                        strategy_selection_dict[s] = np.inf
                init_sample_strategy = min(
                    strategy_selection_dict, key=strategy_selection_dict.get
                )

            if (
                (init_group == "pyramid" or init_group == "dynatrans")
                and sample_step_count % configs["explore_policy_training_per_num_steps"]
                == 0
                and sample_step_count > 1
            ):
                if (
                    not "pyramid_index" in configs[sample_step].keys()
                    or configs[sample_step]["pyramid_index"] >= 0
                ):
                    agent.update_from_transition_table(
                        total_timesteps=configs["explore_policy_training_steps"],
                        train_exploration_agent=True,
                        progress_bar=False,
                    )
                else:
                    pass  # Learning directly from real env so no need to update transition model

            if configs[sample_step]["train_exploit_policy"] or init_group == "baseline":
                if (
                    sample_step_count % configs["exploit_policy_training_per_num_steps"]
                    == 0
                    and sample_step_count > 1
                ):
                    if (
                        init_group == "pyramid"
                        and configs[sample_step]["pyramid_index"] >= 0
                    ):
                        agent.update_from_transition_table(
                            total_timesteps=configs["exploit_policy_training_steps"],
                            train_exploration_agent=False,
                            progress_bar=False,
                            slave_env_index=configs[sample_step]["pyramid_index"],
                            add_noise=True,
                            use_redistribution=sample_step < sample_steps[-2],
                        )
                    elif (
                        init_group == "dynatrans"
                        and configs[sample_step]["pyramid_index"] >= 0
                    ):
                        agent.update_from_transition_table(
                            total_timesteps=configs["exploit_policy_training_steps"],
                            train_exploration_agent=False,
                            progress_bar=False,
                            slave_env_index=configs[sample_steps[-2]]["pyramid_index"],
                            add_noise=True,
                            use_redistribution=False,
                        )
                    else:
                        agent.update_from_real_env(
                            total_timesteps=configs[
                                "exploit_policy_training_per_num_steps"
                            ],
                            real_env=env,
                            progress_bar=False,
                        )
                    exploit_policy_updates += 1

            if configs[sample_step]["test_exploit_policy"]:
                if (
                    not was_testing_exploit_policy
                    or sample_step_count == 1
                    or (sample_step_count + 1)
                    % configs["exploit_policy_test_per_num_steps"]
                    == 0
                    or (sample_step_count == sample_steps[-1] - 1)
                ):
                    was_testing_exploit_policy = True
                    periodic_test_rewards = []
                    frames = []
                    for t in range(configs["exploit_policy_test_episodes"]):
                        test_state, _ = test_env.reset()
                        test_total_reward = 0
                        test_done = False
                        while not test_done:
                            test_action = agent.choose_action(
                                test_state, explore_action=False, greedy=True
                            )
                            (
                                test_next_state,
                                test_reward,
                                test_done,
                                test_truncated,
                                _,
                            ) = test_env.step(test_action)
                            if t == 0:
                                frames.append(test_env.render())
                            test_state = test_next_state
                            test_total_reward += test_reward
                            if test_done or test_truncated:
                                break
                        periodic_test_rewards.append(test_total_reward)

                    if (
                        sample_step_count == 1
                        or (sample_step_count + 1)
                        % configs["exploit_policy_test_per_num_steps"]
                        == 0
                    ):
                        avg_test_reward = np.mean(periodic_test_rewards)
                        test_results.append(avg_test_reward)
                        test_steps.append(sample_step_count)

                    # If this is the last test result, use it as the final result.
                    if (
                        current_step == num_steps_to_sample - 1
                        or sample_step_count == sample_steps[-1] - 1
                    ):
                        final_test_rewards = avg_test_reward

                        # Save GIF for the first test episode
                        gif_path = group_save_path + f"_{sample_step_count}.gif"
                        generate_test_gif(
                            frames, gif_path, to_print=configs["print_training_info"]
                        )

                    # Save GIF for the first test episode
                    gif_path = group_save_path + f"_latest.gif"
                    generate_test_gif(
                        frames, gif_path, to_print=configs["print_training_info"]
                    )

            if (sample_step_count + 1) % configs["save_per_num_steps"] == 0 or (
                sample_step_count == sample_steps[-1] - 1
            ):
                pass
                # graph_path = group_save_path + f"_{sample_step_count}.html"
                # save_csv_file = group_save_path + f"_{sample_step_count}.csv"
                # agent.transition_table_env.print_transition_table_info()
                # agent.save_agent(save_csv_file)
                # if configs["save_mdp_graph"]:
                #     agent.transition_table_env.save_mdp_graph(graph_path, use_encoded_states=True)

            pbar.set_postfix(
                {
                    "Eps": f"{sum(sample_strategy_step_count.values()):.2e}",
                    "Test Rwd": (
                        f"{avg_test_reward:06.1f}"
                        if len(f"{int(avg_test_reward)}") <= 6
                        else f"{avg_test_reward:.1f}"
                    ),
                    "Explore Eps": f"{sample_strategy_step_count['explore_greedy']:.2e}",
                    "Exploit Eps": f"{sample_strategy_step_count['greedy']:.2e}",
                    "Found Trans": f"{len(agent.transition_table_env_e.forward_dict):.2e}",
                    "Updates": f"{exploit_policy_updates:.2e}",
                }
            )
            pbar.update(1)

    pbar.close()

    del agent
    gc.collect()

    return task_name, run_id, init_group, test_results, test_steps, final_test_rewards


# A wrapper function for unpacking arguments
def run_experiment_unpack(args):
    return run_experiment(**args)  # Unpack the dictionary into keyword arguments


# Aggregating results for consistent step-based plotting
def run_all_experiments_and_plot(
    task_names_and_num_experiments: Dict[str, int], max_workers
):
    tasks = []
    run_id = 0
    init_groups = [
        "dynatrans",
        "pyramid",
        "baseline",
    ]
    for task_name, runs in task_names_and_num_experiments.items():
        # Shuffle the sequence just for monitoring more possible cases simultaneously
        random.shuffle(init_groups)
        for init_group in init_groups:
            for _ in range(runs):
                if init_group != "baseline":
                    tasks.append(
                        {
                            "task_name": task_name,
                            "run_id": run_id,
                            "init_group": init_group,
                        }
                    )
                    run_id += 1

    random.shuffle(tasks)

    for task_name, runs in task_names_and_num_experiments.items():
        # Do only one run of baseline...
        if "baseline" in init_groups:
            tasks.insert(
                0,
                {
                    "task_name": task_name,
                    "run_id": run_id,
                    "init_group": "baseline",
                },
            )
            run_id += 1

    print(f"Total tasks: {run_id}.")

    # Execute tasks in parallel
    if max_workers > 1:
        with Pool(processes=max_workers, maxtasksperchild=1) as pool:
            all_results = pool.map(run_experiment_unpack, tasks)
    else:
        all_results = [run_experiment_unpack(task) for task in tasks]

    # Aggregate results for each group
    aggregated_results = {}

    # Group results by task_name and init_group
    grouped_results = {}
    for (
        task_name,
        run_id,
        init_group,
        test_results,
        test_steps,
        final_test_rewards,
    ) in all_results:
        if task_name not in grouped_results:
            grouped_results[task_name] = {}
        if init_group not in grouped_results[task_name]:
            grouped_results[task_name][init_group] = {
                "test_results": [],
                "test_steps": [],
                "final_test_rewards": [],
            }

        # Append results to the corresponding group
        grouped_results[task_name][init_group]["test_results"].append(test_results)
        grouped_results[task_name][init_group]["test_steps"].append(test_steps)
        grouped_results[task_name][init_group]["final_test_rewards"].append(
            final_test_rewards
        )

    # Aggregate data for each task_name and init_group
    for task_name, init_groups in grouped_results.items():
        aggregated_results[task_name] = {}
        for init_group, data in init_groups.items():
            test_results_array = np.array(data["test_results"])  # Shape: (runs, steps)
            test_steps = data["test_steps"][
                0
            ]  # Assume all runs share the same test_steps
            final_test_rewards_array = np.array(
                data["final_test_rewards"]
            )  # Shape: (runs,)

            # Compute mean and std for test_results
            mean_test_results = test_results_array.mean(axis=0)
            std_test_results = test_results_array.std(axis=0)

            # Compute mean and std for final_test_rewards
            mean_final_rewards = final_test_rewards_array.mean()
            std_final_rewards = final_test_rewards_array.std()

            # Store aggregated results
            aggregated_results[task_name][init_group] = {
                "mean_test_results": mean_test_results.tolist(),
                "std_test_results": std_test_results.tolist(),
                "test_steps": test_steps,
                "mean_final_rewards": mean_final_rewards,
                "std_final_rewards": std_final_rewards,
            }

    # Plot results
    for task_name, task_data in aggregated_results.items():
        fig = go.Figure()

        # Use hex color codes instead of names
        colors = [
            "#1f77b4",
            "#2ca02c",
            "#d62728",
            "#ff7f0e",
            "#9467bd",
        ]  # Hex color codes
        line_styles = [
            "dash",
            "dot",
            "longdash",
            "dashdot",
        ]  # Line styles for final results
        color_idx = 0

        for subtask in sorted(task_data.keys()):  # Sort subtasks alphabetically
            subtask_data = task_data[subtask]
            # Extract aggregated data
            mean_test_results = subtask_data["mean_test_results"]
            std_test_results = subtask_data["std_test_results"]
            test_steps = subtask_data["test_steps"]
            mean_final_rewards = subtask_data["mean_final_rewards"]

            # Add mean test results curve
            fig.add_trace(
                go.Scatter(
                    x=test_steps,
                    y=mean_test_results,
                    mode="lines",
                    name=f"{subtask} Mean Test Results",
                    line=dict(color=colors[color_idx], width=2),
                )
            )

            # Add shaded area for std
            fig.add_trace(
                go.Scatter(
                    x=test_steps + test_steps[::-1],  # Create a filled region
                    y=(
                        np.array(mean_test_results) + np.array(std_test_results)
                    ).tolist()
                    + (np.array(mean_test_results) - np.array(std_test_results))[
                        ::-1
                    ].tolist(),
                    fill="toself",
                    fillcolor=f"rgba({int(colors[color_idx][1:3], 16)}, "
                    f"{int(colors[color_idx][3:5], 16)}, "
                    f"{int(colors[color_idx][5:], 16)}, 0.2)",  # Match line color
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{subtask} Std Dev",
                )
            )

            # Add final mean result as a horizontal line
            fig.add_trace(
                go.Scatter(
                    x=[test_steps[0], test_steps[-1]],
                    y=[mean_final_rewards, mean_final_rewards],
                    mode="lines",
                    name=f"{subtask} Final Mean",
                    line=dict(
                        color="black",
                        width=2,
                        dash=line_styles[color_idx % len(line_styles)],
                    ),
                )
            )

            # Add annotation for the final mean result
            fig.add_trace(
                go.Scatter(
                    x=[test_steps[-1]],  # Position at the end of the line
                    y=[mean_final_rewards],
                    mode="text",
                    text=[
                        f"{mean_final_rewards:.2f}"
                    ],  # Format the number with two decimals
                    textposition="top right",
                    showlegend=False,  # Do not show in legend
                )
            )

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
        plot_path = (
            get_envs_discretizers_and_configs(task_name, configs_only=True)["save_path"]
            + ".png"
        )
        fig.write_image(plot_path, scale=2)  # High-resolution PNG
        print(f"Saved plot for {task_name} at {plot_path}")

    return aggregated_results
