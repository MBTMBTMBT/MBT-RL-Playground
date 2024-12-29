import os
import random
import numpy as np
from tqdm import tqdm

from dyna_q import QCutTabularDynaQAgent
from dyna_q_task_configs import get_envs_discretizers_and_configs
from parallel_training import generate_test_gif


def run_experiment(task_name: str, run_id: int):
    env, test_env, state_discretizer, action_discretizer, action_type, configs = get_envs_discretizers_and_configs(task_name)
    dir_path = os.path.dirname(configs["save_path"])
    os.makedirs(dir_path, exist_ok=True)
    save_path = configs["save_path"] + f"-{run_id}"
    sample_strategies = ["explore_greedy", "greedy"]
    test_results = {}
    test_steps = {}
    final_test_rewards = {}

    for init_group in configs["init_groups"].keys():
        group_save_path = save_path + f"-{init_group}"
        init_distribution = configs["init_groups"][init_group]
        sample_steps = sorted([k for k in configs.keys() if isinstance(k, int)])
        test_results[init_group] = []
        test_steps[init_group] = []
        final_test_rewards[init_group] = 0.0

        agent = QCutTabularDynaQAgent(
            state_discretizer,
            action_discretizer,
            bonus_decay=configs["explore_bonus_decay"],
            max_steps=configs["train_max_num_steps_per_episode"],
            rough_reward_resolution=configs["reward_resolution"],
        )

        with tqdm(total=sample_steps[-1], desc=f"[{init_group}] - Sampling", unit="step", leave=False,) as pbar:
            sample_step_count = 0
            avg_test_reward = 0.0
            for sample_step in sample_steps:
                sample_strategy_distribution = configs[sample_step]["explore_policy_exploit_policy_ratio"]
                num_steps_to_sample = sample_step - sample_step_count
                current_step = 0
                sample_strategy_step_count = {s: 0 for s in sample_strategies}

                pbar.set_description(f"[{init_group}] - Sampling stage [{sample_step}/{str(sample_steps)}]")

                state, _ = env.reset()
                if isinstance(state, int) or isinstance(state, float):
                    state = [state]
                encoded_state = agent.state_discretizer.encode_indices(
                    list(agent.state_discretizer.discretize(state)[1])
                )
                agent.transition_table_env.add_start_state(encoded_state)

                init_sample_strategy = random.choices(sample_strategies, weights=sample_strategy_distribution, k=1)[0]
                while current_step < num_steps_to_sample:
                    if random.random() < configs["explore_epsilon"]:
                        action_vec = agent.choose_action(state, strategy="random")
                    else:
                        action_vec = agent.choose_action(state, strategy=init_sample_strategy)
                    if action_type == "int":
                        action = action_vec.astype("int64")
                        action = action[0].item()
                    elif action_type == "float":
                        action = action_vec.astype("float32")
                    next_state, reward, done, truncated, _ = env.step(action)
                    if isinstance(next_state, int) or isinstance(next_state, float):
                        next_state = [next_state]

                    agent.update_from_env(
                        state, action_vec, reward, next_state, done, configs["explore_agent_lr"],
                        configs["explore_value_decay"], update_policy=False,
                    )
                    state = next_state

                    current_step += 1
                    sample_step_count += 1
                    sample_strategy_step_count[init_sample_strategy] += 1

                    if done or truncated:
                        state, _ = env.reset()
                        if isinstance(state, int) or isinstance(state, float):
                            state = [state]
                        encoded_state = agent.state_discretizer.encode_indices(
                            list(agent.state_discretizer.discretize(state)[1])
                        )
                        agent.transition_table_env.add_start_state(encoded_state)
                        strategy_selection_dict = {}
                        for i, s in enumerate(sample_strategies):
                            if init_distribution[i] != 0:
                                strategy_selection_dict[s] = sample_strategy_step_count[s] / init_distribution[i]
                            else:
                                strategy_selection_dict[s] = np.inf
                        init_sample_strategy = min(strategy_selection_dict, key=strategy_selection_dict.get)

                    if configs[sample_step]["train_exploit_policy"]:
                        if current_step % configs["exploit_policy_training_per_num_steps"] == 0 and current_step > 1:
                            agent.update_from_transition_table(
                                configs["exploit_policy_training_steps"],
                                configs[sample_step]["epsilon"],
                                alpha=configs[sample_step]["train_exploit_lr"],
                                strategy=configs[sample_step]["train_exploit_strategy"],
                                init_strategy_distribution=init_distribution,
                                train_exploration_agent=False,
                                num_targets=configs["q_cut_params"]["num_targets"],
                                min_cut_max_flow_search_space=configs["q_cut_params"]["min_cut_max_flow_search_space"],
                                q_cut_space=configs["q_cut_params"]["q_cut_space"],
                                weighted_search=configs["q_cut_params"]["weighted_search"],
                                init_state_reward_prob_below_threshold=configs["q_cut_params"]["init_state_reward_prob_below_threshold"],
                                quality_value_threshold=configs["q_cut_params"]["quality_value_threshold"],
                                take_done_states_as_targets=configs["q_cut_params"]["take_done_states_as_targets"],
                            )

                    if configs[sample_step]["test_exploit_policy"]:
                        if (current_step + 1) % configs["exploit_policy_test_per_num_steps"] == 0 or current_step == num_steps_to_sample - 1:
                            periodic_test_rewards = []
                            frames = []
                            for t in range(configs["exploit_policy_test_episodes"]):
                                test_state, _ = test_env.reset()
                                if isinstance(test_state, int) or isinstance(test_state, float):
                                    test_state = [test_state]
                                test_total_reward = 0
                                test_done = False
                                while not test_done:
                                    test_action = agent.choose_action(test_state, strategy="greedy")
                                    if action_type == "int":
                                        test_action = test_action.astype("int64")[0].item()
                                    elif action_type == "float":
                                        test_action = test_action.astype("float32")
                                    test_next_state, test_reward, test_done, test_truncated, _ = test_env.step(
                                        test_action)
                                    if isinstance(test_next_state, int) or isinstance(test_next_state, float):
                                        test_next_state = [test_next_state]
                                    if t == 0:
                                        frames.append(test_env.render())
                                    test_state = test_next_state
                                    test_total_reward += test_reward
                                    if test_done or test_truncated:
                                        break
                                periodic_test_rewards.append(test_total_reward)

                            if (current_step + 1) % configs["exploit_policy_test_per_num_steps"] == 0:
                                avg_test_reward = np.mean(periodic_test_rewards)
                                test_results[init_group].append(avg_test_reward)
                                test_steps[init_group].append(sample_step_count)

                            # If this is the last test result, use it as the final result.
                            if current_step == num_steps_to_sample - 1:
                                final_test_rewards[init_group] = avg_test_reward

                            # Save GIF for the first test episode
                            gif_path = group_save_path + f"_{current_step}.gif"
                            generate_test_gif(frames, gif_path)

                    if (current_step + 1) % configs["save_per_num_steps"] == 0 or current_step == num_steps_to_sample - 1:
                        graph_path = group_save_path + f"_{current_step}.html"
                        save_csv_file = group_save_path + f"_{current_step}.csv"
                        agent.transition_table_env.print_transition_table_info()
                        agent.save_agent(save_csv_file)
                        if configs["save_mdp_graph"]:
                            agent.transition_table_env.save_mdp_graph(graph_path, use_encoded_states=True)

                    pbar.set_postfix({
                        "Episodes": sum(sample_strategy_step_count.values()),
                        "Avg Test Rwd": avg_test_reward,
                        "Explore Sample Episodes": sample_strategy_step_count["explore_greedy"],
                        "Exploit Sample Episodes": sample_strategy_step_count["greedy"],
                    })
                    pbar.update(1)

    return test_results, test_steps, final_test_rewards
