if __name__ == "__main__":
    from old_stuff.dyna_q import Discretizer, TabularDynaQAgent
    from old_stuff.custom_mountain_car import CustomMountainCarEnv
    from tqdm import tqdm
    import random
    import numpy as np
    from parallel_training import generate_test_gif

    env = CustomMountainCarEnv(custom_gravity=0.0025, render_mode="rgb_array")
    test_env = CustomMountainCarEnv(custom_gravity=0.0025, render_mode="rgb_array")
    save_file = "./experiments/DynaQ_Experiments/dyna_q_agent_mountain_car.csv"

    state_discretizer = Discretizer(
        ranges=[
            (-1.2, 0.6),
            (-0.07, 0.07),
        ],
        num_buckets=[64, 32],
        normal_params=[None, None],
    )

    action_discretizer = Discretizer(
        ranges=[
            (0, 2),
        ],
        num_buckets=[0],
        normal_params=[
            None,
        ],
    )

    action_type = "int"

    # env = CustomCartPoleEnv(render_mode="rgb_array")
    # test_env = CustomCartPoleEnv(render_mode="rgb_array")
    # save_file = "./experiments/DynaQ_Experiments/dyna_q_agent_cartpole.csv"

    # state_discretizer = Discretizer(
    #     ranges=[(-2.4, 2.4), (-2, 2), (-0.25, 0.25), (-2, 2),],
    #     num_buckets=[12, 32, 32, 32],
    #     normal_params=[None, None, None, None,],
    # )
    #
    # action_discretizer = Discretizer(
    #     ranges=[(0, 1),],
    #     num_buckets=[0],
    #     normal_params=[None, ],
    # )

    # env = gym.make("LunarLander-v3")
    # test_env = gym.make("LunarLander-v3")
    # save_file = "./experiments/DynaQ_Experiments/dyna_q_agent_lunarlander.csv"

    # state_discretizer = Discretizer(
    #     ranges=[
    #         (-1.5, 1.5), (-1.5, 1.5), (-5.0, 5.0), (-5.0, 5.0),
    #         (-3.14, 3.14), (-5.0, 5.0), (0, 1), (0, 1),
    #     ],
    #     num_buckets=[16, 16, 32, 32, 24, 32, 0, 0,],
    #     normal_params=[None, None, None, None, None, None, None, None,],
    # )
    #
    # action_discretizer = Discretizer(
    #     ranges=[(0, 1), ],
    #     num_buckets=[0],
    #     normal_params=[None, ],
    # )

    # env = gym.make("Acrobot-v1")
    # test_env = gym.make("Acrobot-v1")
    # save_file = "./experiments/DynaQ_Experiments/dyna_q_agent_acrobot.csv"

    # state_discretizer = Discretizer(
    #     ranges=[
    #         (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
    #         (-6.0, 6.0), (-12.0, 12.0),
    #     ],
    #     num_buckets=[16, 16, 16, 16, 16, 16,],
    #     normal_params=[None, None, None, None, None, None,],
    # )
    #
    # action_discretizer = Discretizer(
    #     ranges=[(0, 2), ],
    #     num_buckets=[0],
    #     normal_params=[None,],
    # )

    # env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")
    # test_env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")
    # save_file = "./experiments/DynaQ_Experiments/dyna_q_agent_bipedalwalker.csv"
    #
    # state_discretizer = Discretizer(
    #     ranges=[
    #         (-3.14, 3.14), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
    #         (-3.14, 3.14), (-5.0, 5.0), (-3.14, 3.14), (-5.0, 5.0),
    #         (-0.0, 5.0), (-3.14, 3.14), (-5.0, 5.0), (-3.14, 3.14),
    #         (-5.0, 5.0), (-0.0, 5.0), (-1.0, 1.0), (-1.0, 1.0),
    #         (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
    #         (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
    #     ],
    #     num_buckets=[8 for _ in range(14)] + [4 for _ in range(10)],
    #     normal_params=[None for _ in range(24)],
    # )
    #
    # action_discretizer = Discretizer(
    #     ranges=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),],
    #     num_buckets=[4, 4, 4, 4,],
    #     normal_params=[None, None, None, None,],
    # )

    # action_type = "float"

    total_steps = int(50e6)
    alpha = 0.25
    rmax = 1.0
    rmax_alpha = 0.25
    gamma = 0.99
    env_epsilon = 0.25
    agent_epsilon = 0.25
    rmax_agent_epsilon = 0.25
    inner_training_per_num_steps = int(0.25e6)
    rmax_inner_training_per_num_steps = int(0.025e6)
    inner_training_steps = int(0.25e6)
    rmax_inner_training_steps = int(0.01e6)
    test_per_num_steps = int(10e3)
    test_runs = 10
    max_steps = 250

    agent = TabularDynaQAgent(
        state_discretizer,
        action_discretizer,
    )
    agent.transition_table_env.max_steps = max_steps

    with tqdm(total=total_steps, leave=False) as pbar:
        done = False
        current_steps = 0
        training_rewards = []
        test_rewards = []
        avg_test_reward = 0.0
        sample_counter = 0
        test_counter = 0

        while current_steps < total_steps:
            state, _ = env.reset()
            done = False
            total_reward = 0
            encoded_state = agent.state_discretizer.encode_indices(
                list(agent.state_discretizer.discretize(state)[1])
            )
            agent.transition_table_env.add_start_state(encoded_state)
            paused = False
            while not done:
                if random.random() < env_epsilon:
                    action_vec = agent.choose_action(state, strategy="random")
                else:
                    # if sample_counter % 3 == 1:
                    #     action = agent.choose_action(state, strategy="rmax_softmax")
                    # elif sample_counter % 3 == 2:
                    #     action = agent.choose_action(state, strategy="softmax")
                    # else:
                    #     action = agent.choose_action(state, strategy="weighted")
                    action_vec = agent.choose_action(state, strategy="softmax")
                if action_type == "int":
                    action = action_vec.astype("int64")
                    action = action[0].item()
                elif action_type == "float":
                    action = action_vec.astype("float32")
                next_state, reward, done, truncated, _ = env.step(action)
                agent.update_from_env(
                    state, action_vec, reward, next_state, done, alpha, gamma
                )
                state = next_state
                total_reward += reward
                current_steps += 1
                pbar.update(1)

                # if current_steps % rmax_inner_training_per_num_steps == 0 and current_steps > 1:
                #     if sample_counter % 3 == 0:
                #         agent.update_from_transition_table(
                #             rmax_inner_training_steps,
                #             rmax_agent_epsilon,
                #             alpha=rmax_alpha,
                #             strategy = "softmax",
                #             init_strategy="random",
                #             train_exploration_agent=True,
                #             rmax=rmax,
                #         )
                #     paused = True
                #

                if (
                    current_steps % inner_training_per_num_steps == 0
                    and current_steps > 1
                ):
                    agent.update_from_transition_table(
                        inner_training_steps,
                        agent_epsilon,
                        alpha=alpha,
                        strategy="greedy",
                        init_strategy="real_start_states",
                        train_exploration_agent=False,
                    )
                    paused = True

                # Periodic testing
                if current_steps % test_per_num_steps == 0:
                    periodic_test_rewards = []
                    frames = []
                    for t in range(test_runs):
                        test_state, _ = test_env.reset()
                        test_total_reward = 0
                        test_done = False
                        while not test_done:
                            test_action = agent.choose_action(
                                test_state, strategy="greedy"
                            )
                            if action_type == "int":
                                test_action = test_action.astype("int64")[0].item()
                            elif action_type == "float":
                                test_action = test_action.astype("float32")
                            (
                                test_next_state,
                                test_reward,
                                test_done,
                                test_truncated,
                                _,
                            ) = test_env.step(test_action)
                            if t == 0 and test_counter % 5 == 0:
                                frames.append(test_env.render())
                            test_state = test_next_state
                            test_total_reward += test_reward
                            if test_done or test_truncated:
                                break
                        periodic_test_rewards.append(test_total_reward)
                    avg_test_reward = np.mean(periodic_test_rewards)
                    recent_avg = np.mean([r for _, r in training_rewards[-10:]])
                    pbar.set_description(
                        f"Epsilon: {agent_epsilon:.4f} | "
                        f"Recent Avg Reward: {recent_avg:.2f} | "
                        f"Avg Test Reward: {avg_test_reward:.2f}"
                    )

                    # Save GIF for the first test episode
                    save_file_gif = save_file.split(".csv")[0] + ".gif"
                    gif_path = save_file_gif.split(".gif")[0] + f"_{test_counter}.gif"
                    graph_path = gif_path.split(".gif")[0] + f".html"
                    if len(frames) > 0:
                        agent.transition_table_env.print_transition_table_info()
                        generate_test_gif(frames, gif_path)
                        agent.transition_table_env.save_mdp_graph(graph_path)
                    test_counter += 1

                if done or truncated:
                    training_rewards.append((current_steps, total_reward))
                    recent_avg = np.mean([r for _, r in training_rewards[-10:]])
                    pbar.set_description(
                        f"Epsilon: {agent_epsilon:.4f} | "
                        f"Recent Avg Reward: {recent_avg:.2f} | "
                        f"Avg Test Reward: {avg_test_reward:.2f}"
                    )
                    break

                if paused:
                    sample_counter += 1
                    paused = False

    print(f"End of training. Avg Test Reward: {avg_test_reward:.2f}.")
    agent.save_agent(save_file)
    agent.load_agent(save_file)
