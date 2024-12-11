from custom_cartpole import CustomCartPoleEnv

if __name__ == '__main__':
    from dyna_q import Discretizer, TabularDynaQAgent
    from custom_mountain_car import CustomMountainCarEnv
    from tqdm import tqdm
    import random
    import numpy as np


    # env = CustomMountainCarEnv(custom_gravity=0.0025)
    # test_env = CustomMountainCarEnv(custom_gravity=0.0025)

    env = CustomCartPoleEnv()
    test_env = CustomCartPoleEnv()

    total_steps = int(1e6)
    alpha = 0.25
    rmax = 1.0
    rmax_alpha = 0.25
    gamma = 0.99
    env_epsilon = 0.1
    agent_epsilon = 0.25
    rmax_agent_epsilon = 0.25
    inner_training_per_num_steps = int(0.1e6)
    rmax_inner_training_per_num_steps = int(0.025e6)
    inner_training_steps = int(0.5e6)
    rmax_inner_training_steps = int(0.05e6)
    test_per_num_steps = int(10e3)
    test_runs = 10
    max_steps = 200

    state_discretizer = Discretizer(
        ranges = [(-1.2, 0.6), (-0.07, 0.07),],
        num_buckets=[64, 32],
        normal_params=[None, None],
    )

    action_discretizer = Discretizer(
        ranges=[(0, 2),],
        num_buckets=[0],
        normal_params=[None,],
    )

    state_discretizer = Discretizer(
        ranges=[(-2.4, 2.4), (-2, 2), (-0.25, 0.25), (-2, 2),],
        num_buckets=[8, 32, 32, 32],
        normal_params=[None, None, None, None,],
    )

    action_discretizer = Discretizer(
        ranges=[(0, 1),],
        num_buckets=[0],
        normal_params=[None, ],
    )

    agent = TabularDynaQAgent(state_discretizer, action_discretizer,)
    agent.transition_table_env.max_steps = max_steps

    with tqdm(total=total_steps, leave=False) as pbar:
        done = False
        current_steps = 0
        training_rewards = []
        test_rewards = []
        avg_test_reward = 0.0
        sample_counter = 0

        while current_steps < total_steps:
            state, _ = env.reset()
            done = False
            total_reward = 0
            encoded_state = agent.state_discretizer.encode_indices(list(agent.state_discretizer.discretize(state)[1]))
            agent.transition_table_env.add_start_state(encoded_state)
            paused = False
            while not done:
                if random.random() < env_epsilon:
                    action = agent.choose_action(state, strategy="random")
                else:
                    if sample_counter % 3 == 1:
                        action = agent.choose_action(state, strategy="rmax_greedy")
                    elif sample_counter % 3 == 2:
                        action = agent.choose_action(state, strategy="softmax")
                    else:
                        action = agent.choose_action(state, strategy="weighted")
                next_state, reward, done, truncated, _ = env.step(action[0].item())
                agent.update_from_env(state, action, reward, next_state, done, alpha, gamma)
                state = next_state
                total_reward += reward
                current_steps += 1
                pbar.update(1)

                if current_steps % rmax_inner_training_per_num_steps == 0 and current_steps > 1:
                    if sample_counter % 3 == 0:
                        agent.update_from_transition_table(
                            rmax_inner_training_steps,
                            rmax_agent_epsilon,
                            alpha=rmax_alpha,
                            strategy = "greedy",
                            init_strategy="random",
                            train_rmax_agent=True,
                            rmax=rmax,
                        )
                    paused = True

                if current_steps % inner_training_per_num_steps == 0 and current_steps > 1:
                    agent.update_from_transition_table(
                        inner_training_steps,
                        agent_epsilon,
                        alpha=alpha,
                        strategy="softmax",
                        init_strategy="random",
                        train_rmax_agent=False,
                    )
                    paused = True

                # Periodic testing
                if current_steps % test_per_num_steps == 0:
                    periodic_test_rewards = []
                    for _ in range(test_runs):
                        test_state, _ = test_env.reset()
                        test_total_reward = 0
                        test_done = False
                        while not test_done:
                            test_action = [np.argmax(agent.get_action_probabilities(test_state, strategy="greedy"))]
                            test_next_state, test_reward, test_done, test_truncated, _ = test_env.step(test_action[0])
                            test_state = test_next_state
                            test_total_reward += test_reward
                            if test_done or test_truncated:
                                break
                        periodic_test_rewards.append(test_total_reward)
                    avg_test_reward = np.mean(periodic_test_rewards)
                    recent_avg = np.mean([r for _, r in training_rewards[-10:]])
                    pbar.set_description(f"Epsilon: {agent_epsilon:.4f} | "
                                         f"Recent Avg Reward: {recent_avg:.2f} | "
                                         f"Avg Test Reward: {avg_test_reward:.2f}")

                if done or truncated:
                    training_rewards.append((current_steps, total_reward))
                    recent_avg = np.mean([r for _, r in training_rewards[-10:]])
                    pbar.set_description(f"Epsilon: {agent_epsilon:.4f} | "
                                         f"Recent Avg Reward: {recent_avg:.2f} | "
                                         f"Avg Test Reward: {avg_test_reward:.2f}")
                    break

                if paused:
                    sample_counter += 1
                    paused = False

    print(f"End of training. Avg Test Reward: {avg_test_reward:.2f}.")
