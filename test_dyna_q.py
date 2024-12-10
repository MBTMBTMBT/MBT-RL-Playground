if __name__ == '__main__':
    from dyna_q import Discretizer, TabularDynaQAgent
    import gymnasium as gym
    from custom_mountain_car import CustomMountainCarEnv
    from tqdm import tqdm


    env = CustomMountainCarEnv(custom_gravity=0.0050)

    total_steps = int(0.1e6)
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.25

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

    agent = TabularDynaQAgent(state_discretizer, action_discretizer,)

    with tqdm(total=total_steps, leave=False) as pbar:
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            pass
            pbar.update(1)
