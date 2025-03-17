import os
import gymnasium as gym
import matplotlib.pyplot as plt
import custom_envs


NUM_SEEDS = 15
map_seeds = list(range(NUM_SEEDS))
SAVE_PATH = "./carracing_mapseed_results"

for map_seed in map_seeds:
    print(f"\n===== CarRacing Map Seed = {map_seed} =====")

    env = gym.make(
        "CarRacingFixedMap-v2",
        continuous=True,
        render_mode=None,
        map_seed=map_seed,
        fixed_start=True,
        backwards_tolerance=5,
        grass_tolerance=15,
        number_of_initial_states=16,
        init_seed=None,
        vector_obs=True,
    )

    env.reset()
    track_img = env.unwrapped.get_track_image(
        figsize=(10, 10),
    )
    map_path = os.path.join(SAVE_PATH, f"car_racing_map_seed_{map_seed}.png")
    plt.imsave(map_path, track_img)
