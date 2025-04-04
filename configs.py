carracing_configs = {
    "env_type": "carracing",
    "num_seeds": 8,
    "n_repeat": 8,
    "n_envs": 16,
    "train_steps": 1_500_000,
    "eval_interval": 2_500 * 16,
    "num_init_states": 16,
    "eval_episodes": 16,
    "near_optimal_score": 8.5,
    "save_path": "./carracing_mapseed_results"
}


lunarlander_configs = {
    "env_type": "lunarlander",
    "num_densities": 5,
    "n_repeat": 8,
    "n_envs": 16,
    "train_steps": 1_200_000,
    "eval_interval": 2_500 * 16,
    "num_init_states": 64,
    "eval_episodes": 64,
    "near_optimal_score": 250,
    "save_path": "./lunar_lander_density_results"
}
