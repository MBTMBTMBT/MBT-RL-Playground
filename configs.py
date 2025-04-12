carracing_config = {
    "env_type": "carracing",
    "num_seeds": 8,
    "n_repeat": 8,
    "n_envs": 24,
    "train_steps": 2_500_000,
    "eval_interval": 5_000 * 24,
    "num_init_states": 24,
    "eval_episodes": 64,
    "near_optimal_score": 8.5,
    "save_path": "./carracing_mapseed_results",
}


lunarlander_config = {
    "env_type": "lunarlander",
    "num_densities": 5,
    "density_start": 3.0,
    "density_stop": 1.0,
    "n_repeat": 8,
    "n_envs": 24,
    "train_steps": 2_500_000,
    "eval_interval": 5_000 * 24,
    "num_init_states": 24,
    "eval_episodes": 64,
    "near_optimal_score": 250,
    "save_path": "./lunar_lander_density_results",
}
