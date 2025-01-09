if __name__ == '__main__':
    from train_dyna_q_parallel_modernized_pyramid import *
    aggregated_results = run_all_experiments_and_plot({"cartpole": 3}, max_workers=3)
    print(aggregated_results)
    aggregated_results = run_all_experiments_and_plot({"acrobot": 3}, max_workers=3)
    print(aggregated_results)
    aggregated_results = run_all_experiments_and_plot({"mountain_car": 3}, max_workers=3)
    print(aggregated_results)
