if __name__ == "__main__":
    from old_stuff.train_dyna_q_parallel_modernized import *

    # aggregated_results = run_all_experiments_and_plot({"cartpole": 3}, max_workers=3)
    # print(aggregated_results)
    # aggregated_results = run_all_experiments_and_plot({"pendulum": 3}, max_workers=3)
    # print(aggregated_results)
    # aggregated_results = run_all_experiments_and_plot({"acrobot": 3}, max_workers=3)
    # print(aggregated_results)
    # aggregated_results = run_all_experiments_and_plot({"hopper": 5}, max_workers=3)
    # print(aggregated_results)
    # aggregated_results = run_all_experiments_and_plot({"reacher": 5}, max_workers=3)
    # print(aggregated_results)
    # aggregated_results = run_all_experiments_and_plot({"mountain_car": 3}, max_workers=3)
    # print(aggregated_results)
    # aggregated_results = run_all_experiments_and_plot({"texi": 4}, max_workers=12)
    # print(aggregated_results)
    aggregated_results = run_all_experiments_and_plot(
        {"half_cheetah": 1}, max_workers=2
    )
    print(aggregated_results)
    # aggregated_results = run_all_experiments_and_plot({"bipedalWalker": 1}, max_workers=1)
    # print(aggregated_results)
    # aggregated_results = run_all_experiments_and_plot({"lunarlander": 3}, max_workers=3)
    # print(aggregated_results)
