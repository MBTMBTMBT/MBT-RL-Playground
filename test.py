from train_dyna_q_parallel_modernized import *

aggregated_results = run_all_experiments_and_plot({"texi": 4}, max_workers=16)
print(aggregated_results)
# aggregated_results = run_all_experiments_and_plot({"cartpole": 1}, max_workers=1)
# print(aggregated_results)
# aggregated_results = run_all_experiments_and_plot({"mountain_car": 1}, max_workers=1)
# print(aggregated_results)
# aggregated_results = run_all_experiments_and_plot({"pendulum": 1}, max_workers=1)
# print(aggregated_results)
# aggregated_results = run_all_experiments_and_plot({"acrobot": 1}, max_workers=1)
# print(aggregated_results)
