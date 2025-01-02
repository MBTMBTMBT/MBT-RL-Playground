from train_dyna_q_parallel import *

# aggregated_results = run_all_experiments_and_plot({"texi": 4}, max_workers=16)
# print(aggregated_results)
# aggregated_results = run_all_experiments_and_plot({"cartpole": 4}, max_workers=8)
# print(aggregated_results)
# aggregated_results = run_all_experiments_and_plot({"mountain_car": 4}, max_workers=16)
# print(aggregated_results)
# aggregated_results = run_all_experiments_and_plot({"pendulum": 4}, max_workers=8)
# print(aggregated_results)
aggregated_results = run_all_experiments_and_plot({"acrobot": 4}, max_workers=2)
print(aggregated_results)
