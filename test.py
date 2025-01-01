from train_dyna_q_parallel import *

aggregated_results = run_all_experiments_and_plot({"texi": 8}, max_workers=16)
print(aggregated_results)
aggregated_results = run_all_experiments_and_plot({"mountain_car": 8}, max_workers=8)
print(aggregated_results)
aggregated_results = run_all_experiments_and_plot({"pendulum": 8}, max_workers=8)
print(aggregated_results)
aggregated_results = run_all_experiments_and_plot({"cartpole": 6}, max_workers=6)
print(aggregated_results)
aggregated_results = run_all_experiments_and_plot({"acrobot": 4}, max_workers=4)
print(aggregated_results)
