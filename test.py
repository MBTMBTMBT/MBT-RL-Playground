from train_dyna_q_parallel import *

aggregated_results = run_all_experiments_and_plot({"cartpole": 8}, max_workers=8)
print(aggregated_results)
