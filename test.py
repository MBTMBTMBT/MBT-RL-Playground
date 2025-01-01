from train_dyna_q_parallel import *

aggregated_results = run_all_experiments_and_plot({"acrobot": 4}, max_workers=16)
print(aggregated_results)
