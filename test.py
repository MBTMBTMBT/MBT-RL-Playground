from train_dyna_q_parallel import *

aggregated_results = run_all_experiments({"cartpole": 8}, max_workers=4)
print(aggregated_results)
