from train_dyna_q_parallel import *

aggregated_results = run_all_experiments({"cartpole": 4}, max_workers=4)
print(aggregated_results)
