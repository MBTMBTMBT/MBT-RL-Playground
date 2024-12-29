from train_dyna_q_parallel import *

test_results, test_steps, final_test_reward = run_experiment("cartpole", 88)

print(test_results)
print(test_steps)
print(final_test_reward)
