from learning.q_learning import q_learning
from learning.naive_learning import naive_learning
from learning.mc import mc_control_epsilon_greedy
from envs import LogisticsEnv
from utils import plotting
from evaluate_rewards import evaluate_rewards

orders = 10
vehicles = 10
kinds = 3

num_episodes = 1000000
smoothing_window = 20000
truncation = 0

env = LogisticsEnv(orders=orders, vehicles=vehicles, kinds=kinds, vehicle_beta=0.5, driver_beta=0.5)
# env_v = LogisticsEnv(orders=orders, vehicles=vehicles, kinds=kinds, vehicle_beta=0.9, driver_beta=0.1)
# env_d = LogisticsEnv(orders=orders, vehicles=vehicles, kinds=kinds, vehicle_beta=0.1, driver_beta=0.9)

Q, stats = q_learning(env=env, num_episodes=num_episodes,
                        discount_factor=0.9, alpha=0.5)

# Q_v, stats_v = q_learning(env=env_v, num_episodes=num_episodes,
#                         discount_factor=0.9, alpha=0.5)
                        
# Q_d, stats_d = q_learning(env=env_d, num_episodes=num_episodes,
#                         discount_factor=0.9, alpha=0.5)

# rewards_stats = evaluate_rewards(env, Q)
# rewards_stats_v = evaluate_rewards(env_v, Q_v)
# rewards_stats_d = evaluate_rewards(env_v, Q_d)

# plotting.plot_rewards_stats(rewards_stats)
# plotting.plot_rewards_stats(rewards_stats_v)
# plotting.plot_rewards_stats(rewards_stats_d)

plotting.plot_episode_stats(
    stats, smoothing_window=smoothing_window, truncation=truncation)

# plotting.plot_episode_stats(
#     stats_v, smoothing_window=smoothing_window, truncation=truncation)

# plotting.plot_episode_stats(
#     stats_d, smoothing_window=smoothing_window, truncation=truncation)
