from learning.q_learning import q_learning
from learning.naive_learning import naive_learning
from learning.mc import mc_control_epsilon_greedy
from envs import LogisticsEnv
from utils import plotting

orders = 8
vehicles = 15
kinds = 3

num_episodes = 200000
smoothing_window = 4000
truncation = 0

env = LogisticsEnv(orders=orders, vehicles=vehicles, kinds=kinds)

Q, stats_q = q_learning(env=env, num_episodes=num_episodes,
                        discount_factor=0.9, alpha=0.5)

# plotting.plot_episode_stats(stats_q, smoothing_window=smoothing_window)

stats_n = naive_learning(env=env, num_episodes=num_episodes)

# plotting.plot_episode_stats(stats_n, smoothing_window=smoothing_window)

plotting.plot_two_episode_stats(
    stats_q, stats_n, smoothing_window=smoothing_window, truncation=truncation)

# Q, policy, stats_m = mc_control_epsilon_greedy(
#     env=env, num_episodes=num_episodes, discount_factor=0.9)

# plotting.plot_episode_stats(stats_m, smoothing_window=smoothing_window)

