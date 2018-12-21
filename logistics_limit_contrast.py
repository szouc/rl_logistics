from utils.plotting import plot_two_episode_stats
from envs.logistics import LogisticsEnv
from learning.q_learning import q_learning
from learning.imitation_learning import imitation_learning

orders = 3
vehicles = 5
kinds = 3

num_episodes = 2000000
smoothing_window = 40000

env = LogisticsEnv(orders=orders, vehicles=vehicles,
                   kinds=kinds, vehicle_beta=0.5, driver_beta=0.5)
Q, stats_1 = q_learning(env=env, num_episodes=num_episodes,
                        discount_factor=0.9, alpha=0.5)
stats_2 = imitation_learning(env=env, num_episodes=num_episodes)

plot_two_episode_stats(stats_1, stats_2, '算例3', '算例4',
                       smoothing_window=smoothing_window)
