from learning.q_learning import q_learning
from envs import LogisticsEnv
from utils import plotting

orders = 10
vehicles = 4
kinds = 3

env = LogisticsEnv(orders=orders, vehicles=vehicles, kinds=kinds)

Q, stats = q_learning(env=env, num_episodes=50000,
                      discount_factor=0.9, alpha=0.5, epsilon=0.1)

plotting.plot_episode_stats(stats, smoothing_window=1000)
