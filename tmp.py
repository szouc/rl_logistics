from learning.q_learning import q_learning
from learning.naive_learning import naive_learning
from envs import LogisticsEnv
from utils import plotting

orders = 10
vehicles = 4
kinds = 3

num_episodes = 10000

env = LogisticsEnv(orders=orders, vehicles=vehicles, kinds=kinds)

Q, stats1 = q_learning(env=env, num_episodes=num_episodes,
                       discount_factor=0.9, alpha=0.5, epsilon=0.1)

# plotting.plot_episode_stats(stats, smoothing_window=1000)

stats2 = naive_learning(env=env, num_episodes=num_episodes)

plotting.plot_many_episode_stats(stats1, stats2, smoothing_window=100)
