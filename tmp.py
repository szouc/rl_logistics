from learning.q_learning import q_learning
from envs import LogisticsEnv
from utils import plotting

orders = 100
vehicles = 10
kinds = 3

env = LogisticsEnv(orders=orders, vehicles=vehicles, kinds=kinds)

Q, stats = q_learning(env=env, num_episodes=100 ,
                      discount_factor=0.9, alpha=0.7, epsilon=0.1)

plotting.plot_episode_stats(stats)