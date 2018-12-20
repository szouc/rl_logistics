from learning.q_learning import q_learning
from learning.naive_learning import naive_learning
from learning.gaussian_learning import gaussian_learning
from learning.gumbel_learning import gumbel_learning
from envs import LogisticsEnv
from utils import plotting
from evaluate.rewards import evaluate_rewards


def evaluate_stats_contrast(orders=5, vehicles=3, kinds=3, num_episodes=2000, smoothing_window=100, truncation=0):

    env = LogisticsEnv(orders=orders, vehicles=vehicles,
                       kinds=kinds, vehicle_beta=0.5, driver_beta=0.5)
    Q, stats = q_learning(env=env, num_episodes=num_episodes,
                          discount_factor=0.9, alpha=0.5)
    stats_n = naive_learning(env=env, num_episodes=num_episodes)
    stats_g = gaussian_learning(env=env, num_episodes=num_episodes)
    stats_l = gumbel_learning(env=env, num_episodes=num_episodes)
    # rewards_stats_q, rewards_stats_n, _, _ = evaluate_rewards(
    #     env, Q, num_episodes=100)
    # plotting.plot_rewards_stats(rewards_stats_q, rewards_stats_n)
    # plotting.plot_episode_stats(
    #     stats, smoothing_window=smoothing_window, truncation=truncation)
    return Q, stats, stats_n, stats_g, stats_l

    # env_v = LogisticsEnv(orders=orders, vehicles=vehicles, kinds=kinds, vehicle_beta=0.9, driver_beta=0.1)
    # Q_v, stats_v = q_learning(env=env_v, num_episodes=num_episodes,
    #                         discount_factor=0.9, alpha=0.5)
    # rewards_stats_v = evaluate_rewards(env_v, Q_v)
    # plotting.plot_rewards_stats(rewards_stats_v)
    # plotting.plot_episode_stats(
    #     stats_v, smoothing_window=smoothing_window, truncation=truncation)

    # env_d = LogisticsEnv(orders=orders, vehicles=vehicles, kinds=kinds, vehicle_beta=0.1, driver_beta=0.9)
    # Q_d, stats_d = q_learning(env=env_d, num_episodes=num_episodes,
    #                         discount_factor=0.9, alpha=0.5)
    # rewards_stats_d = evaluate_rewards(env_v, Q_d)
    # plotting.plot_rewards_stats(rewards_stats_d)
    # plotting.plot_episode_stats(
    #     stats_d, smoothing_window=smoothing_window, truncation=truncation)
