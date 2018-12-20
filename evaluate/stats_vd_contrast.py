from learning.q_learning import q_learning
from envs import LogisticsEnv
from control.rewards import control_rewards


def evaluate_vd_stats_contrast(orders, vehicles, kinds, num_episodes):

    env_55 = LogisticsEnv(orders=orders, vehicles=vehicles,
                          kinds=kinds, vehicle_beta=0.5, driver_beta=0.5)

    env_19 = LogisticsEnv(orders=orders, vehicles=vehicles,
                          kinds=kinds, vehicle_beta=0.1, driver_beta=0.9)

    env_91 = LogisticsEnv(orders=orders, vehicles=vehicles,
                          kinds=kinds, vehicle_beta=0.9, driver_beta=0.1)

    env_37 = LogisticsEnv(orders=orders, vehicles=vehicles,
                          kinds=kinds, vehicle_beta=0.3, driver_beta=0.7)

    Q_55, stats_55 = q_learning(env=env_55, num_episodes=num_episodes,
                                discount_factor=0.9, alpha=0.5)

    Q_19, stats_19 = q_learning(env=env_19, num_episodes=num_episodes,
                                discount_factor=0.9, alpha=0.5)

    Q_91, stats_91 = q_learning(env=env_91, num_episodes=num_episodes,
                                discount_factor=0.9, alpha=0.5)

    Q_37, stats_37 = q_learning(env=env_37, num_episodes=num_episodes,
                                discount_factor=0.9, alpha=0.5)

    # rewards_stats_q, rewards_stats_n, _, _ = evaluate_rewards(
    #     env, Q, num_episodes=100)
    # plotting.plot_rewards_stats(rewards_stats_q, rewards_stats_n)
    # plotting.plot_episode_stats(
    #     stats, smoothing_window=smoothing_window, truncation=truncation)
    return Q_55, Q_19, Q_91, Q_37, stats_55, stats_19, stats_91, stats_37

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
