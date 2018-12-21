from evaluate.stats_vd_contrast import evaluate_vd_stats_contrast
from utils.plotting import plot_four_vd_rewards_stats

orders = 3
vehicles = 5
kinds = 3

num_episodes = 100000
smoothing_window = 5000

_,_,_,_, stats, stats_n, stats_g, stats_l = evaluate_vd_stats_contrast(
    orders=orders, vehicles=vehicles, kinds=kinds, num_episodes=num_episodes)

plot_four_vd_rewards_stats(stats, stats_n, stats_g, stats_l,
                        smoothing_window=smoothing_window)
