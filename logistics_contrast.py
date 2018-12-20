from evaluate.stats_contrast import evaluate_stats_contrast
from utils.plotting import plot_four_rewards_stats

num_episodes = 50000
smoothing_window = 2000

Q, stats, stats_n, stats_g, stats_l = evaluate_stats_contrast(
    orders=5, vehicles=3, kinds=3, num_episodes=num_episodes, smoothing_window=smoothing_window)

plot_four_rewards_stats(stats, stats_n, stats_g, stats_l,
                        smoothing_window=smoothing_window)
