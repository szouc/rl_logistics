from evaluate.stats_contrast import evaluate_stats_contrast
from utils.plotting import plot_four_rewards_stats

num_episodes = 2000
smoothing_window = 100

Q_1, stats, stats_n, stats_g, stats_l = evaluate_stats_contrast(
    orders=3, vehicles=5, kinds=3, num_episodes=num_episodes, smoothing_window=smoothing_window)

plot_four_rewards_stats(stats, stats_n, stats_g, stats_l, smoothing_window=smoothing_window)
