from evaluate.stats import evaluate_stats
from utils.plotting import plot_two_episode_stats

num_episodes = 2000000
smoothing_window = 40000

Q_1, stats_1 = evaluate_stats(
    orders=10, vehicles=5, num_episodes=num_episodes, smoothing_window=smoothing_window)
Q_2, stats_2 = evaluate_stats(
    orders=5, vehicles=10, num_episodes=num_episodes, smoothing_window=smoothing_window)

plot_two_episode_stats(stats_1, stats_2, '算例3', '算例4',
                       smoothing_window=smoothing_window)
