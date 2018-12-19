import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

EpisodeStats = namedtuple(
    "Stats", ["episode_lengths", "episode_rewards", "episode_restVehicles"])


RewardStats = namedtuple("Stats", ["rewards", 'lengths'])


def plot_rewards_stats(stats_q, stats_n, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats_q.episode_rewards, '-', label='Q-learning')
    plt.plot(stats_n.episode_rewards, ':', label='随机均匀分配')
    plt.xlabel("片段")
    plt.ylabel("收益")
    plt.title("每个片段的收益对比")
    plt.legend()
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    fig2 = plt.figure(figsize=(10, 5))
    plt.plot(stats_q.episode_lengths, '-', label='Q-learning')
    plt.plot(stats_n.episode_lengths, ':', label='随机均匀分配')
    plt.xlabel("片段")
    plt.ylabel("时间步")
    plt.title("每个片段含有的时间步")
    plt.legend()
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    return fig1, fig2


def plot_episode_stats(stats, smoothing_window=10, truncation=0, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    lengths_smoothed = pd.Series(stats.episode_lengths[truncation:]).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(lengths_smoothed, '-', label='Q-Learning')
    plt.xlabel("片段")
    plt.ylabel("时间步")
    plt.title("每个片段的时间步数目")
    plt.legend()
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards[truncation:]).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed, '-', label='Q-Learning')
    plt.xlabel("片段")
    plt.ylabel("收益 (平滑)")
    plt.title("每个片段的收益 (平滑窗口 {})".format(
        smoothing_window))
    plt.legend()
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths),
             np.arange(len(stats.episode_lengths)), '-', label='Q-Learning')
    plt.xlabel("时间步")
    plt.ylabel("片段")
    plt.title("时间步与片段关系")
    plt.legend()
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3


def plot_two_episode_stats(stats_q, stats_n, smoothing_window=10, truncation=0, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    lengths_smoothed_q = pd.Series(stats_q.episode_lengths[truncation:]).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    lengths_smoothed_n = pd.Series(stats_n.episode_lengths[truncation:]).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(lengths_smoothed_q, '-', label='Q-Learning')
    plt.plot(lengths_smoothed_n, ':', label='随机分配')
    plt.xlabel("片段")
    plt.ylabel("时间步")
    plt.title("每个片段的时间步数目")
    plt.legend()
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed_q = pd.Series(stats_q.episode_rewards[truncation:]).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_n = pd.Series(stats_n.episode_rewards[truncation:]).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed_q, '-', label='Q-Learning')
    plt.plot(rewards_smoothed_n, ':', label='随机分配')
    plt.xlabel("片段")
    plt.ylabel("收益 (平滑)")
    plt.title("每个片段的收益 (平滑窗口 {})".format(
        smoothing_window))
    plt.legend()
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats_q.episode_lengths),
             np.arange(len(stats_q.episode_lengths)), '-', label='Q-Learning')
    plt.plot(np.cumsum(stats_n.episode_lengths),
             np.arange(len(stats_n.episode_lengths)), ':', label='随机分配')
    plt.xlabel("时间步")
    plt.ylabel("片段")
    plt.title("时间步与片段关系")
    plt.legend()
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    # Plot time steps and episode number
    fig4 = plt.figure(figsize=(10, 5))
    # restVehicles_smoothed_q = pd.Series(stats_q.episode_restVehicles[truncation:]).rolling(
    #     smoothing_window, min_periods=smoothing_window).mean()
    # restVehicles_smoothed_n = pd.Series(stats_n.episode_restVehicles[truncation:]).rolling(
    #     smoothing_window, min_periods=smoothing_window).mean()
    # plt.plot(restVehicles_smoothed_q, label='Q-Learning')
    # plt.plot(restVehicles_smoothed_n, label='随机分配')
    restVechiles_diff = stats_q.episode_restVehicles - stats_n.episode_restVehicles
    restVehicles_diff_smoothed = pd.Series(restVechiles_diff[truncation:]).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(restVehicles_diff_smoothed, label='剩余车辆差')
    # plt.plot(stats_n.episode_restVehicles, '.', label='随机分配')
    plt.xlabel("片段")
    plt.ylabel("剩余车辆差")
    plt.title("剩余车辆差与片段关系")
    plt.legend()
    if noshow:
        plt.close(fig4)
    else:
        plt.show(fig4)

    return fig1, fig2, fig3, fig4
