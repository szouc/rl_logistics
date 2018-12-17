import numpy as np
import sys
import itertools
from collections import defaultdict

from utils.plotting import EpisodeStats


def naive_learning(env, num_episodes):

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action_probs = np.ones(env.action_space.nvec, dtype=float)
            action_index = np.random.choice(
                np.arange(len(action_probs.flatten())))
            action = np.unravel_index(action_index, env.action_space.nvec)
            # print(action)
            _, reward, done, _ = env.step(np.array(action))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break

    return stats
