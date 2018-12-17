import numpy as np
import sys
import itertools

from utils.plotting import EpisodeStats


def naive_policy(state):
    (order_flags, vehicle_flags, vehicle_costs) = state
    return state


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
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action = naive_policy(state)
            next_state, reward, done, _ = env.step(np.array(action))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break

            state = next_state

        # print(Q)

    return stats
