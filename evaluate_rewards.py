import numpy as np
import itertools
from utils.plotting import EpisodeStats


def get_actions_number(action_space):
    actions_number = 1
    for d in action_space.nvec:
        actions_number *= d
    return actions_number


def make_epsilon_greedy_policy(Q, dA, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        dA: Dimensions of actions in the environment.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation, epsilon):
        A = np.ones(dA, dtype=float) * epsilon / nA
        best_action_index = np.argmax(Q[observation])
        best_action = np.unravel_index(best_action_index, dA)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def evaluate_rewards(env, Q, num_episodes=100):

    actions_number = get_actions_number(env.action_space)

    policy = make_epsilon_greedy_policy(
        Q, env.action_space.nvec, actions_number)

    stats_q = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_restVehicles=np.zeros(num_episodes))

    stats_n = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_restVehicles=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        rewards_q = []
        rewards_sum_q = 0
        rewards_n = []
        rewards_sum_n = 0
        state = env.reset()
        state_bak = state[:]
        for t in itertools.count():
            action_probs = policy(state, 1 / (1 + i_episode))
            p = action_probs.flatten()

            action_index = np.random.choice(
                np.arange(len(action_probs.flatten())), p=p)
            action = np.unravel_index(action_index, env.action_space.nvec)
            next_state, reward, done, _ = env.step(np.array(action))

            # Update statistics
            stats_q.episode_rewards[i_episode] += reward
            stats_q.episode_lengths[i_episode] = t
            rewards_sum_q += reward
            rewards_q.append((t, rewards_sum_q))

            if done:
                break

            state = next_state

        env.initial(state_bak)
        for t in itertools.count():
            # Take a step
            action_probs = np.ones(env.action_space.nvec, dtype=float)
            action_index = np.random.choice(
                np.arange(len(action_probs.flatten())))
            action = np.unravel_index(action_index, env.action_space.nvec)
            # print(action)
            next_state, reward, done, _ = env.step(np.array(action))

            # Update statistics
            stats_n.episode_rewards[i_episode] += reward
            stats_n.episode_lengths[i_episode] = t
            rewards_sum_n += reward
            rewards_n.append((t, rewards_sum_n))

            if done:
                break

            state = next_state

        stats_e_q = np.array(rewards_q)
        stats_e_n = np.array(rewards_n)

    return stats_q, stats_n, stats_e_q, stats_e_n
