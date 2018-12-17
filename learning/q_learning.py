import numpy as np
import sys
import itertools
from collections import defaultdict

from utils.plotting import EpisodeStats


def get_actions_number(action_space):
    actions_number = 1
    for d in action_space.nvec:
        actions_number *= d
    return actions_number


def make_epsilon_greedy_policy(Q, epsilon, dA, nA):
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
    def policy_fn(observation):
        A = np.ones(dA, dtype=float) * epsilon / nA
        best_action_index = np.argmax(Q[observation])
        best_action = np.unravel_index(best_action_index, dA)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.nvec))

    actions_number = get_actions_number(env.action_space)

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        Q, epsilon, env.action_space.nvec, actions_number)

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
            action_probs = policy(state)
            p = action_probs.flatten()
            # p /= p.sum()
            action_index = np.random.choice(
                np.arange(len(action_probs.flatten())), p=p)
            action = np.unravel_index(action_index, env.action_space.nvec)
            next_state, reward, done, _ = env.step(np.array(action))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action_index = np.argmax(Q[next_state])
            best_next_action = np.unravel_index(
                best_next_action_index, env.action_space.nvec)
            td_target = reward + discount_factor * \
                Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

        # print(Q)

    return Q, stats


# def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
#     """
#     Monte Carlo Control using Epsilon-Greedy policies.
#     Finds an optimal epsilon-greedy policy.

#     Args:
#         env: OpenAI gym environment.
#         num_episodes: Number of episodes to sample.
#         discount_factor: Gamma discount factor.
#         epsilon: Chance the sample a random action. Float betwen 0 and 1.

#     Returns:
#         A tuple (Q, policy).
#         Q is a dictionary mapping state -> action values.
#         policy is a function that takes an observation as an argument and returns
#         action probabilities
#     """

#     # Keeps track of sum and count of returns for each state
#     # to calculate an average. We could use an array to save all
#     # returns (like in the book) but that's memory inefficient.
#     returns_sum = defaultdict(float)
#     returns_count = defaultdict(float)

#     # The final action-value function.
#     # A nested dictionary that maps state -> (action -> action-value).
#     Q = defaultdict(lambda: np.zeros(env.action_space.nvec))

#     # The policy we're following
#     policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.nvec)

#     for i_episode in range(1, num_episodes + 1):
#         # Print out which episode we're on, useful for debugging.
#         if i_episode % 1000 == 0:
#             print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
#             sys.stdout.flush()

#         # Generate an episode.
#         # An episode is an array of (state, action, reward) tuples
#         episode = []
#         state = env.reset()
#         for t in range(100):
#             probs = policy(state)
#             action = np.random.choice(np.arange(len(probs)), p=probs)
#             next_state, reward, done, _ = env.step(action)
#             episode.append((state, action, reward))
#             if done:
#                 break
#             state = next_state

#         # Find all (state, action) pairs we've visited in this episode
#         # We convert each state to a tuple so that we can use it as a dict key
#         sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
#         for state, action in sa_in_episode:
#             sa_pair = (state, action)
#             # Find the first occurance of the (state, action) pair in the episode
#             first_occurence_idx = next(i for i, x in enumerate(episode)
#                                        if x[0] == state and x[1] == action)
#             # Sum up all rewards since the first occurance
#             G = sum([x[2]*(discount_factor**i)
#                      for i, x in enumerate(episode[first_occurence_idx:])])
#             # Calculate average return for this state over all sampled episodes
#             returns_sum[sa_pair] += G
#             returns_count[sa_pair] += 1.0
#             Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

#         # The policy is improved implicitly by changing the Q dictionary

#     return Q, policy
