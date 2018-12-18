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


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5):
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
        episode_rewards=np.zeros(num_episodes),
        episode_restVehicles=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        Q, env.action_space.nvec, actions_number)

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
            action_probs = policy(state, 1 / (1 + i_episode))
            p = action_probs.flatten()
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
                _, _, vehicle_flags, _ = next_state
                stats.episode_restVehicles[i_episode] = vehicle_flags.count(0)
                break

            state = next_state

    return Q, stats
