import numpy as np
import itertools
from utils.plotting import RewardStats

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

def evaluate_rewards(env, Q):

    actions_number = get_actions_number(env.action_space)
    
    policy = make_epsilon_greedy_policy(
        Q, env.action_space.nvec, actions_number)

    # lengths = len(np.zeros(env.action_space.nvec, dtype=float).flatten())
    rewards_q = []
    rewards_sum_q = 0
    rewards_n = []
    rewards_sum_n = 0
    state = env.reset()
    for t in itertools.count():
        action_probs = policy(state, 1 / 10 )
        p = action_probs.flatten()

        action_index = np.random.choice(
            np.arange(len(action_probs.flatten())), p=p)
        action = np.unravel_index(action_index, env.action_space.nvec)
        next_state, reward, done, _ = env.step(np.array(action))

        # Update statistics
        rewards_sum_q += reward
        rewards_q.append((t, rewards_sum_q))

        if done:
            break

        state = next_state
    
    state = env.reset()
    for t in itertools.count():
        # Take a step
        action_probs = np.ones(env.action_space.nvec, dtype=float)
        action_index = np.random.choice(
            np.arange(len(action_probs.flatten())))
        action = np.unravel_index(action_index, env.action_space.nvec)
        # print(action)
        next_state, reward, done, _ = env.step(np.array(action))

        # Update statistics
        rewards_sum_n += reward
        rewards_n.append((t, rewards_sum_n))

        if done:
            break

        state = next_state

    stats_q = np.array(rewards_q)
    stats_n = np.array(rewards_n)

    return stats_q, stats_n
