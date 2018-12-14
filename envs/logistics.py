import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from spaces import Choice


class LogisticsEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        # 5 vehicles and 3 duties
        self.action_space = spaces.Tuple((spaces.Discrete(5), Choice([2, 3, 6])))
        self.observation_space = spaces.MultiDiscrete([7, 7, 7, 7, 7])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        
        vehicle_no, cost = action

        vi = self.state[vehicle_no]
        vehicle_weight = 1 + cost / (cost + vi)
        driver_weight = vi / (cost + vi)
        reward = -1 * vehicle_weight * 0.5 + -1 * driver_weight * 0.5

        self.state = self.state + action
        done = self._terminal()

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.zeros((5,), dtype=int)
        return np.array(self.state)

    def _terminal(self):
        s = self.state
        return min(s) > 6
