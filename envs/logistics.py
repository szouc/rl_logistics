import numpy as np
import math

import gym
from gym import spaces
from gym.utils import seeding

from spaces import Choice


class LogisticsEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, vehicles=10, orders=10, vehicle_beta=0.5, driver_beta=0.5):
        self.vehicles = vehicles
        self.orders = orders
        self.vehicle_beta = vehicle_beta
        self.driver_beta = driver_beta
        self.action_space = spaces.Tuple(
            (spaces.MultiDiscrete((self.orders, self.vehicles)), Choice([2, 3, 6])))
        self.observation_space = spaces.Tuple((spaces.MultiBinary(
            self.orders), spaces.MultiBinary(self.vehicles), spaces.MultiDiscrete([7 for _ in range(self.vehicles)])))

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        (order_index, vehicle_index), cost = action
        order_flags, vehicle_flags, vehicle_costs = self.state

        if order_flags[order_index] == 1:
            reward = 0
        elif vehicle_costs[vehicle_index] + cost > 6:
            reward = 0
        else:
            vehicle_times = vehicle_flags[vehicle_index]
            vi = vehicle_costs[vehicle_index]
            vehicle_weight = vi / (cost + vi)
            driver_weight = cost / (cost + vi)
            reward = -1 * 1 / (1 + math.e ** (-vehicle_times * vehicle_weight * self.vehicle_beta)) + - \
                1 * 1 / (1 + math.e ** (vehicle_times *
                                        driver_weight * self.driver_beta))

            order_flags[order_index] = 1
            vehicle_flags[vehicle_index] += 1
            vehicle_costs[vehicle_index] += cost
            self.state = (order_flags, vehicle_flags, vehicle_costs)

        done = self._terminal()

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = (np.zeros((self.orders,), dtype=int),
                      np.zeros((self.vehicles,), dtype=int),
                      np.zeros((self.vehicles,), dtype=int))
        return np.array(self.state)

    def _terminal(self):
        s, _, v = self.state
        return sum(s) == self.orders or min(v) >= 5
