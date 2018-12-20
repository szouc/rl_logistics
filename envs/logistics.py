import numpy as np
import math

import gym
from gym import spaces
from gym.utils import seeding

kinds = [2, 3, 6]


class LogisticsEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, orders=10, vehicles=10, kinds=3, vehicle_beta=0.5, driver_beta=0.5):
        self.vehicles = vehicles
        self.orders = orders
        self.kinds = kinds
        self.vehicle_beta = vehicle_beta
        self.driver_beta = driver_beta
        self.action_space = spaces.MultiDiscrete(
            (self.orders, self.vehicles))
        self.observation_space = spaces.Tuple((spaces.MultiBinary(
            self.orders), spaces.MultiDiscrete([self.kinds + 1 for _ in range(self.orders)]), spaces.MultiDiscrete([self.orders + 1 for _ in range(self.vehicles)]), spaces.MultiDiscrete([7 for _ in range(self.vehicles)])))

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        (order_index, vehicle_index) = action
        (order_flags, order_costs, vehicle_flags, vehicle_costs) = self.state
        cost = kinds[order_costs[order_index]]

        if order_flags[order_index] == 0:
            reward = -2.0
        elif vehicle_costs[vehicle_index] + cost > 6:
            reward = -2.0
        else:
            vehicle_times = vehicle_flags[vehicle_index]
            vi = vehicle_costs[vehicle_index]
            vehicle_weight = vi / (cost + vi)
            driver_weight = cost / (cost + vi)
            vehicle_state_diff = 1.0 / \
                (1.0 + math.e ** (-vehicle_times *
                                  vehicle_weight * self.vehicle_beta)) - 0.5
            driver_state_diff = 1.0 / (1.0 + math.e ** (vehicle_times *
                                                        driver_weight * self.driver_beta)) - 0.5
            reward = -1.0 + vehicle_state_diff + driver_state_diff

            order_flags[order_index] = 0
            vehicle_flags[vehicle_index] += 1
            vehicle_costs[vehicle_index] += cost
            self.state = (order_flags, order_costs,
                          vehicle_flags, vehicle_costs)

        done = self._terminal()
        # if reward != 0:
        #     print(self.state, reward, sep='____')

        return self._get_obs(self.state), reward, done, {}

    def _get_obs(self, state):
        tuple_state = tuple(tuple(s) for s in state)
        return tuple_state

    def reset(self):
        self.state = (np.ones((self.orders,), dtype=int),
                      np.random.randint(
                          self.kinds, size=self.orders, dtype=int),
                      np.zeros((self.vehicles,), dtype=int),
                      np.zeros((self.vehicles,), dtype=int))
        return self._get_obs(self.state)

    def initial(self, state):
        self.state = np.array([np.array(s) for s in state])

    def _terminal(self):
        o, o_c, _, v_c = self.state
        if sum(o) == 0:
            return True
        if min(v_c) >= 5:
            return True
        rest_order_cost_dim = np.nonzero(o)
        # rest_order_cost_index = np.nonzero(np.bitwise_and(np.invert(o), o_c))
        min_rest_cost = kinds[min(o_c[rest_order_cost_dim])]
        if min(v_c) + min_rest_cost > 6:
            return True
        return False
