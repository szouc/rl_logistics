import numpy as np
import gym

class Choice(gym.Space):

    def __init__(self, choice):
        self.choice = choice
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        return gym.spaces.np_random.choice(self.choice)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int in self.choice

    def __repr__(self):
        return "Choice(%d)" % self.choice

    def __eq__(self, other):
        return self.choice == other.choice