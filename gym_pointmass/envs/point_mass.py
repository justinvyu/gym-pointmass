
import gym
from gym import GoalEnv
from multiworld.core.multitask_env import MultitaskEnv
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box
import numpy as np
import math

class PointMassEnv(MultitaskEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, dimension=2, n=1):
        MultitaskEnv.__init__(self)

        self.dimension = dimension
        self.num_goals = n

        # Observation is a 2D vector (the x/y coordinates of the agent)
        self.observation_space = Box(low=-1, high=1, shape=(dimension,), dtype=np.float32)
        # Action is a 2D vector, representing the how far/in what direction the agent moves.
        self.action_space = Box(low=-1, high=1, shape=(dimension,), dtype=np.float32)

        self.agent_position = self._get_random_point()
        self.current_goal = None
        self.goals = self._init_goals(n)

    def _init_goals(self, num_goals):
        """Generates `num_goals` goal points, distributed evenly along the unit circle.
        ex: for `num_goals` == 2 --> returns [(1, 0), (-1, 0)]
        ex: for `num_goals` == 3 --> returns [(1, 0), (-1/2, sqrt(3)/2), (-1/2, -sqrt(3)/2)]
        ex: for `num_goals` == 4 --> returns [(1, 0), (0, 1), (-1, 0), (0, -1)]
        """
        degrees = np.arange(0, 2 * math.pi, 2 * math.pi / num_goals)
        pts = np.column_stack(np.cos(degrees), np.sin(degrees))
        return pts.round(4) # 4 significant figures

    def _get_random_point(self):
        """Returns a random point (x, y, ...) in the space [-1, 1]^dim"""
        return 2 * np.random.rand(self.dimension) - 1

    def step(self, action):
        """Adds action, which should be a n-dimensional vector, to the current agent"""

    def reset(self):
        """
        Performs the following to reset the environment:
        1. Resets the position of the agent to a random point.
        2. Samples a goal position.
        3. Returns the dictionary containing the new observation and the (sampled) goal.
        """
        self.agent_position = self._get_random_point()
        goal = self.sample_goal()
        self.current_goal = goal["state_desired_goal"]
        return {
            "observation": self.agent_position,
            "desired_goal": self.current_goal,
            "achieved_goal": self.agent_position
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_goal(self):
        pass

    def sample_goals(self, batch_size):
        return { "state_desired_goal": np.random.choice(self.goals, batch_size) }

    def compute_rewards(self, actions, obs):
        pass