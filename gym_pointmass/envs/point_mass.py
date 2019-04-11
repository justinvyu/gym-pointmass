
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
        self.goal_index = None
        self.goals = self._init_goals(n)

    def _init_goals(self, num_goals):
        """Generates `num_goals` goal points, distributed evenly along the unit circle.
        ex: for `num_goals` == 2 --> returns [(1, 0), (-1, 0)]
        ex: for `num_goals` == 3 --> returns [(1, 0), (-1/2, sqrt(3)/2), (-1/2, -sqrt(3)/2)]
        ex: for `num_goals` == 4 --> returns [(1, 0), (0, 1), (-1, 0), (0, -1)]
        """
        degrees = np.arange(0, 2 * math.pi, 2 * math.pi / num_goals)
        pts = np.column_stack((np.cos(degrees), np.sin(degrees)))
        return pts.round(4) # 4 significant figures

    def _get_random_point(self):
        """Returns a random point (x, y, ...) in the space [-1, 1]^dim"""
        return 2 * np.random.rand(self.dimension) - 1

    def _get_observation(self):
        return {
            "observation": self.agent_position,
            "desired_goal": self.get_goal(),
            "achieved_goal": self.agent_position
        }

    def _get_info(self):
        pass

    def step(self, action):
        """Adds action, which should be a n-dimensional vector, to the current agent."""
        action = np.array(action)
        assert len(action) == self.dimension, "Dimension of action does not align, expected" \
                                              " ({0},) but got {1}".format(self.dimension, action.shape)

        # Move the agent.
        self.agent_position += action

        obs = self._get_observation()
        reward = self.compute_reward(action, obs)
        info = self._get_info()
        done = True
        return obs, reward, done, info

    def reset(self):
        """
        Performs the following to reset the environment:
        1. Resets the position of the agent to a random point.
        2. Samples a goal position.
        3. Returns the dictionary containing the new observation and the (sampled) goal.
        """
        self.agent_position = self._get_random_point()
        goal = self.sample_goal()
        self.goal_index = goal["goal_index"]
        return self._get_observation()

    def render(self, mode='human'):
        pass

    def get_goal(self):
        return self.goals[self.goal_index]

    def sample_goals(self, batch_size):
        return { "goal_index": np.random.choice(range(len(self.goals)), batch_size, replace=False) }

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']

        # Reward is the negative L2-norm. Want to maximize the negative distance -->
        # minimizing distance from the goal.
        r = -np.linalg.norm(achieved_goals - desired_goals, axis=1)

        return r