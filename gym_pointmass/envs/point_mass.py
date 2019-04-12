
import gym
from gym import GoalEnv
from multiworld.core.multitask_env import MultitaskEnv
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import numpy as np
import math
import matplotlib.pyplot as plt
from enum import Enum
from gym_pointmass.core.utils import encode_one_hot

class PointMassEnvRewardType(Enum):
    DISTANCE = 1
    SPARSE = 2

class PointMassEnv(MultitaskEnv):
    """Implementation of a 2D PointMass environment."""

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 dimension=2,
                 n=1,
                 reward_type=PointMassEnvRewardType.DISTANCE,
                 max_time_steps=100,
                 epsilon=1e-4):
        MultitaskEnv.__init__(self)

        self.dimension = dimension
        self.num_goals = n
        self.reward_type = reward_type
        self.max_time_steps = max_time_steps
        self.epsilon = epsilon

        # Observation is a 2D vector (the x/y coordinates of the agent)
        self.obs_space = Box(low=-1, high=1, shape=(dimension,), dtype=np.float32)
        # Action is a 2D vector, representing the how far/in what direction the agent moves.
        self.action_space = Box(low=-1, high=1, shape=(dimension,), dtype=np.float32)
        self.goal_space = Box(low=-1, high=1, shape=(dimension,), dtype=np.float32)

        self.observation_space = Dict({
            "observation": self.obs_space,
            "desired_goal": self.goal_space
        })

        self.agent_position = self._get_random_point()
        self.goal_index = None
        self.goals = self._init_goals(n)
        self.steps = 0

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

    def _move_agent(self, action):
        """Expects an 2D action vector.
        >>> e = PointMassEnv()
        >>> e.agent_position = np.array([-1., 0.])
        >>> ne, se, sw, nw = np.array([1, 1]), np.array([1, -1]), np.array([-1, -1]), np.array([-1, 1])
        >>> print(e._move_agent(2 * ne))
        [0. 1.]
        >>> print(e._move_agent(2 * se))
        [1. 0.]
        >>> print(e._move_agent(2 * sw))
        [ 0. -1.]
        >>> print(e._move_agent(2 * nw))
        [-1.  0.]
        """
        assert len(action) == self.dimension, "Dimension of action does not align, expected" \
                                              " ({0},) but got {1}".format(self.dimension, action.shape)
        new_pos = self.agent_position + action
        new_x, new_y = new_pos[0], new_pos[1]
        old_x, old_y = self.agent_position[0], self.agent_position[1]
        act_x, act_y = action[0], action[1]

        if act_y == 0:  # Moving left/right
            new_x = min(new_x, 1) if new_x > 1 else max(-1, new_x)
        elif act_x == 0: # Moving up/down
            new_y = min(new_y, 1) if new_y > 1 else max(-1, new_y)
        elif np.abs(new_x) > 1 or np.abs(new_y) > 1:
            check_x, check_y = None, None
            if act_x > 0 and act_y > 0: # NE direction
                check_x, check_y = 1, 1
            elif act_x > 0 and act_y < 0: # SE direction
                check_x, check_y = 1, -1
            elif act_x < 0 and act_y < 0: # SW direction
                check_x, check_y = -1, -1
            elif act_x < 0 and act_y > 0: # NW direction
                check_x, check_y = -1, 1

            slope = act_y / act_x
            intersect_x = (slope * old_x + check_y - old_y) / slope
            intersect_y = slope * (check_x - old_x) + old_y
            if np.abs(intersect_x) <= 1:
                new_x, new_y = intersect_x, check_y
            elif np.abs(intersect_y) <= 1:
                new_x, new_y = check_x, intersect_y

        self.agent_position = np.array([new_x, new_y])
        return self.agent_position

    def step(self, action):
        """Adds action, which should be a n-dimensional vector, to the current agent."""
        action = np.array(action)

        # Move the agent.
        self._move_agent(action)

        obs = self._get_observation()
        reward = self.compute_reward(action, obs)
        info = self._get_info()
        self.steps += 1
        done = self.steps >= self.max_time_steps
        return obs, reward, done, info

    def reset(self):
        """
        Performs the following to reset the environment:
        1. Resets the position of the agent to a random point.
        2. Samples a goal position.
        3. Returns the dictionary containing the new observation and the (sampled) goal.
        """
        self.steps = 0
        self.agent_position = self._get_random_point()
        goal = self.sample_goal()
        self.goal_index = goal["goal_index"]
        return self._get_observation()

    def render(self, mode='human'):
        plt.clf()
        plt.scatter(self.agent_position[0], self.agent_position[1], color='r')
        plt.scatter(self.get_goal()[0], self.get_goal()[1], color='g')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()

    def get_goal(self):
        return encode_one_hot(self.num_goals, self.goal_index)

    def sample_goals(self, batch_size):
        rand_goal_index = np.random.choice(range(len(self.goals)), batch_size, replace=False)

        return {
            "goal_index": rand_goal_index
        }

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']

        goal_index = np.argmax(desired_goals)
        dist = np.linalg.norm(achieved_goals - self.goals[goal_index], axis=1)

        if self.reward_type == PointMassEnvRewardType.DISTANCE:
            # Reward is the negative L2-norm. Want to maximize the negative distance -->
            # minimizing distance from the goal.
            r = -dist
        elif self.reward_type == PointMassEnvRewardType.SPARSE:
            # Reward in this case is -1 if the agent's position is not within some
            # radius `epsilon` from the goal point. 0 if the agent's position is
            # near the goal.
            r = -(dist > self.epsilon).astype(int)

        return r

if __name__ == "__main__":
    # env = PointMassEnv()
    # env.reset()
    # env.agent_position = np.array([0, 0])
    # env.step(np.array([0.5, 0.5]))
    # print(env.agent_position)
    # env.step(np.array([0.2, 0.1]))
    # print(env.agent_position)
    # env.step(np.array([1.0, 1.0]))
    # print(env.agent_position)
    # env.step(np.array([0, 100.0]))
    # print(env.agent_position)

    env2 = PointMassEnv()
    env2.reset()
    env2.agent_position = np.array([1., 0.8])
    env2.step(np.array([0.5, 0.5]))
    print(env2.agent_position)
    env2.step(np.array([0.5, 0.5]))
    print(env2.agent_position)
    env2.step(np.array([0, -2]))
    print(env2.agent_position)
    env2.step(np.array([-3, 3]))
    print(env2.agent_position)
    env2.step(np.array([1, -4]))
    print(env2.agent_position)


