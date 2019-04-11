from gym.envs.registration import register

register(
    id='PointMass-v0',
    entry_point='gym_pointmass.envs:PointMassEnv',
)