from gymnasium.envs.registration import register

register(
    id="UR5Env-v0",
    entry_point="sim.ur5:UR5Env",
)
