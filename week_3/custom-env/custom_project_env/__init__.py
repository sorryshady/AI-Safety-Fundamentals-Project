from gymnasium.envs.registration import register

register(
    id="custom_project_env/GridWorld-v0",
    entry_point="custom_project_env.envs:CustomGridWorldEnv",
    max_episode_steps=100
)
