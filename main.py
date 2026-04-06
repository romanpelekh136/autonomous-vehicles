import gymnasium as gym
from custom_env import CarRacingCustom

gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom'
)

env = gym.make('CarRacingCustom-v0', render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() 
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()