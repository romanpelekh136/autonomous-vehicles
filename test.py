import gymnasium as gym
from stable_baselines3 import PPO
from custom_env import CarRacingCustom

gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom'
)

env = gym.make('CarRacingCustom-v0', render_mode="human")

# Завантажуємо навчену модель
model = PPO.load("ppo_car_racing")

observation, info = env.reset()
total_reward = 0

while True:
    # Модель передбачає дію на основі даних з лідара
    action, _states = model.predict(observation, deterministic=True)
    
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    env.render()
    
    if terminated or truncated:
        print(f"Епізод завершено. Нагорода: {total_reward:.2f}")
        observation, info = env.reset()
        total_reward = 0

env.close()