import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from custom_env import CarRacingCustom

# Реєструємо середовище
gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom'
)

# Створюємо паралельне середовище (8 світів)
env = make_vec_env('CarRacingCustom-v0', n_envs=8)

# Створюємо модель PPO
model = PPO("MlpPolicy", env, verbose=1, n_steps=256)

print("Починаємо навчання...")
# Тренуємо 150 000 кроків
model.learn(total_timesteps=150000)

# Зберігаємо навчений "мозок"
model.save("ppo_car_racing")
print("Модель збережено!")

env.close()