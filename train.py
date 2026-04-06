import gymnasium as gym
from stable_baselines3 import PPO
from custom_env import CarRacingCustom

# Реєструємо середовище
gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom'
)

# Створюємо середовище без візуалізації (щоб вчилося максимально швидко)
env = gym.make('CarRacingCustom-v0')

# Створюємо модель PPO (MultiInputPolicy підходить для різних типів даних)
model = PPO("MlpPolicy", env, verbose=1)

print("Починаємо навчання...")
# Тренуємо 50 000 кроків (для початку)
model.learn(total_timesteps=50000)

# Зберігаємо навчений "мозок"
model.save("ppo_car_racing")
print("Модель збережено!")

env.close()