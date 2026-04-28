import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from custom_env import CarRacingCustom

# Реєструємо середовище
gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom'
)

if __name__ == '__main__':
    # Створюємо паралельне середовище (8 світів) з незалежними процесами
    env = make_vec_env('CarRacingCustom-v0', n_envs=8, vec_env_cls=SubprocVecEnv)
    
    # Створюємо модель PPO
    model = PPO("MlpPolicy", env, verbose=1, n_steps=256, device="cpu", gamma=0.999)
    
    print("Починаємо навчання...")
    # Тренуємо 150 000 кроків
    model.learn(total_timesteps=5000000)
    
    # Зберігаємо навчений "мозок"
    model.save("ppo_car_racing")
    print("Модель збережено!")
    
    env.close()