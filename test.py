import gymnasium as gym
from stable_baselines3 import PPO
from custom_env import CarRacingCustom
import json
import time

gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom'
)

env = gym.make('CarRacingCustom-v0', render_mode="human")

# Завантажуємо навчену модель
model = PPO.load("ppo_car_racing")

observation, info = env.reset()
total_reward = 0

action_log = []
frame_count = 0

print("Запуск симуляції... Натисніть Ctrl+C в терміналі або закрийте вікно, щоб зупинити.")

try:
    while True:
        # Модель передбачає дію на основі даних з лідара
        action, _states = model.predict(observation, deterministic=True)
        
        # Записуємо дію та стан машини
        action_log.append({
            "frame": frame_count,
            "steer": float(action[0]),
            "gas": float(action[1]),
            "brake": float(action[2]),
            "x": float(env.unwrapped.car_x),
            "y": float(env.unwrapped.car_y),
            "speed": float(env.unwrapped.speed),
            "angle": float(env.unwrapped.angle)
        })
        frame_count += 1
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        env.render()
        
        if terminated or truncated:
            print(f"Епізод завершено. Нагорода: {total_reward:.2f}")
            
            # Зберігаємо лог заїзду
            log_filename = f"run_log_{int(time.time())}.json"
            with open(log_filename, "w") as f:
                json.dump(action_log, f, indent=4)
            print(f"Лог збережено у файл: {log_filename}")
            
            # Скидаємо для наступного заїзду
            observation, info = env.reset()
            total_reward = 0
            action_log = []
            frame_count = 0

except KeyboardInterrupt:
    print("\nВихід за запитом користувача...")
    # Зберігаємо лог, якщо вихід відбувся посеред заїзду
    if action_log:
        log_filename = f"run_log_partial_{int(time.time())}.json"
        with open(log_filename, "w") as f:
            json.dump(action_log, f, indent=4)
        print(f"Лог незавершеного заїзду збережено у файл: {log_filename}")

finally:
    env.close()