import gymnasium as gym
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from custom_env import CarRacingCustom

# Реєструємо середовище
gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom',
    max_episode_steps=2000
)

# Перемикач режимів:
# True - шукаємо ідеальні параметри (Optuna)
# False - тренуємо модель на максимум з готовими параметрами
OPTIMIZE = False

def optimize_ppo(trial):
    # 1. Optuna пропонує випадкові гіперпараметри для цієї спроби
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.9999)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # 2. Створюємо паралельне середовище
    env = make_vec_env('CarRacingCustom-v0', n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs={"track_name": "track_01"})
    
    # 3. Ініціалізуємо модель з параметрами від Optuna
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=0,
        device="cpu"
    )
    
    # 4. Швидке тренування (збільшено до 150 000 кроків для адекватної оцінки)
    try:
        model.learn(total_timesteps=150000)
    except Exception as e:
        env.close()
        return -1000.0

    # 5. Оцінюємо наскільки добре вона навчилася (тестуємо 5 епізодів)
    eval_env = gym.make('CarRacingCustom-v0', track_name="track_01")
    try:
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    finally:
        env.close()
        eval_env.close()
    
    # Optuna буде намагатися максимізувати цей результат
    return mean_reward

if __name__ == '__main__':
    if OPTIMIZE:
        print("Запуск Optuna. Шукаємо ідеальні параметри...")
        # Створюємо "дослідження"
        study = optuna.create_study(direction="maximize")
        
        # Запускаємо 50 спроб (trials)
        # n_jobs=1 тому що одне тренування вже використовує всі 8 ядер через SubprocVecEnv
        study.optimize(optimize_ppo, n_trials=50, n_jobs=1) 
        
        print("\n=== Оптимізація завершена ===")
        print("Найкращі параметри знайдені:")
        print(study.best_params)
        
    else:
        # Твій звичайний блок для фінального довгого навчання
        # (Впиши сюди найкращі параметри після оптимізації)
        best_params = {
            'learning_rate': 0.0001417252777348529, 
            'gamma': 0.984848197853803, 
            'ent_coef': 0.005347766258285577, 
            'clip_range': 0.11224583877971807, 
            'n_steps': 1024, 
            'batch_size': 64
        }
        
        print("Починаємо фінальне навчання з найкращими параметрами...")
        env = make_vec_env('CarRacingCustom-v0', n_envs=8, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", **best_params)
        
        model.learn(total_timesteps=5000000)
        
        model.save("ppo_car_racing_optimized")
        print("Модель збережено!")
        env.close()