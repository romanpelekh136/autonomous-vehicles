import gymnasium as gym
import optuna
import subprocess
import webbrowser
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from custom_env import CarRacingCustom

# Лінійне зменшення learning_rate від initial до 0
def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

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
        device="cuda"
    )
    
    # 4. Швидке тренування (збільшено до 150 000 кроків для адекватної оцінки)
    try:
        model.learn(total_timesteps=250000)
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
          'learning_rate': linear_schedule(3e-4),
          'gamma': 0.99,
          'ent_coef': 0.001,
          'clip_range': 0.2,
          'n_steps': 512,
          'batch_size': 64,
          'target_kl': 0.01
        }
        
        print("Починаємо фінальне навчання з найкращими параметрами...")
        raw_env = make_vec_env('CarRacingCustom-v0', n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs={"track_name": "track_01"})
        
        # VecNormalize: нормалізує нагороди до маленьких значень (спостереження вже нормалізовані)
        env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        
        model = PPO("MlpPolicy", env, verbose=1, device="cuda",
                    tensorboard_log="./tb_logs/",
                    policy_kwargs=dict(net_arch=[256, 256]),
                    **best_params)
        
        # Запускаємо TensorBoard автоматично у фоні
        tb_process = subprocess.Popen(
            ["tensorboard", "--logdir=./tb_logs/", "--port=6006"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        webbrowser.open("http://localhost:6006")
        print("TensorBoard запущено: http://localhost:6006")
        
        # Зберігаємо найкращу модель автоматично
        eval_raw = make_vec_env('CarRacingCustom-v0', n_envs=1, env_kwargs={"track_name": "track_01"})
        eval_env = VecNormalize(eval_raw, norm_obs=False, norm_reward=True, clip_reward=10.0)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path="./best_model/",
            eval_freq=25_000 // 8,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1
        )
        
        model.learn(total_timesteps=1000000, callback=eval_cb)
        
        model.save("ppo_car_racing_optimized")
        print("Модель збережено!")
        eval_env.close()
        env.close()