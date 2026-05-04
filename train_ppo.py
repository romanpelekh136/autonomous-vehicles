import gymnasium as gym
import optuna
import numpy as np
import random
import subprocess
import webbrowser
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from custom_env import CarRacingCustom
from stable_baselines3.common.monitor import Monitor

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


class CheckpointMetricCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_progress = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "progress" in info:
                self.ep_progress.append(info["progress"])
                if len(self.ep_progress) >= 100:
                    mean_prog = np.mean(self.ep_progress)
                    self.logger.record("custom/mean_progress", mean_prog)
                    self.ep_progress = []
        return True

gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom',
    max_episode_steps=2000
)
def make_env():
    tracks = [ "track_02", "track_03", "track_04"]
    track = random.choice(tracks)
    def _init():
        return Monitor(gym.make('CarRacingCustom-v0', track_name=track))
    return _init
OPTIMIZE = False

def optimize_ppo(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.01, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    target_kl = trial.suggest_float("target_kl", 0.01, 0.05)
    n_epochs = trial.suggest_int("n_epochs", 3, 15)

    env = SubprocVecEnv([make_env() for _ in range(8)])

    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_steps=n_steps,
        batch_size=batch_size,
        target_kl=target_kl,
        n_epochs=n_epochs,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[512, 512])),
        verbose=0,
        device="cpu"
    )

    try:
        model.learn(total_timesteps=1_000_000)
    except Exception as e:
        env.close()
        return -10000.0

    eval_env = SubprocVecEnv([
        lambda: gym.make('CarRacingCustom-v0', track_name="track_02"),
        lambda: gym.make('CarRacingCustom-v0', track_name="track_03"),
        lambda: gym.make('CarRacingCustom-v0', track_name="track_04")
    ])
    
    total_progress = []
    try:
        for _ in range(3): # 3 епізоди на кожну трасу
            obs = eval_env.reset()
            dones = np.array([False, False, False])
            episode_progresses = [0.0] * 3
            
            while not all(dones):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, current_dones, infos = eval_env.step(action)
                
                for i in range(3):
                    if not dones[i]:
                        if "progress" in infos[i]:
                            episode_progresses[i] = infos[i]["progress"]
                        if current_dones[i]:
                            dones[i] = True
                            
            total_progress.extend(episode_progresses)
    finally:
        env.close()
        eval_env.close()

    return np.mean(total_progress) if total_progress else 0.0

if __name__ == '__main__':
    if OPTIMIZE:
        print("Запуск Optuna. Шукаємо ідеальні параметри...")
        study = optuna.create_study(direction="maximize")
        study.optimize(optimize_ppo, n_trials=50, n_jobs=1) 
        
        print("\n=== Оптимізація завершена ===")
        print("Найкращі параметри знайдені:")
        print(study.best_params)
        
    else:
        best_params = {
          'learning_rate': linear_schedule(0.00011308338187272194),  # старий
          'gamma': 0.983,           # старий — довший горизонт
          'ent_coef': 0.0035,       # від Optuna
          'clip_range': 0.176,      # від Optuna
          'n_steps': 1024,          # старий
          'batch_size': 128,        # старий
          'target_kl': 0.0196,      # від Optuna — новий корисний
          'n_epochs': 10,
        }
        
        print("Починаємо фінальне навчання з найкращими параметрами...")
        env = SubprocVecEnv([make_env() for _ in range(8)])
        
        model = PPO("MlpPolicy", env, verbose=1, device="cpu",
                    tensorboard_log="./tb_logs/",
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[512, 512])),
                    **best_params)
        
        tb_process = subprocess.Popen(
            ["tensorboard", "--logdir=./tb_logs/", "--port=6006", "--reload_interval=5"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        webbrowser.open("http://localhost:6006")
        print("TensorBoard запущено: http://localhost:6006")
        
        eval_env = SubprocVecEnv([
            lambda: gym.make('CarRacingCustom-v0', track_name="track_02"),
            lambda: gym.make('CarRacingCustom-v0', track_name="track_03"),
            lambda: gym.make('CarRacingCustom-v0', track_name="track_04")
        ])
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path="models/ppo_best/",
            eval_freq=25_000 // 8,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1
        )
        cp_callback = CheckpointMetricCallback()

        model.learn(total_timesteps=5000000, callback=CallbackList([eval_cb, cp_callback]))
        
        model.save("models/ppo_final")
        print("Модель збережено!")
        eval_env.close()
        env.close()