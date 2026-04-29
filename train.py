import gymnasium as gym
import optuna
import numpy as np
import subprocess
import webbrowser
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from custom_env import CarRacingCustom

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func


class CheckpointMetricCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_checkpoints = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "checkpoints" in info:
                self.ep_checkpoints.append(info["checkpoints"])
                if len(self.ep_checkpoints) >= 100:
                    mean_cp = np.mean(self.ep_checkpoints)
                    self.logger.record("custom/mean_checkpoints", mean_cp)
                    self.ep_checkpoints = []
        return True

gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom',
    max_episode_steps=2000
)

OPTIMIZE = False

def optimize_ppo(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.9999)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    env = make_vec_env('CarRacingCustom-v0', n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs={"track_name": "track_02"})
    
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
    
    try:
        model.learn(total_timesteps=250000)
    except Exception as e:
        env.close()
        return -1000.0

    eval_env = make_vec_env('CarRacingCustom-v0', n_envs=1, env_kwargs={"track_name": "track_02"})
    total_checkpoints = []
    try:
        for _ in range(5):
            obs = eval_env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = eval_env.step(action)
                done = dones[0]
            total_checkpoints.append(infos[0].get("checkpoints", 0))
    finally:
        env.close()
        eval_env.close()

    return np.mean(total_checkpoints)

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
          'learning_rate': linear_schedule(0.00011308338187272194),
          'gamma': 0.9831961146510467,
          'ent_coef': 0.006260879193144504,
          'clip_range': 0.19157532502131835,
          'n_steps': 1024,
          'batch_size': 128,
          'n_epochs': 10,
        }
        
        print("Починаємо фінальне навчання з найкращими параметрами...")
        env = make_vec_env('CarRacingCustom-v0', n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs={"track_name": "track_02"})
        
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
        
        eval_env = make_vec_env('CarRacingCustom-v0', n_envs=1, env_kwargs={"track_name": "track_02"})
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path="./best_model/",
            eval_freq=25_000 // 8,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1
        )
        cp_callback = CheckpointMetricCallback()

        model.learn(total_timesteps=5000000, callback=CallbackList([eval_cb, cp_callback]))
        
        model.save("ppo_car_racing_optimized")
        print("Модель збережено!")
        eval_env.close()
        env.close()