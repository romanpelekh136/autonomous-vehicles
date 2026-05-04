import gymnasium as gym
import argparse
import time
import json
from custom_env import CarRacingCustom

gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom',
    max_episode_steps=2000
)

def log_action(env_unwrapped, action, frame_count):
    return {
        "frame": frame_count,
        "steer": float(action[0]),
        "gas": float(action[1]),
        "brake": float(action[2]),
        "x": float(env_unwrapped.car_x),
        "y": float(env_unwrapped.car_y),
        "speed": float(env_unwrapped.speed),
        "angle": float(env_unwrapped.angle)
    }

def save_log(action_log, is_partial=False):
    prefix = "run_log_partial" if is_partial else "run_log"
    log_filename = f"{prefix}_{int(time.time())}.json"
    with open(log_filename, "w") as f:
        json.dump(action_log, f, indent=4)
    print(f"Лог збережено у файл: {log_filename}")

def test_ppo(track_name):
    from stable_baselines3 import PPO
    model = PPO.load("models/ppo_best/best_model.zip", device="cpu")
    env = gym.make('CarRacingCustom-v0', render_mode="human", track_name=track_name)
    obs, _ = env.reset()
    
    total_reward = 0
    action_log = []
    frame_count = 0
    
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action_log.append(log_action(env.unwrapped, action, frame_count))
            frame_count += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            
            if terminated or truncated:
                print(f"Епізод завершено. Загальна нагорода: {total_reward:.2f}")
                save_log(action_log)
                obs, _ = env.reset()
                total_reward = 0
                action_log = []
                frame_count = 0
    except KeyboardInterrupt:
        print("\nВихід...")
        if action_log:
            save_log(action_log, is_partial=True)
        env.close()

def test_neat(track_name):
    import neat
    import pickle
    with open("models/neat_best/best_model.pkl", "rb") as f:
        winner = pickle.load(f)
    
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        "neat_config.txt"
    )
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = gym.make('CarRacingCustom-v0', render_mode="human", track_name=track_name)
    obs, _ = env.reset()
    
    total_reward = 0
    action_log = []
    frame_count = 0
    
    try:
        while True:
            output = net.activate(obs)
            action = [
                output[0],
                max(0.0, output[1]),
                max(0.0, output[2])
            ]
            action_log.append(log_action(env.unwrapped, action, frame_count))
            frame_count += 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            
            if terminated or truncated:
                print(f"Епізод завершено. Загальна нагорода: {total_reward:.2f}")
                save_log(action_log)
                obs, _ = env.reset()
                total_reward = 0
                action_log = []
                frame_count = 0
    except KeyboardInterrupt:
        print("\nВихід...")
        if action_log:
            save_log(action_log, is_partial=True)
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ppo", "neat"], default="ppo")
    parser.add_argument("--track", default="track_02")
    args = parser.parse_args()
    
    if args.method == "ppo":
        test_ppo(args.track)
    else:
        test_neat(args.track)
