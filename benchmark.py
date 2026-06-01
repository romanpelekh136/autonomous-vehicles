import time
import argparse
import numpy as np
import gymnasium as gym
from custom_env import CarRacingCustom

gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom',
    max_episode_steps=2000
)

def benchmark_ppo(track_name, n_episodes=20, add_noise=False):
    from stable_baselines3 import PPO
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.1,
    }
    model = PPO.load("models/ppo_best/best_model.zip", device="cpu", custom_objects=custom_objects)
    results = []
    
    for _ in range(n_episodes):
        env = gym.make('CarRacingCustom-v0', track_name=track_name)
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            
            if add_noise:
                action[0] = np.clip(action[0] + np.random.normal(0, 0.05), -1.0, 1.0)
                
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                results.append({
                    "progress": info.get("progress", 0),
                    "crashed": info.get("crashed", False)
                })
                break
        env.close()
    return results

def benchmark_neat(track_name, n_episodes=20, add_noise=False):
    import neat, pickle
    with open("models/neat_best/best_model.pkl", "rb") as f:
        winner = pickle.load(f)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.txt")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    results = []
    
    for _ in range(n_episodes):
        env = gym.make('CarRacingCustom-v0', track_name=track_name)
        obs, _ = env.reset()
        while True:
            output = net.activate(obs)
            steer = output[0]
            
            if add_noise:
                steer = np.clip(steer + np.random.normal(0, 0.05), -1.0, 1.0)
                
            action = [steer, max(0.0, output[1]), max(0.0, output[2])]
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                results.append({
                    "progress": info.get("progress", 0),
                    "crashed": info.get("crashed", False)
                })
                break
        env.close()
    return results

def measure_inference():
    """Порівнює швидкість прийняття рішень PPO та NEAT."""
    import neat, pickle
    from stable_baselines3 import PPO
    
    obs = np.zeros(26, dtype=np.float32)
    
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.1,
    }
    model = PPO.load("models/ppo_best/best_model.zip", device="cpu", custom_objects=custom_objects)
    start = time.time()
    for _ in range(1000):
        model.predict(obs, deterministic=True)
    ppo_ms = (time.time() - start)
    
    with open("models/neat_best/best_model.pkl", "rb") as f:
        winner = pickle.load(f)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.txt")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    start = time.time()
    for _ in range(1000):
        net.activate(obs)
    neat_ms = (time.time() - start)
    
    print(f"\nInference speed (1000 calls):")
    print(f"PPO:  {ppo_ms*1000:.1f}ms total | {ppo_ms:.4f}ms/call")
    print(f"NEAT: {neat_ms*1000:.1f}ms total | {neat_ms:.4f}ms/call")
    print(f"NEAT is {ppo_ms/neat_ms:.1f}x faster")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PPO and NEAT models")
    parser.add_argument("--noise", action="store_true", help="Add random steering noise (robustness test)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes per track")
    args = parser.parse_args()

    tracks = ["track_01", "track_02", "track_03", "track_04"]
    
    mode_text = "NOISY (Robustness test)" if args.noise else "CLEAN"
    print(f"=== BENCHMARK: {mode_text} ===")
    
    if not args.noise:
        measure_inference()
    
    for track in tracks:
        ppo_res = benchmark_ppo(track, n_episodes=args.episodes, add_noise=args.noise)
        neat_res = benchmark_neat(track, n_episodes=args.episodes, add_noise=args.noise)
        
        ppo_prog = [r["progress"] for r in ppo_res]
        neat_prog = [r["progress"] for r in neat_res]
        
        crashes_ppo = sum(1 for r in ppo_res if r["crashed"])
        crashes_neat = sum(1 for r in neat_res if r["crashed"])
        
        print(f"\nTrack: {track} | Episodes: {args.episodes}")
        print(f"{'Metric':<20} {'PPO':>10} {'NEAT':>10}")
        print("-" * 42)
        print(f"{'Mean progress':<20} {np.mean(ppo_prog):>10.2f} {np.mean(neat_prog):>10.2f}")
        print(f"{'Max progress':<20} {np.max(ppo_prog):>10.2f} {np.max(neat_prog):>10.2f}")
        print(f"{'Min progress':<20} {np.min(ppo_prog):>10.2f} {np.min(neat_prog):>10.2f}")
        print(f"{'Std deviation':<20} {np.std(ppo_prog):>10.2f} {np.std(neat_prog):>10.2f}")
        print(f"{'Crashes':<20} {crashes_ppo:>10} {crashes_neat:>10}")