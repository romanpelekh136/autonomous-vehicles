"""
collect_data.py — Збирає всі дані для аналізу в Jupyter-ноутбуку.

Використання:
    python collect_data.py

Зібрані дані:
    - Бенчмарк (progress, crashes, std) на 4 трасах - чисті умови
    - Бенчмарк з шумом steering - тест стійкості
    - Траєкторії (x, y, speed, steer, gas, brake) - 3 епізоди на трасу
    - Crash-мапа (x, y) з шумних запусків
    - Швидкість інференсу
    - Метадані тренування (архітектура, гіперпараметри)

Результат: ./analysis_data/
    benchmark_clean.json
    benchmark_noisy.json
    trajectories_ppo.json
    trajectories_neat.json
    crash_map.json
    inference_speed.json
    training_metadata.json
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import gymnasium as gym
from custom_env import CarRacingCustom
from tqdm import tqdm

os.makedirs("analysis_data", exist_ok=True)

gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom',
    max_episode_steps=2000
)

TRACKS = ["track_01", "track_02", "track_03", "track_04"]
N_BENCHMARK_EPISODES = 20  # епізодів на трасу для бенчмарку
N_TRAJ_EPISODES = 3        # епізодів на трасу для траєкторій
N_CRASH_EPISODES = 30      # епізодів на трасу для crash-мапи


# ─── ЗАВАНТАЖЕННЯ МОДЕЛЕЙ ─────────────────────────────────────────────────────────────

def load_ppo():
    from stable_baselines3 import PPO
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.1,
    }
    return PPO.load("models/ppo_best/best_model.zip", device="cpu", custom_objects=custom_objects)

def load_neat():
    import neat, pickle
    with open("models/neat_best/best_model.pkl", "rb") as f:
        winner = pickle.load(f)
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        "neat_config.txt"
    )
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    return net, winner  


# ─── ОДИН ЕПІЗОД ──────────────────────────────────────────────────────────────────

def run_episode_ppo(model, track_name, add_noise=False, save_trajectory=False):
    """Один епізод з PPO. Можна зберігати траєкторію та додавати шум."""
    env = gym.make('CarRacingCustom-v0', track_name=track_name)
    obs, _ = env.reset()
    trajectory = [] if save_trajectory else None
    crash_pos = None
    frame = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        if add_noise:
            action[0] = np.clip(action[0] + np.random.normal(0, 0.15), -1.0, 1.0)

        if save_trajectory:
            u = env.unwrapped
            trajectory.append({
                "frame": frame,
                "x": float(u.car_x),
                "y": float(u.car_y),
                "speed": float(u.speed),
                "angle": float(u.angle),
                "steer": float(action[0]),
                "gas":   float(action[1]),
                "brake": float(action[2]),
            })

        obs, _, terminated, truncated, info = env.step(action)
        frame += 1
        
        if terminated or truncated:
            if info.get("crashed", False):
                u = env.unwrapped
                crash_pos = {"x": float(u.car_x), "y": float(u.car_y)}
            break

    env.close()
    return {
        "progress": float(info.get("progress", 0)),
        "crashed":  bool(info.get("crashed", False)),
        "crash_pos": crash_pos,
        "checkpoints": int(info.get("checkpoints", 0)),
        "frames": frame,
        "trajectory": trajectory,
    }


def run_episode_neat(net, track_name, add_noise=False, save_trajectory=False):
    """Один епізод з NEAT. Можна зберігати траєкторію та додавати шум."""
    env = gym.make('CarRacingCustom-v0', track_name=track_name)
    obs, _ = env.reset()
    trajectory = [] if save_trajectory else None
    crash_pos = None
    frame = 0

    while True:
        output = net.activate(obs)
        steer = output[0]
        if add_noise:
            steer = np.clip(steer + np.random.normal(0, 0.15), -1.0, 1.0)
        action = [steer, max(0.0, output[1]), max(0.0, output[2])]

        if save_trajectory:
            u = env.unwrapped
            trajectory.append({
                "frame": frame,
                "x": float(u.car_x),
                "y": float(u.car_y),
                "speed": float(u.speed),
                "angle": float(u.angle),
                "steer": float(action[0]),
                "gas":   float(action[1]),
                "brake": float(action[2]),
            })

        obs, _, terminated, truncated, info = env.step(action)
        frame += 1
        
        if terminated or truncated:
            if info.get("crashed", False):
                u = env.unwrapped
                crash_pos = {"x": float(u.car_x), "y": float(u.car_y)}
            break

    env.close()
    return {
        "progress": float(info.get("progress", 0)),
        "crashed":  bool(info.get("crashed", False)),
        "crash_pos": crash_pos,
        "checkpoints": int(info.get("checkpoints", 0)),
        "frames": frame,
        "trajectory": trajectory,
    }


# ─── БЕНЧМАРК ───────────────────────────────────────────────────────────────────

def run_benchmark(ppo_model, neat_net, add_noise=False):
    """Запускає бенчмарк на всіх трасах і зберігає результати."""
    results = {}
    mode = "noisy" if add_noise else "clean"
    total_eps = len(TRACKS) * N_BENCHMARK_EPISODES
    print(f"\n=== BENCHMARK [{mode.upper()}] ===")
    print(f"  Total episodes: {total_eps}")

    for track in TRACKS:
        ppo_episodes, neat_episodes = [], []

        bar = tqdm(
            range(N_BENCHMARK_EPISODES),
            desc=f"  {track}",
            unit="ep",
            dynamic_ncols=True,
            file=sys.stdout,
            colour="cyan",
        )
        for ep in bar:
            ppo_ep  = run_episode_ppo(ppo_model, track, add_noise=add_noise)
            neat_ep = run_episode_neat(neat_net,  track, add_noise=add_noise)
            ppo_episodes.append(ppo_ep)
            neat_episodes.append(neat_ep)
            bar.set_postfix({
                "PPO": f"{ppo_ep['progress']:.2f}",
                "NEAT": f"{neat_ep['progress']:.2f}",
            }, refresh=True)

        ppo_prog  = [e["progress"]  for e in ppo_episodes]
        neat_prog = [e["progress"]  for e in neat_episodes]

        results[track] = {
            "ppo": {
                "mean_progress": float(np.mean(ppo_prog)),
                "max_progress":  float(np.max(ppo_prog)),
                "std_progress":  float(np.std(ppo_prog)),
                "crash_rate":    float(sum(e["crashed"] for e in ppo_episodes) / N_BENCHMARK_EPISODES),
            },
            "neat": {
                "mean_progress": float(np.mean(neat_prog)),
                "max_progress":  float(np.max(neat_prog)),
                "std_progress":  float(np.std(neat_prog)),
                "crash_rate":    float(sum(e["crashed"] for e in neat_episodes) / N_BENCHMARK_EPISODES),
            }
        }

    filename = f"analysis_data/benchmark_{mode}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {filename}")
    return results


# ─── ТРАЄКТОРІЇ ──────────────────────────────────────────────────────────────────

def collect_trajectories(ppo_model, neat_net):
    """Збирає повні траєкторії для візуалізації."""
    print(f"\n=== COLLECTING TRAJECTORIES ===")
    ppo_data, neat_data = {}, {}

    for track in TRACKS:
        ppo_data[track]  = []
        neat_data[track] = []

        bar = tqdm(range(N_TRAJ_EPISODES), desc=f"  {track}", unit="ep")
        for ep in bar:
            ppo_ep  = run_episode_ppo(ppo_model, track, save_trajectory=True)
            neat_ep = run_episode_neat(neat_net,  track, save_trajectory=True)
            ppo_data[track].append({"trajectory": ppo_ep["trajectory"]})
            neat_data[track].append({"trajectory": neat_ep["trajectory"]})

    with open("analysis_data/trajectories_ppo.json",  "w") as f:
        json.dump(ppo_data, f)
    with open("analysis_data/trajectories_neat.json", "w") as f:
        json.dump(neat_data, f)
    print("  Saved: analysis_data/trajectories_ppo.json")
    print("  Saved: analysis_data/trajectories_neat.json")


# ─── CRASH-МАПА ───────────────────────────────────────────────────────────────────

def collect_crash_map(ppo_model, neat_net):
    """Запускає епізоди з шумом і збирає позиції аварій."""
    print(f"\n=== COLLECTING CRASH MAP (NOISY) ===")
    crash_data = {}

    for track in TRACKS:
        crash_data[track] = {"ppo": [], "neat": []}

        for ep in tqdm(range(N_CRASH_EPISODES), desc=f"  {track}"):
            ppo_ep  = run_episode_ppo(ppo_model, track, add_noise=True, save_trajectory=True)
            neat_ep = run_episode_neat(neat_net,  track, add_noise=True, save_trajectory=True)
            
            if ppo_ep["crashed"]:
                crash_data[track]["ppo"].append({"pos": ppo_ep["crash_pos"], "trajectory": ppo_ep["trajectory"]})
            if neat_ep["crashed"]:
                crash_data[track]["neat"].append({"pos": neat_ep["crash_pos"], "trajectory": neat_ep["trajectory"]})

    with open("analysis_data/crash_map.json", "w") as f:
        json.dump(crash_data, f)
    print("  Saved: analysis_data/crash_map.json")


# ─── ІНФЕРЕНС ────────────────────────────────────────────────────────────────────

def measure_inference(ppo_model, neat_net):
    """Вимірює швидкість інференсу для обох моделей."""
    print("\n=== INFERENCE SPEED ===")
    obs = np.zeros(26, dtype=np.float32)
    N = 5000

    t0 = time.perf_counter()
    for _ in range(N):
        ppo_model.predict(obs, deterministic=True)
    ppo_total = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N):
        neat_net.activate(obs)
    neat_total = time.perf_counter() - t0

    data = {
        "n_calls": N,
        "ppo":  {"total_s": ppo_total,  "per_call_ms": (ppo_total / N) * 1000},
        "neat": {"total_s": neat_total, "per_call_ms": (neat_total / N) * 1000},
        "speedup_neat_over_ppo": ppo_total / neat_total,
    }
    print(f"  PPO:  {(ppo_total / N) * 1000:.4f} ms/call")
    print(f"  NEAT: {(neat_total / N) * 1000:.4f} ms/call")
    print(f"  NEAT is {ppo_total/neat_total:.1f}x faster")

    with open("analysis_data/inference_speed.json", "w") as f:
        json.dump(data, f, indent=2)
    print("  Saved: analysis_data/inference_speed.json")
    return data


# ─── МЕТАДАНІ ТРЕНУВАННЯ ─────────────────────────────────────────────────────────────

def save_training_metadata(neat_winner):
    """Зберігає архітектуру та метадані тренування."""
    ppo_total_interactions = 5_000_000 * 8
    ppo_training_time_min  = 40.27
    ppo_params = (26 * 256 + 256) + (256 * 256 + 256) + \
                 (26 * 512 + 512) + (512 * 512 + 512) + \
                 (256 * 3 + 3) + (512 * 1 + 1)

    n_nodes = len(neat_winner.nodes)
    n_connections = len([c for c in neat_winner.connections.values() if c.enabled])
    neat_total_interactions = 385 * 150 * 3 * 600

    data = {
        "ppo": {
            "total_timesteps":          5_000_000,
            "n_envs":                   8,
            "total_env_interactions":   ppo_total_interactions,
            "training_time_min":        ppo_training_time_min,
            "n_parameters":             ppo_params,
            "architecture":             "MlpPolicy pi=[256,256] vf=[512,512]",
            "optimizer":                "Adam (Optuna-tuned)",
        },
        "neat": {
            "generations":              385,
            "population_size":          150,
            "total_env_interactions":   neat_total_interactions,
            "winner_nodes":             n_nodes,
            "winner_connections":       n_connections,
            "architecture":             "FeedForward, no hidden layers initially",
            "activation":               "tanh",
        }
    }

    with open("analysis_data/training_metadata.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\n  Saved: analysis_data/training_metadata.json")
    return data


# ─── ОСНОВНИЙ ЗАПУСК ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Пропустити бенчмарк")
    parser.add_argument("--skip-trajectories", action="store_true",
                        help="Пропустити збір траєкторій")
    parser.add_argument("--skip-crashes", action="store_true",
                        help="Пропустити збір crash-мапи")
    args = parser.parse_args()

    script_start = time.perf_counter()

    print("Loading models...")
    ppo_model = load_ppo()
    neat_net, neat_winner = load_neat()
    print("Models loaded.")

    total_benchmark_eps = 0 if args.skip_benchmark else len(TRACKS) * N_BENCHMARK_EPISODES * 2
    total_traj_eps      = 0 if args.skip_trajectories else len(TRACKS) * N_TRAJ_EPISODES * 2
    total_crash_eps     = 0 if args.skip_crashes else len(TRACKS) * N_CRASH_EPISODES * 2
    
    if not args.skip_benchmark:
        run_benchmark(ppo_model, neat_net, add_noise=False)
        run_benchmark(ppo_model, neat_net, add_noise=True)

    if not args.skip_trajectories:
        collect_trajectories(ppo_model, neat_net)

    if not args.skip_crashes:
        collect_crash_map(ppo_model, neat_net)

    measure_inference(ppo_model, neat_net)
    save_training_metadata(neat_winner)

    elapsed = time.perf_counter() - script_start
    mins, secs = divmod(int(elapsed), 60)
    print(f"\nAll data collected in {mins}m {secs}s.")