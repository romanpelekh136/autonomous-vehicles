import gymnasium as gym
import argparse
from custom_env import CarRacingCustom

gym.envs.registration.register(
    id='CarRacingCustom-v0',
    entry_point='custom_env:CarRacingCustom',
    max_episode_steps=2000
)

def test_ppo(track_name):
    from stable_baselines3 import PPO
    model = PPO.load("models/ppo_best/best_model.zip", device="cpu")
    env = gym.make('CarRacingCustom-v0', render_mode="human", track_name=track_name)
    obs, _ = env.reset()
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

def test_neat(track_name):
    import neat
    import pickle
    with open("models/neat_best.pkl", "rb") as f:
        winner = pickle.load(f)
    
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        "neat_config.txt"
    )
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = gym.make('CarRacingCustom-v0', render_mode="human", track_name=track_name)
    obs, _ = env.reset()
    
    while True:
        output = net.activate(obs)
        action = [
            output[0],
            (output[1] + 1.0) / 2.0,
            (output[2] + 1.0) / 2.0
        ]
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ppo", "neat"], default="ppo")
    parser.add_argument("--track", default="track_02")
    args = parser.parse_args()
    
    if args.method == "ppo":
        test_ppo(args.track)
    else:
        test_neat(args.track)
