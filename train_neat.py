import neat
import pickle
import os
import multiprocessing
from custom_env import CarRacingCustom
from torch.utils.tensorboard import SummaryWriter
from neat.reporting import BaseReporter
import subprocess
import webbrowser

class BestGenomeSaver(BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        os.makedirs("models", exist_ok=True)
        with open("models/neat_best.pkl", "wb") as f:
            pickle.dump(best_genome, f)

class TensorBoardReporter(BaseReporter):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.generation = 0
        self.best_fitness = -float('inf')
        os.makedirs("models/neat_best", exist_ok=True)

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [c.fitness for c in population.values() if c.fitness is not None]
        if fitnesses:
            mean_fitness = sum(fitnesses) / len(fitnesses)
            
            # Нагороди
            self.writer.add_scalar("NEAT/mean_fitness", mean_fitness, self.generation)
            self.writer.add_scalar("NEAT/best_fitness", best_genome.fitness, self.generation)
            
            # Різноманітність (кількість видів)
            self.writer.add_scalar("NEAT/species_count", len(species.species), self.generation)
            
            # Складність нейромережі (як вона росте з часом)
            avg_nodes = sum(len(g.nodes) for g in population.values()) / len(population)
            avg_conns = sum(len(g.connections) for g in population.values()) / len(population)
            self.writer.add_scalar("NEAT/avg_nodes", avg_nodes, self.generation)
            self.writer.add_scalar("NEAT/avg_connections", avg_conns, self.generation)
            
            # Збереження найкращої моделі
            if best_genome.fitness is not None and best_genome.fitness > self.best_fitness:
                self.best_fitness = best_genome.fitness
                with open("models/neat_best/best_model.pkl", "wb") as f:
                    pickle.dump(best_genome, f)
                print(f"\n[+] Нова найкраща модель NEAT збережена (Нагорода: {self.best_fitness:.2f})\n")
            
        self.generation += 1

def eval_genome(genome, config):
    tracks = ["track_02", "track_03", "track_04"]
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_reward = 0
    total_progress = 0
    
    for track in tracks:
        env = CarRacingCustom(track_name=track)
        obs, _ = env.reset()
        info = {}
        
        steps_without_cp = 0
        checkpoints_reached = 0
        
        for _ in range(2000):
            output = net.activate(obs)
            
            action = [
                output[0],
                max(0.0, output[1]),
                max(0.0, output[2])
            ]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            current_cp = info.get("checkpoints", 0)
            if current_cp > checkpoints_reached:
                checkpoints_reached = current_cp
                steps_without_cp = 0
            else:
                steps_without_cp += 1
                
            if steps_without_cp > 400:
                total_reward -= 100.0
                break
            
            if terminated or truncated:
                break
        total_progress += info.get("progress", 0)
        env.close()
        
    genome.avg_progress = total_progress / len(tracks)
    return total_reward / len(tracks)

def run_neat():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neat_config.txt"
    )
    
    population = neat.Population(config)
    
    # Стандартний вивід у консоль
    population.add_reporter(neat.StdOutReporter(True))
    
    population.add_reporter(BestGenomeSaver())
    
    # Збір внутрішньої статистики NEAT
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Знаходимо наступний номер для папки логів NEAT
    run_num = 1
    while os.path.exists(f"./tb_logs/NEAT_{run_num}"):
        run_num += 1
    run_dir = f"./tb_logs/NEAT_{run_num}"
    
    # Наш новий репортер для графіків
    population.add_reporter(TensorBoardReporter(run_dir))
    
    # Запуск TensorBoard
    tb_process = subprocess.Popen(
        ["tensorboard", "--logdir=./tb_logs/", "--port=6006", "--reload_interval=5"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    webbrowser.open("http://localhost:6006")
    print("TensorBoard запущено: http://localhost:6006")
    
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = population.run(pe.evaluate, n=500)
    
    os.makedirs("models", exist_ok=True)
    with open("models/neat_final.pkl", "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    run_neat()