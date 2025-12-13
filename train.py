import numpy as np
from ai_controller import GeneticAlgorithm, NeuralNetwork
from game_env import TankGame
import time
import os
import pygame
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Helper function for parallel execution
def evaluate_single_game(weights):
    # Create environment inside the process
    # We use 2 players: AI (index 0) vs Random (index 1)
    # Suppress pygame output
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    
    env = TankGame(render_mode=False, num_players=2)
    
    # Reconstruct AI
    ai = NeuralNetwork(20, 64, 3)
    ai.set_weights(weights)
    
    states = env.reset()
    done = False
    total_reward = 0
    stats = {"hit": 0, "suicide": 0, "dead": 0, "win": 0}
    
    while not done:
        # AI Action
        action_ai = ai.forward(states[0])
        
        # Random Opponent Action
        # [move, turn, shoot]
        # move: -1 to 1, turn: -1 to 1, shoot: 0 to 1
        action_random = [
            np.random.uniform(-1, 1), 
            np.random.uniform(-1, 1), 
            np.random.uniform(0, 1)
        ]
        
        actions = [action_ai, action_random]
        
        next_states, rewards, done, infos = env.step(actions)
        
        total_reward += rewards[0]
        
        # Accumulate stats for AI (player 0)
        for k in stats:
            stats[k] += infos[0].get(k, 0)
            
        states = next_states
        
    return total_reward, stats

def train(resume_path=None, log_file="training_log.jsonl"):
    # Hyperparameters
    POPULATION_SIZE = 50
    GENERATIONS = 200
    GAMES_PER_GEN = 5 # Play 5 games against random opponent to average noise
    
    INPUT_SIZE = 20
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 3
    
    ga = GeneticAlgorithm(POPULATION_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    
    start_gen = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}...")
        try:
            ga.load_best(resume_path)
            best_weights = ga.population[0].get_weights()
            for i in range(1, POPULATION_SIZE):
                ga.population[i].set_weights(best_weights)
                ga.population[i].mutate(mutation_rate=0.2, mutation_scale=0.3)
            
            try:
                base_name = os.path.basename(resume_path)
                if "gen_" in base_name:
                    start_gen = int(base_name.split("gen_")[1].split(".")[0]) + 1
            except:
                pass
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    print(f"Starting parallel training with population {POPULATION_SIZE} for {GENERATIONS} generations...")
    
    for gen in range(start_gen, start_gen + GENERATIONS):
        fitness_scores = np.zeros(POPULATION_SIZE)
        gen_stats = {"hit": 0, "suicide": 0, "dead": 0, "win": 0}
        
        # Prepare tasks: Each individual plays GAMES_PER_GEN games
        tasks = []
        for i in range(POPULATION_SIZE):
            weights = ga.population[i].get_weights()
            for _ in range(GAMES_PER_GEN):
                tasks.append((i, weights))
        
        # Run tasks in parallel
        # Use max_workers=None (defaults to cpu_count)
        # User requested to use half of the CPU resources
        max_workers = max(1, (os.cpu_count() or 1) // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(evaluate_single_game, t[1]) for t in tasks]
            
            # Collect results with progress bar
            for i, future in tqdm(enumerate(futures), total=len(futures), desc=f"Gen {gen}"):
                try:
                    reward, stats = future.result()
                    pop_idx = tasks[i][0]
                    fitness_scores[pop_idx] += reward
                    
                    # Aggregate stats for the whole generation
                    for k in stats:
                        gen_stats[k] += stats[k]
                except Exception as e:
                    print(f"Error in game execution: {e}")

        # Average fitness
        fitness_scores /= GAMES_PER_GEN
        
        # Stats
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        avg_fitness = np.mean(fitness_scores)
        
        print(f"Gen {gen}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
        
        # Prepare log entry
        best_weights = ga.population[best_idx].get_weights()
        # Convert weights to list for JSON serialization
        serializable_weights = {k: v.tolist() for k, v in best_weights.items()}
        
        log_entry = {
            "generation": gen,
            "best_fitness": float(best_fitness),
            "avg_fitness": float(avg_fitness),
            "timestamp": time.time(),
            "stats": {k: v / (POPULATION_SIZE * GAMES_PER_GEN) for k, v in gen_stats.items()}, # Average per game
            "best_weights": serializable_weights
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Save best pickle as well (for easy loading)
        if gen % 10 == 0:
            ga.save_best(f"best_ai_gen_{gen}.pkl")
            
        # Evolve
        ga.evolve(fitness_scores)

    ga.save_best("best_ai_final.pkl")
    print("Training complete!")

def watch_game(model_path="best_ai_final.pkl"):
    INPUT_SIZE = 20
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 3
    
    ga = GeneticAlgorithm(1, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    try:
        ga.load_best(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print("Could not load model, using random weights")
    
    net = ga.population[0]
    env = TankGame(render_mode=True, num_players=2)
    
    while True:
        states = env.reset()
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            action_ai = net.forward(states[0])
            action_random = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(0, 1)]
            
            states, rewards, done, infos = env.step([action_ai, action_random])
            env.render()
            env.timer.tick(60)

if __name__ == "__main__":
    # Fix for multiprocessing
    try:
        pygame.quit()
    except:
        pass
        
    parser = argparse.ArgumentParser(description="Train or Watch Tank AI")
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "watch"], help="Mode: train or watch")
    parser.add_argument("--resume", type=str, help="Path to .pkl file to resume training from")
    parser.add_argument("--log", type=str, default="training_log.jsonl", help="Path to log file")
    parser.add_argument("--model", type=str, default="best_ai_final.pkl", help="Model path for watch mode")
    
    args = parser.parse_args()
    
    if args.mode == "watch":
        watch_game(args.model)
    else:
        train(resume_path=args.resume, log_file=args.log)
