import numpy as np
from ai_controller import GPUGeneticAlgorithm, GeneticAlgorithm
from game_env import TankGame
from vec_env import SubprocVecEnv
import time
import os
import pygame
import argparse
import json
from tqdm import tqdm
import torch
import pickle

# Wrapper for SubprocVecEnv
def make_env():
    return TankGame(render_mode=False, num_players=2)

def train(resume_path=None, log_file="training_log.jsonl", opponent_path=None):
    # Hyperparameters
    POPULATION_SIZE = 50
    GENERATIONS = 200
    GAMES_PER_GEN = 2 # Play 5 games against random opponent
    
    INPUT_SIZE = 20
    HIDDEN_SIZE_1 = 256
    HIDDEN_SIZE_2 = 64
    OUTPUT_SIZE = 3
    
    # Load opponent weights if provided
    opponent_weights = None
    if opponent_path and os.path.exists(opponent_path):
        print(f"Loading opponent from {opponent_path}...")
        try:
            with open(opponent_path, 'rb') as f:
                opponent_weights = pickle.load(f)
            print("Opponent loaded successfully.")
        except Exception as e:
            print(f"Error loading opponent: {e}")
            return

    # Use GPU GA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    ga = GPUGeneticAlgorithm(POPULATION_SIZE, INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE, device=device)
    
    start_gen = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from {resume_path}...")
        try:
            ga.load_best(resume_path)
            try:
                base_name = os.path.basename(resume_path)
                if "gen_" in base_name:
                    start_gen = int(base_name.split("gen_")[1].split(".")[0]) + 1
            except:
                pass
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Vectorized Environment
    # Determine number of parallel environments based on CPU cores
    # We want to use CPU efficiently but not overload it
    num_envs = max(1, os.cpu_count() - 2) # Leave some cores for system/GPU driving
    print(f"Starting vectorized training with {num_envs} environments...")
    
    envs = SubprocVecEnv([make_env for _ in range(num_envs)], opponent_weights=opponent_weights)
    
    for gen in range(start_gen, start_gen + GENERATIONS):
        fitness_scores = np.zeros(POPULATION_SIZE)
        gen_stats = {"hit": 0, "suicide": 0, "dead": 0, "win": 0}
        
        # We need to evaluate POPULATION_SIZE individuals
        # We have num_envs environments
        # We process them in batches
        
        with tqdm(total=POPULATION_SIZE * GAMES_PER_GEN, desc=f"Gen {gen}") as pbar:
            for i in range(0, POPULATION_SIZE, num_envs):
                # Indices of individuals in this batch
                indices = list(range(i, min(i + num_envs, POPULATION_SIZE)))
                current_batch_size = len(indices)
                
                # Reset environments for this batch
                # We only use the first current_batch_size environments
                obs = envs.reset()[:current_batch_size]
                
                # Track games played for each individual in the batch
                games_played = np.zeros(current_batch_size, dtype=int)
                batch_fitness = np.zeros(current_batch_size)
                
                # Run until all individuals in this batch have played enough games
                while np.any(games_played < GAMES_PER_GEN):
                    # Get actions from GPU
                    # obs is numpy array (batch, input)
                    # indices is list of population indices
                    # We need to map batch index 0 -> population index indices[0]
                    
                    # Convert obs to tensor on GPU
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    
                    # Get actions for the specific individuals
                    # We use forward_subset
                    actions_tensor = ga.pop.forward_subset(obs_tensor, indices)
                    actions = actions_tensor.cpu().numpy()
                    
                    # Step environments
                    # We need to step ALL envs, but we only care about the first current_batch_size
                    # SubprocVecEnv expects actions for all envs
                    # We can pad actions with zeros/random
                    full_actions = np.zeros((num_envs, OUTPUT_SIZE))
                    full_actions[:current_batch_size] = actions
                    
                    next_obs, rewards, dones, infos = envs.step(full_actions)
                    
                    # Process results
                    for k in range(current_batch_size):
                        if games_played[k] < GAMES_PER_GEN:
                            batch_fitness[k] += rewards[k]
                            
                            # Update stats
                            for key in gen_stats:
                                gen_stats[key] += infos[k].get(key, 0)
                            
                            if dones[k]:
                                games_played[k] += 1
                                pbar.update(1)
                    
                    obs = next_obs[:current_batch_size]
                
                # Update population fitness
                fitness_scores[indices] = batch_fitness / GAMES_PER_GEN

        # Stats
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        avg_fitness = np.mean(fitness_scores)
        
        print(f"Gen {gen}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
        
        # Log
        best_weights = ga.pop.get_best_weights(best_idx)
        serializable_weights = {k: v.tolist() for k, v in best_weights.items()}
        
        log_entry = {
            "generation": gen,
            "best_fitness": float(best_fitness),
            "avg_fitness": float(avg_fitness),
            "timestamp": time.time(),
            "stats": {k: v / (POPULATION_SIZE * GAMES_PER_GEN) for k, v in gen_stats.items()},
            "best_weights": serializable_weights
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        if gen % 10 == 0:
            ga.save_best(f"best_ai_gen_{gen}.pkl")
            
        ga.evolve(fitness_scores)

    ga.save_best("best_ai_final.pkl")
    envs.close()
    print("Training complete!")

def watch_game(model_path="best_ai_final.pkl", opponent_path=None):
    INPUT_SIZE = 20
    HIDDEN_SIZE_1 = 256
    HIDDEN_SIZE_2 = 256
    OUTPUT_SIZE = 3
    
    # Use CPU for watching
    ga = GeneticAlgorithm(1, INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE)
    try:
        ga.load_best(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print("Could not load model, using random weights")
    
    net = ga.population[0]
    
    # Load opponent if provided
    opponent_net = None
    if opponent_path and os.path.exists(opponent_path):
        try:
            # Create another GA instance just to load weights easily
            ga_opp = GeneticAlgorithm(1, INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE)
            ga_opp.load_best(opponent_path)
            opponent_net = ga_opp.population[0]
            print(f"Loaded opponent from {opponent_path}")
        except Exception as e:
            print(f"Could not load opponent: {e}")
            
    env = TankGame(render_mode=True, num_players=2)
    
    while True:
        states = env.reset()
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            action_ai = net.forward(states[0])
            
            if opponent_net:
                action_opponent = opponent_net.forward(states[1])
            else:
                action_opponent = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(0, 1)]
            
            states, rewards, done, infos = env.step([action_ai, action_opponent])
            env.render()
            env.timer.tick(60)

if __name__ == "__main__":
    try:
        pygame.quit()
    except:
        pass
        
    parser = argparse.ArgumentParser(description="Train or Watch Tank AI")
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "watch"], help="Mode: train or watch")
    parser.add_argument("--resume", type=str, help="Path to .pkl file to resume training from")
    parser.add_argument("--log", type=str, default="training_log.jsonl", help="Path to log file")
    parser.add_argument("--model", type=str, default="best_ai_final.pkl", help="Model path for watch mode")
    parser.add_argument("--opponent", type=str, help="Path to .pkl file for opponent AI")
    
    args = parser.parse_args()
    
    if args.mode == "watch":
        watch_game(args.model, args.opponent)
    else:
        train(resume_path=args.resume, log_file=args.log, opponent_path=args.opponent)
