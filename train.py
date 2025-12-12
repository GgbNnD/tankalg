import numpy as np
from ai_controller import GeneticAlgorithm
from game_env import TankGame
import time
import os
import pygame
import argparse
import csv
from tqdm import tqdm

def train(resume_path=None, log_file="training_log.csv"):
    # Hyperparameters
    POPULATION_SIZE = 50
    GENERATIONS = 200
    GAMES_PER_GEN = 1 # Deterministic environment, one game is enough
    NUM_PLAYERS = 50 # Train all agents in parallel
    
    INPUT_SIZE = 16 # Reduced input size
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 3
    
    ga = GeneticAlgorithm(POPULATION_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    
    start_gen = 0
    if resume_path:
        if os.path.exists(resume_path):
            print(f"Resuming training from {resume_path}...")
            try:
                ga.load_best(resume_path)
                # Seed the population with the loaded best model to jumpstart
                # We keep the first one (loaded) as is
                # We mutate the rest to create a diverse population around the best solution
                best_weights = ga.population[0].get_weights()
                for i in range(1, POPULATION_SIZE):
                    ga.population[i].set_weights(best_weights)
                    # Apply stronger mutation to explore around the optimum
                    ga.population[i].mutate(mutation_rate=0.2, mutation_scale=0.3)
                
                # Try to guess generation number from filename if possible
                # e.g., best_ai_gen_10.pkl
                try:
                    base_name = os.path.basename(resume_path)
                    if "gen_" in base_name:
                        start_gen = int(base_name.split("gen_")[1].split(".")[0]) + 1
                except:
                    pass
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print(f"Checkpoint {resume_path} not found. Starting from scratch.")

    env = TankGame(render_mode=False, num_players=NUM_PLAYERS)
    
    # Initialize log file
    if not os.path.exists(log_file):
        with open(log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Best Fitness", "Avg Fitness", "Timestamp"])
    
    print(f"Starting training with population {POPULATION_SIZE} for {GENERATIONS} generations...")
    
    for gen in range(start_gen, start_gen + GENERATIONS):
        fitness_scores = np.zeros(POPULATION_SIZE)
        
        # Evaluate population
        games_played = np.zeros(POPULATION_SIZE)
        total_rewards = np.zeros(POPULATION_SIZE)
        
        # Estimate total matches needed
        total_matches_needed = int(np.ceil(POPULATION_SIZE * GAMES_PER_GEN / NUM_PLAYERS))
        
        with tqdm(total=total_matches_needed, desc=f"Gen {gen}", unit="game") as pbar:
            # Continue playing until everyone has played at least GAMES_PER_GEN games
            while np.min(games_played) < GAMES_PER_GEN:
                # Prioritize those who haven't played enough
                candidates = [i for i in range(POPULATION_SIZE) if games_played[i] < GAMES_PER_GEN]
                
                if len(candidates) >= NUM_PLAYERS:
                    player_indices = np.random.choice(candidates, NUM_PLAYERS, replace=False)
                else:
                    # Not enough candidates, fill with others
                    others = [i for i in range(POPULATION_SIZE) if i not in candidates]
                    fillers = np.random.choice(others, NUM_PLAYERS - len(candidates), replace=False)
                    player_indices = np.concatenate([candidates, fillers])
                    player_indices = player_indices.astype(int)

                # Play game
                nets = [ga.population[idx] for idx in player_indices]
                
                states = env.reset()
                done = False
                game_rewards = np.zeros(NUM_PLAYERS)
                
                while not done:
                    # Get actions
                    actions = [net.forward(state) for net, state in zip(nets, states)]
                    
                    # Step
                    next_states, rewards, done = env.step(actions)
                    
                    # Accumulate reward
                    game_rewards += rewards
                    states = next_states
                
                # Update stats for all players in this game
                for idx_in_game, population_idx in enumerate(player_indices):
                    total_rewards[population_idx] += game_rewards[idx_in_game]
                    games_played[population_idx] += 1
                
                pbar.update(1)
            
        fitness_scores = total_rewards / games_played
        
        # Stats
        best_fitness = np.max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        print(f"Gen {gen}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
        
        # Log stats
        with open(log_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([gen, best_fitness, avg_fitness, time.time()])
        
        # Save best
        if gen % 10 == 0:
            ga.save_best(f"best_ai_gen_{gen}.pkl")
            
        # Evolve
        ga.evolve(fitness_scores)

    # Save final model
    ga.save_best("best_ai_final.pkl")
    print("Training complete!")

def watch_game(model_path="best_ai_final.pkl"):
    # Watch two AIs play
    INPUT_SIZE = 16
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 3
    NUM_PLAYERS = 50
    
    ga = GeneticAlgorithm(2, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    try:
        ga.load_best(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print("Could not load model, using random weights")
    
    # Use the same model for all players
    nets = [ga.population[0] for _ in range(NUM_PLAYERS)]
    
    env = TankGame(render_mode=True, num_players=NUM_PLAYERS)
    
    while True:
        states = env.reset()
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            actions = [net.forward(state) for net, state in zip(nets, states)]
            
            states, rewards, done = env.step(actions)
            env.render()
            env.timer.tick(60) # Limit FPS for watching

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Watch Tank AI")
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "watch"], help="Mode: train or watch")
    parser.add_argument("--resume", type=str, help="Path to .pkl file to resume training from")
    parser.add_argument("--log", type=str, default="training_log.csv", help="Path to log file")
    parser.add_argument("--model", type=str, default="best_ai_final.pkl", help="Model path for watch mode")
    
    args = parser.parse_args()
    
    if args.mode == "watch":
        watch_game(args.model)
    else:
        train(resume_path=args.resume, log_file=args.log)
