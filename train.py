import numpy as np
from ai_controller import GeneticAlgorithm
from game_env import TankGame
import time
import os
import pygame
import argparse
import csv

def train(resume_path=None, log_file="training_log.csv"):
    # Hyperparameters
    POPULATION_SIZE = 50
    GENERATIONS = 200
    GAMES_PER_GEN = 5 # Each AI plays this many games
    NUM_PLAYERS = 4 # Number of tanks in one game
    
    INPUT_SIZE = 28
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
        # Each individual plays against random opponents
        for i in range(POPULATION_SIZE):
            total_reward = 0
            for _ in range(GAMES_PER_GEN):
                # Pick random opponents
                # We need to pick NUM_PLAYERS - 1 opponents
                opponents_indices = []
                while len(opponents_indices) < NUM_PLAYERS - 1:
                    idx = np.random.randint(POPULATION_SIZE)
                    if idx != i and idx not in opponents_indices:
                        opponents_indices.append(idx)
                
                # Play game
                # Current player is at index 0 in the game
                player_indices = [i] + opponents_indices
                nets = [ga.population[idx] for idx in player_indices]
                
                states = env.reset()
                done = False
                game_reward = 0
                
                while not done:
                    # Get actions
                    actions = [net.forward(state) for net, state in zip(nets, states)]
                    
                    # Step
                    next_states, rewards, done = env.step(actions)
                    
                    # Accumulate reward for the current player (index 0)
                    game_reward += rewards[0]
                    states = next_states
                
                total_reward += game_reward
            
            fitness_scores[i] = total_reward / GAMES_PER_GEN
        
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
    INPUT_SIZE = 28
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 3
    NUM_PLAYERS = 4
    
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
