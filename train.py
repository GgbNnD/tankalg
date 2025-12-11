import numpy as np
from ai_controller import GeneticAlgorithm
from game_env import TankGame
import time
import os
import pygame

def train():
    # Hyperparameters
    POPULATION_SIZE = 50
    GENERATIONS = 100
    GAMES_PER_GEN = 5 # Each AI plays this many games
    
    INPUT_SIZE = 28
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 3
    
    ga = GeneticAlgorithm(POPULATION_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    env = TankGame(render_mode=False)
    
    print(f"Starting training with population {POPULATION_SIZE} for {GENERATIONS} generations...")
    
    for gen in range(GENERATIONS):
        fitness_scores = np.zeros(POPULATION_SIZE)
        
        # Evaluate population
        # Each individual plays against random opponents
        for i in range(POPULATION_SIZE):
            total_reward = 0
            for _ in range(GAMES_PER_GEN):
                # Pick random opponent
                opponent_idx = np.random.randint(POPULATION_SIZE)
                while opponent_idx == i:
                    opponent_idx = np.random.randint(POPULATION_SIZE)
                
                # Play game
                p1_net = ga.population[i]
                p2_net = ga.population[opponent_idx]
                
                state1, state2 = env.reset()
                done = False
                game_reward = 0
                
                while not done:
                    # Get actions
                    action1 = p1_net.forward(state1)
                    action2 = p2_net.forward(state2)
                    
                    # Step
                    next_state1, next_state2, rewards, done = env.step(action1, action2)
                    
                    game_reward += rewards[0]
                    state1, state2 = next_state1, next_state2
                
                total_reward += game_reward
            
            fitness_scores[i] = total_reward / GAMES_PER_GEN
        
        # Stats
        best_fitness = np.max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        print(f"Gen {gen+1}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
        
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
    
    ga = GeneticAlgorithm(2, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    try:
        ga.load_best(model_path)
        print(f"Loaded model from {model_path}")
    except:
        print("Could not load model, using random weights")
    
    # Use the same model for both players
    p1_net = ga.population[0]
    p2_net = ga.population[0] # Self-play
    
    env = TankGame(render_mode=True)
    
    while True:
        state1, state2 = env.reset()
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            action1 = p1_net.forward(state1)
            action2 = p2_net.forward(state2)
            
            state1, state2, rewards, done = env.step(action1, action2)
            env.render()
            env.timer.tick(60) # Limit FPS for watching

if __name__ == "__main__":
    import sys
    import pygame # Import here to access pygame.event in watch_game
    
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        watch_game()
    else:
        train()
