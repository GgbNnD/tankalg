import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
import random

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maze_generator import MazeGenerator
from train import DQN

def get_state(agent_pos, goal_pos, width, height, grid_graph, visited_map):
    # State: (7, H, W)
    # 0: Wall Up
    # 1: Wall Down
    # 2: Wall Left
    # 3: Wall Right
    # 4: Agent Pos
    # 5: Goal Pos
    # 6: Visited Map
    
    state = np.zeros((7, height, width), dtype=np.float32)
    
    # Fill Wall Channels
    for y in range(height):
        for x in range(width):
            neighbors = grid_graph.get((x, y), [])
            # If (x, y-1) NOT in neighbors, then Wall Up is 1
            if (x, y - 1) not in neighbors: state[0, y, x] = 1.0
            if (x, y + 1) not in neighbors: state[1, y, x] = 1.0
            if (x - 1, y) not in neighbors: state[2, y, x] = 1.0
            if (x + 1, y) not in neighbors: state[3, y, x] = 1.0
    
    # Agent Pos
    ax, ay = agent_pos
    state[4, ay, ax] = 1.0
    
    # Goal Pos
    gx, gy = goal_pos
    state[5, gy, gx] = 1.0
    
    # Visited Map
    state[6] = visited_map
    
    return state

def test_agent():
    # Parameters (must match training)
    WIDTH, HEIGHT = 10, 10
    MODEL_PATH = "maze_dqn_model.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        return

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Note: Input channels=7, Height=10, Width=10, Output=4
    model = DQN(7, HEIGHT, WIDTH, 4).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("It seems the model architecture has changed. Please retrain the model using 'python train.py'.")
        return

    model.eval()
    print("Model loaded successfully.")

    # Generate Maze
    print("Generating Maze...")
    # Use block=False so we can run our own loop
    maze = MazeGenerator(WIDTH, HEIGHT)
    maze.generate(algo='dfs', block=False)
    maze.title.set_text("DQN Agent Testing (CNN)")

    # Randomize Start and Goal for testing
    while True:
        ax, ay = random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1)
        gx, gy = random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1)
        if (ax, ay) != (gx, gy):
            agent_pos = (ax, ay)
            goal_pos = (gx, gy)
            break
            
    print(f"Test Start: {agent_pos}, Goal: {goal_pos}")

    visited_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    visited_map[agent_pos[1], agent_pos[0]] = 1.0
    
    # Draw Goal (Red 'E' is default, but let's highlight our random goal)
    # Clear default start/end text if needed, but for now just add new markers
    goal_patch = plt.Circle(
        (goal_pos[0] + 0.5, maze.height - goal_pos[1] - 0.5), 
        0.3, color='red', zorder=19
    )
    maze.ax.add_patch(goal_patch)

    # Draw agent (Green Circle)
    agent_patch = plt.Circle(
        (agent_pos[0] + 0.5, maze.height - agent_pos[1] - 0.5), 
        0.3, color='green', zorder=20
    )
    maze.ax.add_patch(agent_patch)
    
    print("Starting navigation...")
    steps = 0
    max_steps = 100
    
    while agent_pos != goal_pos and steps < max_steps:
        # Get state
        state = get_state(agent_pos, goal_pos, WIDTH, HEIGHT, maze.grid_graph, visited_map)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Predict action
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item()
            
        # Execute action
        x, y = agent_pos
        dx, dy = 0, 0
        if action == 0: dy = -1    # Up
        elif action == 1: dy = 1   # Down
        elif action == 2: dx = -1  # Left
        elif action == 3: dx = 1   # Right
        
        nx, ny = x + dx, y + dy
        
        # Check validity
        neighbors = maze.grid_graph.get((x, y), [])
        if (nx, ny) in neighbors:
            agent_pos = (nx, ny)
            visited_map[ny, nx] = 1.0
            print(f"Step {steps}: Moved to {agent_pos}")
        else:
            print(f"Step {steps}: Hit wall at {agent_pos} trying to go {['Up','Down','Left','Right'][action]}")
            
        # Update visualization
        agent_patch.center = (agent_pos[0] + 0.5, maze.height - agent_pos[1] - 0.5)
        maze.fig.canvas.draw()
        maze.fig.canvas.flush_events()
        
        steps += 1
        time.sleep(0.2) # Slow down to see movement

    if agent_pos == goal_pos:
        maze.title.set_text("Success! Goal Reached!")
        print("Success! Goal Reached!")
    else:
        maze.title.set_text("Failed: Max steps reached")
        print("Failed: Max steps reached")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_agent()
