import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import sys
import os
import logging
import matplotlib.pyplot as plt
import argparse

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maze_generator import MazeGenerator

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dqn_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Running on device: {device}")

# --- Headless Maze Generator ---
class HeadlessMazeGenerator(MazeGenerator):
    def __init__(self, width, height):
        # Initialize only logic-related attributes, skip visualization
        self.width = width
        self.height = height
        self.visited = [[False for _ in range(width)] for _ in range(height)]
        self.grid_graph = {}
        for y in range(height):
            for x in range(width):
                self.grid_graph[(x, y)] = []
        # No plt figures
        self.lines = []

    def remove_wall_visual(self, x, y, direction):
        pass # Do nothing

    def update_head(self, x, y):
        pass # Do nothing

    def generate(self, algo='dfs', block=False):
        # Override to avoid UI calls
        if algo == 'dfs':
            self._dfs(0, 0)
        elif algo == 'prim':
            self._prim(0, 0)

# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, input_channels, height, width, output_dim):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.flatten_size = 64 * height * width
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

# --- Environment Wrapper ---
class MazeEnv:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.maze = None
        self.agent_pos = (0, 0)
        self.goal_pos = (width - 1, height - 1)
        self.visited_map = np.zeros((height, width), dtype=np.float32)
        self.reset()

    def reset(self):
        self.maze = HeadlessMazeGenerator(self.width, self.height)
        # Randomize algorithm to make AI robust to different maze structures
        algo = random.choice(['dfs', 'prim'])
        self.maze.generate(algo=algo)
        
        # Randomize Start and Goal Positions
        while True:
            ax, ay = random.randint(0, self.width-1), random.randint(0, self.height-1)
            gx, gy = random.randint(0, self.width-1), random.randint(0, self.height-1)
            # Ensure minimum distance to avoid trivial episodes and pollution
            if abs(ax - gx) + abs(ay - gy) > min(self.width, self.height) / 2:
                self.agent_pos = (ax, ay)
                self.goal_pos = (gx, gy)
                break
        
        self.visited_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.visited_map[self.agent_pos[1], self.agent_pos[0]] = 1.0
        return self.get_state()

    def get_state(self):
        # State: (7, H, W)
        # 0: Wall Up
        # 1: Wall Down
        # 2: Wall Left
        # 3: Wall Right
        # 4: Agent Pos
        # 5: Goal Pos
        # 6: Visited Map
        
        state = np.zeros((7, self.height, self.width), dtype=np.float32)
        
        # Fill Wall Channels
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self.maze.grid_graph.get((x, y), [])
                # If (x, y-1) NOT in neighbors, then Wall Up is 1
                if (x, y - 1) not in neighbors: state[0, y, x] = 1.0
                if (x, y + 1) not in neighbors: state[1, y, x] = 1.0
                if (x - 1, y) not in neighbors: state[2, y, x] = 1.0
                if (x + 1, y) not in neighbors: state[3, y, x] = 1.0
        
        # Agent Pos
        ax, ay = self.agent_pos
        state[4, ay, ax] = 1.0
        
        # Goal Pos
        gx, gy = self.goal_pos
        state[5, gy, gx] = 1.0
        
        # Visited Map
        state[6] = self.visited_map
        
        return state

    def step(self, action):
        x, y = self.agent_pos
        
        # 0: Up, 1: Down, 2: Left, 3: Right
        dx, dy = 0, 0
        if action == 0: dy = -1
        elif action == 1: dy = 1
        elif action == 2: dx = -1
        elif action == 3: dx = 1
        
        nx, ny = x + dx, y + dy
        
        neighbors = self.maze.grid_graph.get((x, y), [])
        
        reward = -0.1 # Increased Step penalty to encourage shorter paths
        done = False
        
        if (nx, ny) in neighbors:
            self.agent_pos = (nx, ny)
            
            # Revisit penalty
            if self.visited_map[ny, nx] > 0:
                reward = -0.5 # Penalty for revisiting
            else:
                self.visited_map[ny, nx] = 1.0
                reward = 0.05 # Small reward, but net step is still negative (-0.1 + 0.05 = -0.05)
            
            if self.agent_pos == self.goal_pos:
                reward = 20.0 # Stronger goal reward
                done = True
        else:
            reward = -1.0 # Stronger Hit wall penalty
            
        return self.get_state(), reward, done

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        logging.info(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_episode = checkpoint['episode'] + 1
            epsilon = checkpoint['epsilon']
            logging.info(f"Loaded checkpoint '{filename}' (episode {checkpoint['episode']})")
            return start_episode, epsilon
        except RuntimeError as e:
            logging.warning(f"Failed to load checkpoint due to architecture mismatch: {e}")
            logging.warning("Starting from scratch.")
            return 0, 1.0
    else:
        logging.info(f"No checkpoint found at '{filename}'")
        return 0, 1.0

# --- Training Loop ---
def train(resume=False):
    # Hyperparameters
    WIDTH, HEIGHT = 10, 10
    EPISODES = 5000 
    BATCH_SIZE = 128 # Increased batch size for smoother gradients
    GAMMA = 0.95 # Slightly increased to look a bit further ahead
    EPSILON_START = 1.0
    EPSILON_END = 0.05 # Allow more exploitation at the end
    EPSILON_DECAY = 0.9995 # Slower decay to explore more thoroughly
    LR = 0.0001 
    TARGET_UPDATE = 200 # Less frequent target updates for stability
    MEMORY_SIZE = 50000 # Larger memory to reduce correlation

    env = MazeEnv(WIDTH, HEIGHT)
    
    input_channels = 7 
    output_dim = 4 
    
    policy_net = DQN(input_channels, HEIGHT, WIDTH, output_dim).to(device)
    target_net = DQN(input_channels, HEIGHT, WIDTH, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    start_episode = 0

    if resume:
        start_episode, epsilon = load_checkpoint("checkpoint.pth", policy_net, optimizer)
        target_net.load_state_dict(policy_net.state_dict())
    
    logging.info(f"Starting training from episode {start_episode}...")
    
    rewards_history = []

    for episode in range(start_episode, EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 300: # Increased max steps
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train
            if len(replay_buffer) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                q_values = policy_net(states)
                q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_q_values = target_net(next_states)
                    next_q_value = next_q_values.max(1)[0]
                    expected_q_value = rewards + GAMMA * next_q_value * (1 - dones)
                
                loss = nn.MSELoss()(q_value, expected_q_value)
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        # Update epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        rewards_history.append(total_reward)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if episode % 50 == 0:
            logging.info(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}, Steps: {steps}")
            # Save checkpoint
            save_checkpoint({
                'episode': episode,
                'state_dict': policy_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epsilon': epsilon
            }, filename="checkpoint.pth")

    # Save model
    torch.save(policy_net.state_dict(), "maze_dqn_model.pth")
    logging.info("Training complete. Model saved to maze_dqn_model.pth")

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Training Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('training_rewards.png')
    logging.info("Training rewards plot saved to training_rewards.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN for Maze')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    train(resume=args.resume)
