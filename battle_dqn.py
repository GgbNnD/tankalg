import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'None'
import pickle

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maze_generator import MazeGenerator
from maze_game import MazeGame, AI, Entity
from train import DQN

class DQN_AI(AI):
    def __init__(self, x, y, color, ax, maze_height, maze_width, model_path):
        super().__init__(x, y, color, ax, maze_height, maze_width)
        
        # Initialize model (7 channels, output 4 actions)
        self.model = DQN(7, maze_height, maze_width, 4)
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    params = pickle.load(f)
                self.model.set_params(params)
                print("DQN Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Model file not found: {model_path}")
            
        # Memory for DQN (Visited Map)
        self.visited_map = np.zeros((maze_height, maze_width), dtype=np.float32)
        self.visited_map[y, x] = 1.0
        
        # To prevent getting stuck in loops when chasing a moving target,
        # we might need to decay the visited map or reset it occasionally.
        # For now, we'll implement a simple decay mechanism.
        self.step_counter = 0

    def get_state(self, grid_graph, player_pos):
        # Construct 7-channel state
        # 0-3: Walls, 4: Self, 5: Target(Player), 6: Visited
        h, w = self.maze_height, self.maze_width
        state = np.zeros((7, h, w), dtype=np.float32)
        
        # Walls (Static, could be cached but fast enough to compute)
        for y in range(h):
            for x in range(w):
                neighbors = grid_graph.get((x, y), [])
                if (x, y - 1) not in neighbors: state[0, y, x] = 1.0 # Up
                if (x, y + 1) not in neighbors: state[1, y, x] = 1.0 # Down
                if (x - 1, y) not in neighbors: state[2, y, x] = 1.0 # Left
                if (x + 1, y) not in neighbors: state[3, y, x] = 1.0 # Right
        
        # Self
        state[4, self.y, self.x] = 1.0
        
        # Target (Player)
        px, py = player_pos
        state[5, py, px] = 1.0
        
        # Visited
        state[6] = self.visited_map
        
        return state

    def update(self, grid_graph, player, player_bullet):
        if not self.alive:
            return

        # 1. Dodge Logic (Inherited from AI, but we call it explicitly here to be sure)
        if player_bullet and player_bullet.active:
            if self.is_in_danger(player_bullet, grid_graph):
                safe_move = self.find_safe_move(grid_graph, player_bullet)
                if safe_move:
                    self.move(safe_move[0], safe_move[1], grid_graph, [player])
                    # Reset visited map on dodge to allow re-evaluating paths
                    self.visited_map.fill(0)
                    self.visited_map[self.y, self.x] = 1.0
                    return

        # 2. Attack Logic
        if self.can_shoot_player(grid_graph, player):
            dx = 1 if player.x > self.x else -1 if player.x < self.x else 0
            dy = 1 if player.y > self.y else -1 if player.y < self.y else 0
            if dx != 0: dy = 0
            self.facing = (dx, dy)
            self.shoot()
            return

        # 3. DQN Pathfinding (Replaces A*)
        state = self.get_state(grid_graph, (player.x, player.y))
        state_batch = state[np.newaxis, :] # (1, C, H, W)
        
        q_values = self.model.forward(state_batch)
        action = np.argmax(q_values)
            
        # Map action to move
        dx, dy = 0, 0
        if action == 0: dy = -1   # Up
        elif action == 1: dy = 1  # Down
        elif action == 2: dx = -1 # Left
        elif action == 3: dx = 1  # Right
        
        # Try to move
        nx, ny = self.x + dx, self.y + dy
        neighbors = grid_graph.get((self.x, self.y), [])
        
        if (nx, ny) in neighbors:
             self.move(dx, dy, grid_graph, [player])
             self.visited_map[self.y, self.x] = 1.0
        else:
            # DQN tried to hit a wall. Fallback to random valid move to avoid getting stuck.
            if neighbors:
                nx, ny = random.choice(neighbors)
                self.move(nx - self.x, ny - self.y, grid_graph, [player])
                self.visited_map[self.y, self.x] = 1.0
        
        # Decay visited map slowly to allow revisiting old areas eventually
        self.step_counter += 1
        if self.step_counter % 10 == 0:
            self.visited_map *= 0.8
            # Ensure current pos is still marked
            self.visited_map[self.y, self.x] = 1.0

class BattleGame(MazeGame):
    def __init__(self, width, height, model_path):
        # Initialize MazeGenerator
        self.maze = MazeGenerator(width, height)
        self.maze.generate(algo='prim', block=False)
        self.maze.title.set_text("DQN Battle! You: Blue vs AI: Red")
        
        self.player = Entity(0, 0, 'blue', self.maze.ax, height, width)
        # Use DQN_AI
        self.ai = DQN_AI(width-1, height-1, 'red', self.maze.ax, height, width, model_path)
        
        self.maze.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.game_over = False
        self.restart_requested = False

        # Timer
        self.timer = self.maze.fig.canvas.new_timer(interval=200)
        self.timer.add_callback(self.update)
        self.timer.start()

if __name__ == "__main__":
    print("欢迎来到 DQN 迷宫大逃杀！")
    try:
        w = int(input("请输入迷宫宽度 (建议 10-15): "))
        h = int(input("请输入迷宫高度 (建议 10-15): "))
    except ValueError:
        print("输入无效，使用默认值 10x10")
        w, h = 10, 10

    model_path = "maze_dqn_model.pkl"
    if not os.path.exists(model_path):
        print("警告：未找到模型文件 maze_dqn_model.pkl，AI 可能无法正常移动。")

    while True:
        game = BattleGame(w, h, model_path)
        game.run()
        if not game.restart_requested:
            break
    print("游戏结束")
