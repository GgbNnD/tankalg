import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'None'
import pickle

# 将当前目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maze_generator import MazeGenerator
from maze_game import MazeGame, AI, Entity
from train import DQN

class DQN_AI(AI):
    def __init__(self, x, y, color, ax, maze_height, maze_width, model_path):
        super().__init__(x, y, color, ax, maze_height, maze_width)
        
        # 初始化模型 (7 个通道，输出 4 个动作)
        self.model = DQN(7, maze_height, maze_width, 4)
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    params = pickle.load(f)
                self.model.set_params(params)
                print("DQN 模型加载成功。")
            except Exception as e:
                print(f"加载模型时出错: {e}")
        else:
            print(f"未找到模型文件: {model_path}")
            
        # DQN 的记忆 (访问地图)
        self.visited_map = np.zeros((maze_height, maze_width), dtype=np.float32)
        self.visited_map[y, x] = 1.0
        
        # 为了防止在追逐移动目标时陷入循环，
        # 我们可能需要衰减访问地图或偶尔重置它。
        # 目前，我们将实现一个简单的衰减机制。
        self.step_counter = 0

    def get_state(self, grid_graph, player_pos):
        # 构建 7 通道状态
        # 0-3: 墙壁, 4: 自身, 5: 目标(玩家), 6: 已访问
        h, w = self.maze_height, self.maze_width
        state = np.zeros((7, h, w), dtype=np.float32)
        
        # 墙壁 (静态，可以缓存，但计算速度足够快)
        for y in range(h):
            for x in range(w):
                neighbors = grid_graph.get((x, y), [])
                if (x, y - 1) not in neighbors: state[0, y, x] = 1.0 # 上
                if (x, y + 1) not in neighbors: state[1, y, x] = 1.0 # 下
                if (x - 1, y) not in neighbors: state[2, y, x] = 1.0 # 左
                if (x + 1, y) not in neighbors: state[3, y, x] = 1.0 # 右
        
        # 自身
        state[4, self.y, self.x] = 1.0
        
        # 目标 (玩家)
        px, py = player_pos
        state[5, py, px] = 1.0
        
        # 已访问
        state[6] = self.visited_map
        
        return state

    def update(self, grid_graph, player, player_bullet):
        if not self.alive:
            return

        # 1. 躲避逻辑 (继承自 AI，但我们在这里显式调用以确保)
        if player_bullet and player_bullet.active:
            if self.is_in_danger(player_bullet, grid_graph):
                safe_move = self.find_safe_move(grid_graph, player_bullet)
                if safe_move:
                    self.move(safe_move[0], safe_move[1], grid_graph, [player])
                    # 躲避时重置访问地图以允许重新评估路径
                    self.visited_map.fill(0)
                    self.visited_map[self.y, self.x] = 1.0
                    return

        # 2. 攻击逻辑
        if self.can_shoot_player(grid_graph, player):
            dx = 1 if player.x > self.x else -1 if player.x < self.x else 0
            dy = 1 if player.y > self.y else -1 if player.y < self.y else 0
            if dx != 0: dy = 0
            self.facing = (dx, dy)
            self.shoot()
            return

        # 3. DQN 寻路 (替代 A*)
        state = self.get_state(grid_graph, (player.x, player.y))
        state_batch = state[np.newaxis, :] # (1, C, H, W)
        
        q_values = self.model.forward(state_batch)
        action = np.argmax(q_values)
            
        # 将动作映射到移动
        dx, dy = 0, 0
        if action == 0: dy = -1   # 上
        elif action == 1: dy = 1  # 下
        elif action == 2: dx = -1 # 左
        elif action == 3: dx = 1  # 右
        
        # 尝试移动
        nx, ny = self.x + dx, self.y + dy
        neighbors = grid_graph.get((self.x, self.y), [])
        
        if (nx, ny) in neighbors:
             self.move(dx, dy, grid_graph, [player])
             self.visited_map[self.y, self.x] = 1.0
        else:
            # DQN 试图撞墙。回退到随机有效移动以避免卡住。
            if neighbors:
                nx, ny = random.choice(neighbors)
                self.move(nx - self.x, ny - self.y, grid_graph, [player])
                self.visited_map[self.y, self.x] = 1.0
        
        # 缓慢衰减访问地图以允许最终重新访问旧区域
        self.step_counter += 1
        if self.step_counter % 10 == 0:
            self.visited_map *= 0.8
            # 确保当前位置仍被标记
            self.visited_map[self.y, self.x] = 1.0

class BattleGame(MazeGame):
    def __init__(self, width, height, model_path):
        # 初始化迷宫生成器
        self.maze = MazeGenerator(width, height)
        self.maze.generate(algo='prim', block=False)
        self.maze.title.set_text("DQN 对战！你: 蓝色 vs AI: 红色")
        
        self.player = Entity(0, 0, 'blue', self.maze.ax, height, width)
        # 使用 DQN_AI
        self.ai = DQN_AI(width-1, height-1, 'red', self.maze.ax, height, width, model_path)
        
        self.maze.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.game_over = False
        self.restart_requested = False

        # 计时器
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
