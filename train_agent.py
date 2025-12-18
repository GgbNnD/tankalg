import numpy as np
import random
import matplotlib.pyplot as plt
from maze_generator import MazeGenerator
from maze_game import Entity, AI, MazeGame

import os

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重 (使用标准正态分布)
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        # 前向传播
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.tanh(self.z1) # 激活函数
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2 # 返回原始输出 (Logits)，在外部进行处理

    def save(self, filename):
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        
    def load(self, filename):
        data = np.load(filename)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']

    def get_weights(self):
        return np.concatenate([self.w1.flatten(), self.b1.flatten(), self.w2.flatten(), self.b2.flatten()])

    def set_weights(self, weights):
        # 从扁平数组恢复权重
        w1_end = self.input_size * self.hidden_size
        self.w1 = weights[:w1_end].reshape(self.input_size, self.hidden_size)
        
        b1_end = w1_end + self.hidden_size
        self.b1 = weights[w1_end:b1_end]
        
        w2_end = b1_end + self.hidden_size * self.output_size
        self.w2 = weights[b1_end:w2_end].reshape(self.hidden_size, self.output_size)
        
        self.b2 = weights[w2_end:]

    def mutate(self, rate=0.1, strength=0.5):
        weights = self.get_weights()
        mask = np.random.random(weights.shape) < rate
        noise = np.random.randn(weights.shape[0]) * strength
        weights[mask] += noise[mask]
        self.set_weights(weights)

class GeneticAgent(Entity):
    def __init__(self, x, y, color, ax, maze_height, maze_width, brain=None):
        super().__init__(x, y, color, ax, maze_height, maze_width)
        self.last_move = (0, 0) # 记录上一步的移动方向
        # 输入增加 2 维: 上一步的 dx, dy
        input_size = 12 
        hidden_size = 24 
        output_size = 5 
        
        if brain:
            self.brain = brain
        else:
            self.brain = NeuralNetwork(input_size, hidden_size, output_size)

    def move(self, dx, dy, grid_graph, opponents=None):
        old_x, old_y = self.x, self.y
        super().move(dx, dy, grid_graph, opponents)
        # 如果位置发生了变化，更新 last_move
        if self.x != old_x or self.y != old_y:
            self.last_move = (dx, dy)
        else:
            # 如果撞墙没动，重置 last_move，或者保持不变也可以，这里选择重置表示"没能移动"
            self.last_move = (0, 0)

    def get_state(self, grid_graph, opponent, opponent_bullet):
        w, h = self.maze_width, self.maze_height
        
        # 相对位置
        dx = (opponent.x - self.x) / w
        dy = (opponent.y - self.y) / h
        
        # 危险感知 (敌方子弹相对位置)
        b_dx = 0
        b_dy = 0
        if opponent_bullet and opponent_bullet.active:
            b_dx = (opponent_bullet.x - self.x) / w
            b_dy = (opponent_bullet.y - self.y) / h
        
        # 墙壁检测
        neighbors = grid_graph.get((self.x, self.y), [])
        can_up = 1 if (self.x, self.y - 1) in neighbors else 0
        can_down = 1 if (self.x, self.y + 1) in neighbors else 0
        can_left = 1 if (self.x - 1, self.y) in neighbors else 0
        can_right = 1 if (self.x + 1, self.y) in neighbors else 0
        
        state = np.array([
            dx,
            dy,
            1 - can_up,   # Wall Up
            1 - can_down, # Wall Down
            1 - can_left, # Wall Left
            1 - can_right,# Wall Right
            1 if self.bullet and self.bullet.active else 0,
            1 if opponent_bullet and opponent_bullet.active else 0,
            b_dx,
            b_dy,
            self.last_move[0], # 上一步 X 方向
            self.last_move[1]  # 上一步 Y 方向
        ])
        return state

    def decide(self, grid_graph, opponent, opponent_bullet):
        state = self.get_state(grid_graph, opponent, opponent_bullet)
        outputs = self.brain.forward(state)
        
        # 分离移动和射击的输出
        # 0:Up, 1:Down, 2:Left, 3:Right
        move_logits = outputs[:4]
        # 4:Shoot
        shoot_logit = outputs[4]
        
        # --- 处理移动 ---
        # 获取当前位置的邻居，判断哪些方向可行
        neighbors = grid_graph.get((self.x, self.y), [])
        
        # 初始化 mask
        mask = np.array([0.0, 0.0, 0.0, 0.0])
        if (self.x, self.y - 1) in neighbors: mask[0] = 1.0 # Up
        if (self.x, self.y + 1) in neighbors: mask[1] = 1.0 # Down
        if (self.x - 1, self.y) in neighbors: mask[2] = 1.0 # Left
        if (self.x + 1, self.y) in neighbors: mask[3] = 1.0 # Right
        
        # Softmax 计算移动概率
        exp_scores = np.exp(move_logits - np.max(move_logits))
        probs = exp_scores / np.sum(exp_scores)
        
        # 应用掩码
        masked_probs = probs * mask
        
        # 选择移动方向
        if np.sum(masked_probs) > 0:
            action = np.argmax(masked_probs)
            if action == 0: # Up
                self.move(0, -1, grid_graph, [opponent])
            elif action == 1: # Down
                self.move(0, 1, grid_graph, [opponent])
            elif action == 2: # Left
                self.move(-1, 0, grid_graph, [opponent])
            elif action == 3: # Right
                self.move(1, 0, grid_graph, [opponent])
        
        # --- 处理射击 ---
        # 独立于移动，只要输出大于 0 就射击
        if shoot_logit > 0:
            self.shoot()

class DummyAI(Entity):
    def update(self, grid_graph, player, player_bullet):
        pass

class HeadlessGame:
    def __init__(self, width, height, brain):
        self.width = width
        self.height = height
        self.maze = MazeGenerator(width, height, visualize=False)
        self.maze.generate(algo='prim', block=False)
        # 关闭图形
        # plt.close(self.maze.fig) # 不再需要，因为 visualize=False 时没有创建 fig
        
        # 创建无图形实体
        self.agent = GeneticAgent(0, 0, 'blue', None, height, width, brain)
        # 使用 DummyAI (静态靶子) 代替 AI，降低初期训练难度
        self.opponent = DummyAI(width-1, height-1, 'red', None, height, width)
        
        self.steps = 0
        self.max_steps = 400 # 限制每局步数

    def run(self):
        # 运行一局游戏，返回 fitness
        # Fitness = 存活时间 + (击杀 * 100) - (被击杀 * 50)
        fitness = 0
        visited = set()
        visited.add((self.agent.x, self.agent.y))
        
        while self.steps < self.max_steps:
            self.steps += 1
            
            # 1. Agent 决策
            self.agent.decide(self.maze.grid_graph, self.opponent, self.opponent.bullet)
            
            # 奖励探索：如果移动到了新位置
            if (self.agent.x, self.agent.y) not in visited:
                fitness += 5 # 探索奖励
                visited.add((self.agent.x, self.agent.y))
            else:
                fitness -= 2 # 惩罚重复访问，防止原地徘徊
            
            # (已移除每帧距离奖励，改为终局结算，防止陷入局部最优)
            # 这样 AI 就不会因为需要绕路（暂时远离目标）而被惩罚

            # 2. Opponent 决策 (使用原来的 AI 逻辑)
            self.opponent.update(self.maze.grid_graph, self.agent, self.agent.bullet)
            
            # 3. 更新子弹
            for _ in range(2):
                if self.agent.bullet:
                    self.agent.bullet.move(self.maze.grid_graph, [self.opponent])
                if self.opponent.bullet:
                    self.opponent.bullet.move(self.maze.grid_graph, [self.agent])
            
            # 4. 检查状态
            if not self.agent.alive:
                fitness -= 50 # 死亡惩罚
                break
            
            if not self.opponent.alive:
                fitness += 100 # 击杀奖励
                # fitness += (self.max_steps - self.steps) * 0.5 # 快速击杀奖励
                break
                
            # fitness -= 0.2 # 存活奖励 (降低，避免为了刷分而苟活)

        # 游戏结束后的距离奖励
        final_dist = abs(self.agent.x - self.opponent.x) + abs(self.agent.y - self.opponent.y)
        # 给予较大的权重 (例如 5.0)，鼓励最终停留在靠近目标的地方
        # 迷宫宽+高约为 20，最大奖励约 100 分，与击杀奖励相当
        fitness += (self.width + self.height - final_dist) * 5.0

        return fitness

import csv

def train():
    POPULATION_SIZE = 100
    GENERATIONS = 80 # 增加代数
    WIDTH, HEIGHT = 10, 10 # 训练时用小一点的迷宫加快速度
    
    # 初始化种群 (注意 hidden_size 必须与 GeneticAgent 中一致)
    population = [NeuralNetwork(12, 24, 5) for _ in range(POPULATION_SIZE)]
    
    print(f"开始训练... 种群大小: {POPULATION_SIZE}, 代数: {GENERATIONS}")
    
    # 打开 CSV 文件准备写入
    with open('training_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Generation', 'Best Fitness', 'Average Fitness'])

        for gen in range(GENERATIONS):
            scores = []
            for i, brain in enumerate(population):
                # 每个个体玩 3 局取平均分，减少随机性
                game_scores = []
                for _ in range(3):
                    game = HeadlessGame(WIDTH, HEIGHT, brain)
                    game_scores.append(game.run())
                avg_score = sum(game_scores) / len(game_scores)
                scores.append((avg_score, brain))
                
            # 统计
            avg_fitness = sum(s[0] for s in scores) / len(scores)
            
            # 排序
            scores.sort(key=lambda x: x[0], reverse=True)
            best_score = scores[0][0]
            
            print(f"Generation {gen+1}: Best Fitness = {best_score:.2f}, Avg Fitness = {avg_fitness:.2f}")
            writer.writerow([gen+1, best_score, avg_fitness])
            
            # 优胜劣汰
            top_performers = [s[1] for s in scores[:5]] # 选前5名
            
            # 繁殖下一代
            new_population = []
            # 保留精英
            new_population.extend(top_performers)
            
            while len(new_population) < POPULATION_SIZE:
                parent = random.choice(top_performers)
                # 克隆并变异
                child = NeuralNetwork(12, 24, 5)
                child.set_weights(parent.get_weights())
                child.mutate(rate=0.2, strength=0.5)
                new_population.append(child)
                
            population = new_population

    # 保存最好的模型
    print("保存最佳模型到 best_model.npz")
    scores[0][1].save("best_model.npz")

    return scores[0][1] # 返回最好的大脑

class VisualGameWithAgent(MazeGame):
    def __init__(self, width, height, brain):
        super().__init__(width, height)
        # 替换玩家为 AI Agent
        # 注意：MazeGame.__init__ 已经创建了 self.player，我们需要替换它
        # 并且要移除旧的 patch
        self.player.patch.remove()
        self.player = GeneticAgent(0, 0, 'blue', self.maze.ax, height, width, brain)
        
        # 替换 AI 为 DummyAI (静态靶子)
        self.ai.patch.remove()
        self.ai = DummyAI(width-1, height-1, 'red', self.maze.ax, height, width)
        
        self.maze.title.set_text("AI vs Dummy Target (Learning Pathfinding)")
        
        # 移除键盘控制，改为自动更新
        # 我们需要重写 update 方法或者在 update 中加入 agent 的逻辑
        
    def on_key(self, event):
        # 禁用键盘控制移动，只保留重启/退出
        if self.game_over:
            super().on_key(event)

    def update(self):
        if not self.game_over and self.player.alive and self.ai.alive:
            # Agent 自动决策
            self.player.decide(self.maze.grid_graph, self.ai, self.ai.bullet)
            
        super().update()

if __name__ == "__main__":
    MODEL_FILE = "best_model.npz"
    TRAIN_MODE = True # 设置为 False 可以直接加载模型跳过训练
    
    # 检查是否存在模型文件
    if os.path.exists(MODEL_FILE):
        print(f"发现已保存的模型文件: {MODEL_FILE}")
        choice = input("是否加载模型并跳过训练? (y/n): ").lower()
        if choice == 'y':
            TRAIN_MODE = False
    
    best_brain = None
    
    if TRAIN_MODE:
        print("正在使用遗传算法训练 AI Agent...")
        best_brain = train()
    else:
        print("正在加载模型...")
        best_brain = NeuralNetwork(12, 24, 5)
        best_brain.load(MODEL_FILE)
    
    print("\n训练完成！正在展示最强个体的表现...")
    print("蓝色方块现在由神经网络控制。")
    
    while True:
        game = VisualGameWithAgent(10, 10, best_brain)
        game.run()
        if not game.restart_requested:
            break
