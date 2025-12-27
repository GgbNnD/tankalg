import numpy as np
import random
from collections import deque
import sys
import os
import logging
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import pickle

# 将当前目录添加到路径以确保导入正常工作
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maze_generator import MazeGenerator
import simple_nn as nn

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dqn_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info(f"Running on device: CPU (NumPy)")

# --- 无图形界面迷宫生成器 ---
class HeadlessMazeGenerator(MazeGenerator):
    def __init__(self, width, height):
        # 仅初始化逻辑相关属性，跳过可视化
        self.width = width
        self.height = height
        self.visited = [[False for _ in range(width)] for _ in range(height)]
        self.grid_graph = {}
        for y in range(height):
            for x in range(width):
                self.grid_graph[(x, y)] = []
        # 无图形
        self.lines = []

    def remove_wall_visual(self, x, y, direction):
        pass 

    def update_head(self, x, y):
        pass

    def generate(self, algo='dfs', block=False):
        # 重写以避免 UI 调用
        if algo == 'dfs':
            self._dfs(0, 0)
        elif algo == 'prim':
            self._prim(0, 0)

# --- DQN 模型 ---
class DQN(nn.Sequential):
    def __init__(self, input_channels, height, width, output_dim):
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.flatten_size = 128 * height * width
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        super().__init__(self.conv, self.fc)

# --- 经验回放缓冲区 ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

# --- 环境包装器 ---
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
        # 随机化算法以使 AI 对不同的迷宫结构具有鲁棒性
        algo = random.choice(['dfs', 'prim'])
        self.maze.generate(algo=algo)
        
        # 随机化起点和终点位置
        while True:
            ax, ay = random.randint(0, self.width-1), random.randint(0, self.height-1)
            gx, gy = random.randint(0, self.width-1), random.randint(0, self.height-1)
            # 确保最小距离以避免过于简单的回合和数据污染
            if abs(ax - gx) + abs(ay - gy) > min(self.width, self.height) / 2:
                self.agent_pos = (ax, ay)
                self.goal_pos = (gx, gy)
                break
        
        self.visited_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.visited_map[self.agent_pos[1], self.agent_pos[0]] = 1.0
        
        return self.get_state()

    def get_state(self):
        # 状态: (7, H, W)
        # 0: 上方墙壁
        # 1: 下方墙壁
        # 2: 左方墙壁
        # 3: 右方墙壁
        # 4: 智能体位置
        # 5: 目标位置
        # 6: 访问地图
        
        state = np.zeros((7, self.height, self.width), dtype=np.float32)
        
        # 填充墙壁通道
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self.maze.grid_graph.get((x, y), [])
                # 如果 (x, y-1) 不在邻居中，则上方墙壁为 1
                if (x, y - 1) not in neighbors: state[0, y, x] = 1.0
                if (x, y + 1) not in neighbors: state[1, y, x] = 1.0
                if (x - 1, y) not in neighbors: state[2, y, x] = 1.0
                if (x + 1, y) not in neighbors: state[3, y, x] = 1.0
        
        # 智能体位置
        ax, ay = self.agent_pos
        state[4, ay, ax] = 1.0
        
        # 目标位置
        gx, gy = self.goal_pos
        state[5, gy, gx] = 1.0
        
        # 访问地图
        state[6] = self.visited_map
        
        return state

    def step(self, action):
        x, y = self.agent_pos
        
        # 0: 上, 1: 下, 2: 左, 3: 右
        dx, dy = 0, 0
        if action == 0: dy = -1
        elif action == 1: dy = 1
        elif action == 2: dx = -1
        elif action == 3: dx = 1
        
        nx, ny = x + dx, y + dy
        
        neighbors = self.maze.grid_graph.get((x, y), [])
        
        reward = -0.1 # 增加步数惩罚以鼓励更短的路径
        done = False
        
        if (nx, ny) in neighbors:
            self.agent_pos = (nx, ny)
            
            # 重复访问惩罚
            if self.visited_map[ny, nx] > 0:
                reward = -0.5 # 重复访问的惩罚
            else:
                self.visited_map[ny, nx] = 1.0
                reward = 0.2 # 净负值以鼓励速度，但比重复访问要好
            
            if self.agent_pos == self.goal_pos:
                reward = 20.0 # 更强的目标奖励
                done = True
        else:
            reward = -1.0 # 更强的撞墙惩罚
            
        return self.get_state(), reward, done

def save_checkpoint(state, filename="checkpoint.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        logging.info(f"Loading checkpoint '{filename}'")
        try:
            with open(filename, 'rb') as f:
                checkpoint = pickle.load(f)
            model.set_params(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer']) # 简单的 Adam 尚未实现
            start_episode = checkpoint['episode'] + 1
            epsilon = checkpoint['epsilon']
            logging.info(f"Loaded checkpoint '{filename}' (episode {checkpoint['episode']})")
            return start_episode, epsilon
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
            logging.warning("Starting from scratch.")
            return 0, 1.0
    else:
        logging.info(f"No checkpoint found at '{filename}'")
        return 0, 1.0

# --- 训练循环 ---
def train(resume=False):
    # 超参数
    WIDTH, HEIGHT = 10, 10
    EPISODES = 10000 
    BATCH_SIZE = 32 # 增加批量大小以获得更平滑的梯度
    GAMMA = 0.9 # 略微增加以看得更远
    EPSILON_START = 1.0
    EPSILON_END = 0.05 # 允许在最后更多利用经验
    EPSILON_DECAY_EPISODES = 4000 # 线性衰减持续时间
    LR = 0.0001 
    TARGET_UPDATE = 200 # 降低目标更新频率以提高稳定性
    MEMORY_SIZE = 50000 # 更大的内存以减少相关性

    env = MazeEnv(WIDTH, HEIGHT)
    
    input_channels = 7 
    output_dim = 4 
    
    policy_net = DQN(input_channels, HEIGHT, WIDTH, output_dim)
    target_net = DQN(input_channels, HEIGHT, WIDTH, output_dim)
    target_net.set_params(policy_net.get_params())
    
    optimizer = nn.Adam(policy_net.get_params(), lr=LR)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    
    epsilon = EPSILON_START
    start_episode = 0

    if resume:
        start_episode, epsilon = load_checkpoint("checkpoint.pkl", policy_net, optimizer)
        target_net.set_params(policy_net.get_params())
    
    logging.info(f"Starting training from episode {start_episode}...")
    
    rewards_history = []

    for episode in range(start_episode, EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 300: # 增加最大步数
            # Epsilon-贪婪动作选择
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                # 前向传播
                state_batch = state[np.newaxis, :] # (1, C, H, W)
                q_values = policy_net.forward(state_batch)
                action = np.argmax(q_values)
            
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # 训练
            if len(replay_buffer) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                
                # 前向传播
                q_values = policy_net.forward(states)
                
                # 获取所采取动作的 Q 值
                # q_values: (B, 4), actions: (B,)
                q_value = q_values[np.arange(BATCH_SIZE), actions]
                
                # 目标 Q 值
                next_q_values = target_net.forward(next_states)
                next_q_value = np.max(next_q_values, axis=1)
                expected_q_value = rewards + GAMMA * next_q_value * (1 - dones)
                
                # 损失
                loss = loss_fn.forward(q_value, expected_q_value)
                
                # 反向传播
                grad_loss = loss_fn.backward() # (B,)
                
                # 我们需要反向传播到特定的动作输出
                grad_q_values = np.zeros_like(q_values)
                grad_q_values[np.arange(BATCH_SIZE), actions] = grad_loss
                
                policy_net.backward(grad_q_values)
                
                # 优化器步骤
                optimizer.step(policy_net.get_grads())
        
        # 更新 epsilon (线性衰减)
        if epsilon > EPSILON_END:
            epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY_EPISODES
            epsilon = max(EPSILON_END, epsilon)
        
        rewards_history.append(total_reward)

        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            target_net.set_params(policy_net.get_params())
            
        if episode % 50 == 0:
            logging.info(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}, Steps: {steps}")
            # 保存检查点
            save_checkpoint({
                'episode': episode,
                'state_dict': policy_net.get_params(),
                'epsilon': epsilon
            }, filename="checkpoint.pkl")

    # 保存模型
    with open("maze_dqn_model.pkl", 'wb') as f:
        pickle.dump(policy_net.get_params(), f)
    logging.info("Training complete. Model saved to maze_dqn_model.pkl")

    # 绘制奖励
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
