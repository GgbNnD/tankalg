import torch
import matplotlib.pyplot as plt
# plt.rcParams['toolbar'] = 'None'
import numpy as np
import sys
import os
import time
import random

# 将当前目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maze_generator import MazeGenerator
from train_torch import DQN

def get_state(agent_pos, goal_pos, width, height, grid_graph, visited_map):
    # 状态: (7, H, W)
    # 0: 上方墙壁
    # 1: 下方墙壁
    # 2: 左方墙壁
    # 3: 右方墙壁
    # 4: 智能体位置
    # 5: 目标位置
    # 6: 访问地图
    
    state = np.zeros((7, height, width), dtype=np.float32)
    
    # 填充墙壁通道
    for y in range(height):
        for x in range(width):
            neighbors = grid_graph.get((x, y), [])
            # 如果 (x, y-1) 不在邻居中，则上方墙壁为 1
            if (x, y - 1) not in neighbors: state[0, y, x] = 1.0
            if (x, y + 1) not in neighbors: state[1, y, x] = 1.0
            if (x - 1, y) not in neighbors: state[2, y, x] = 1.0
            if (x + 1, y) not in neighbors: state[3, y, x] = 1.0
    
    # 智能体位置
    ax, ay = agent_pos
    state[4, ay, ax] = 1.0
    
    # 目标位置
    gx, gy = goal_pos
    state[5, gy, gx] = 1.0
    
    # 访问地图
    state[6] = visited_map
    
    return state

def test_agent():
    # 参数 (必须与训练匹配)
    WIDTH, HEIGHT = 10, 10
    MODEL_PATH = "maze_dqn_model.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 未找到模型文件 '{MODEL_PATH}'。请先训练模型。")
        return

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 注意: 输入通道=7, 高度=10, 宽度=10, 输出=4
    model = DQN(7, HEIGHT, WIDTH, 4).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print(f"加载模型时出错: {e}")
        print("模型架构似乎已更改。请使用 'python train.py' 重新训练模型。")
        return

    model.eval()
    print("模型加载成功。")

    # 生成迷宫
    print("正在生成迷宫...")
    # 使用 block=False 以便我们可以运行自己的循环
    maze = MazeGenerator(WIDTH, HEIGHT)
    maze.generate(algo='dfs', block=False)
    maze.title.set_text("DQN 智能体测试 (CNN)")

    # 随机化测试的起点和终点
    while True:
        ax, ay = random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1)
        gx, gy = random.randint(0, WIDTH-1), random.randint(0, HEIGHT-1)
        if (ax, ay) != (gx, gy):
            agent_pos = (ax, ay)
            goal_pos = (gx, gy)
            break
            
    print(f"测试开始: {agent_pos}, 目标: {goal_pos}")

    visited_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    visited_map[agent_pos[1], agent_pos[0]] = 1.0
    
    # 绘制目标 (红色 'E' 是默认值，但让我们突出显示我们的随机目标)
    # 如果需要，清除默认的开始/结束文本，但现在只需添加新标记
    goal_patch = plt.Circle(
        (goal_pos[0] + 0.5, maze.height - goal_pos[1] - 0.5), 
        0.3, color='red', zorder=19
    )
    maze.ax.add_patch(goal_patch)

    # 绘制智能体 (绿色圆圈)
    agent_patch = plt.Circle(
        (agent_pos[0] + 0.5, maze.height - agent_pos[1] - 0.5), 
        0.3, color='green', zorder=20
    )
    maze.ax.add_patch(agent_patch)
    
    print("开始导航...")
    steps = 0
    max_steps = 100
    
    while agent_pos != goal_pos and steps < max_steps:
        # 获取状态
        state = get_state(agent_pos, goal_pos, WIDTH, HEIGHT, maze.grid_graph, visited_map)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # 预测动作
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item()
            
        # 执行动作
        x, y = agent_pos
        dx, dy = 0, 0
        if action == 0: dy = -1    # 上
        elif action == 1: dy = 1   # 下
        elif action == 2: dx = -1  # 左
        elif action == 3: dx = 1   # 右
        
        nx, ny = x + dx, y + dy
        
        # 检查有效性
        neighbors = maze.grid_graph.get((x, y), [])
        if (nx, ny) in neighbors:
            agent_pos = (nx, ny)
            visited_map[ny, nx] = 1.0
            print(f"步骤 {steps}: 移动到 {agent_pos}")
        else:
            print(f"步骤 {steps}: 在 {agent_pos} 撞墙，试图向 {['上','下','左','右'][action]} 移动")
            
        # 更新可视化
        agent_patch.center = (agent_pos[0] + 0.5, maze.height - agent_pos[1] - 0.5)
        maze.fig.canvas.draw()
        maze.fig.canvas.flush_events()
        
        steps += 1
        time.sleep(0.2) # 减慢速度以查看移动

    if agent_pos == goal_pos:
        maze.title.set_text("成功！到达目标！")
        print("成功！到达目标！")
    else:
        maze.title.set_text("失败：达到最大步数")
        print("失败：达到最大步数")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_agent()
