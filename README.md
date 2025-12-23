# Tank Trouble - Python Edition (Maze Battle)

这是一个基于 Python 和 Matplotlib 的迷宫坦克对战游戏。包含迷宫生成、A* 寻路演示以及核心的坦克对战玩法。

**特别说明**：本项目中的 AI 采用了全知全能的算法设计（基于 A* 寻路和实时射线检测），它知道地图结构、你的位置以及子弹轨迹。准备好迎接挑战了吗？

## 项目结构

- `maze_game.py`: **主游戏文件**。玩家控制蓝色坦克，与红色 AI 坦克在迷宫中对战。
- `maze_generator.py`: 迷宫生成器演示。支持 DFS (递归回溯) 和 Prim 算法生成迷宫。
- `maze_solver.py`: 迷宫求解器演示。使用 A* 算法寻找迷宫路径。

## 算法详解 (Smart AI)

本项目的 AI 放弃了传统的强化学习训练，转而采用基于规则的“全知全能”算法，以提供更具挑战性的游戏体验。

### 1. 核心逻辑优先级
AI 每帧会按照以下优先级进行决策：
1.  **生存优先 (Survival)**: 检测是否有子弹正在飞向自己。
2.  **击杀优先 (Kill)**: 检测玩家是否暴露在枪口下。
3.  **追击优先 (Chase)**: 如果安全且无法射击，则向玩家移动。

### 2. 关键技术实现

#### A* 寻路算法 (A-Star Pathfinding)
- **用途**: 用于计算从 AI 当前位置到玩家位置的最短路径。
- **实现**: 使用曼哈顿距离作为启发式函数 (Heuristic)，结合优先队列 (Priority Queue) 高效搜索迷宫图结构。
- **效果**: AI 能够绕过死胡同，以最短路线逼近玩家。

#### 射线检测 (Ray Casting / Line of Sight)
- **用途**: 判断两点之间是否存在墙壁阻挡。
- **实现**: 遍历两点连线上的所有网格坐标，检查 `grid_graph` 中是否存在连通性。
- **效果**: 
    - **攻击判定**: 只有当玩家在同一直线且无墙壁遮挡时，AI 才会开火，杜绝了“描边枪法”。
    - **躲避判定**: AI 能预判子弹轨迹，只有当子弹真的会打到自己（且中间无墙）时才进行躲避。

#### 动态躲避策略 (Dynamic Evasion)
- **用途**: 在被子弹锁定时寻找安全位置。
- **实现**: 遍历当前位置的所有邻居节点，筛选出不在子弹轨迹上的节点作为移动目标。
- **效果**: AI 能够像“黑客帝国”一样侧身躲避来袭的子弹。

## 环境要求

- Python 3.x

## 安装与运行全流程

建议使用 Python 虚拟环境 (`.venv`) 来运行本项目，以保持环境整洁。

### 1. 创建虚拟环境

在项目根目录下打开终端 (Terminal / PowerShell / CMD)，运行以下命令创建名为 `.venv` 的虚拟环境：

```bash
python -m venv .venv
```

### 2. 激活虚拟环境

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```
*注意：如果遇到权限错误，请先运行 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`。*

**Windows (CMD):**
```cmd
.\.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

激活成功后，命令行提示符前会出现 `(.venv)` 字样。

### 3. 安装依赖

在激活的虚拟环境中，安装项目所需的依赖库 (主要是 `matplotlib`)：

```bash
pip install -r requirements.txt
```

### 4. 运行游戏

#### 启动坦克对战 (主游戏)
```bash
python maze_game.py
```
- **控制方式**:
  - **方向键 (↑ ↓ ← →)**: 移动
  - **空格键 (Space)**: 射击
- **目标**: 躲避 AI 的子弹并击毁 AI 坦克。
- **AI 特性**:
  - **自动寻路**: 使用 A* 算法实时追踪玩家。
  - **自动射击**: 当玩家暴露在枪口下且无遮挡时，AI 会瞬间开火。
  - **自动躲避**: AI 会尝试躲避来袭的子弹。

#### 运行迷宫生成演示
```bash
python maze_generator.py
```
- 按照提示输入迷宫宽高和选择生成算法。

#### 运行迷宫求解演示
```bash
python maze_solver.py
```
- 演示 A* 算法如何找到从起点到终点的路径。

## 退出虚拟环境

游戏结束后，如果想退出虚拟环境，可以运行：
```bash
deactivate
```
