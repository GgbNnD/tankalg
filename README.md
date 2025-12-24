# TANKTROUBLE - Python Edition

这是一个基于 Python 和 Pygame 的 Tank Trouble 复刻版。
本项目包含单人练习（对抗 AI）和本地多人对战功能。

## 特性

- **智能 AI 对手**：玩家 2 默认由智能 AI 控制，能够自动寻路、躲避子弹和攻击。
- **本地多人对战**：支持最多 3 个实体（包括 AI）在同一键盘上对战。
- **物理引擎**：基于 Pygame 的 AABB 碰撞检测和反弹逻辑。
- **固定地图**：目前使用包含两面竖直墙壁的固定竞技场地图。

## 环境要求

- Python 3.10+
- Pygame
- Numpy

## 安装与运行

建议使用虚拟环境运行本项目。

### Windows (PowerShell)

1. **创建并激活虚拟环境**
   ```powershell
   python -m venv .venv
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\.venv\Scripts\Activate.ps1
   ```

2. **安装依赖**
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. **运行游戏**
   ```powershell
   python main.py
   ```

## 游戏操作

### 主菜单

- **Q**: 启用/禁用 **玩家 1** (红色)
- **U**: 启用/禁用 **玩家 2** (绿色 - **AI 控制**)
- **DELETE**: 启用/禁用 **玩家 3** (蓝色)
- **SPACE**: 开始游戏 (至少需要启用 2 个玩家)

### 游戏内控制

| 玩家 | 移动 | 射击 | 说明 |
| --- | --- | --- | --- |
| **Player 1** (红) | `W` `A` `S` `D` | `Q` | 人类玩家 |
| **Player 2** (绿) | `I` `J` `K` `L` | `U` | **由电脑 (AI) 控制** (手动按键无效) |
| **Player 3** (蓝) | `↑` `↓` `←` `→` | `DELETE` | 人类玩家 |

- **ESC**: 退出到主菜单
- **P**: 暂停游戏

## 游戏模式说明

- **人机对战**: 在菜单中启用 **Player 1** 和 **Player 2**。
- **双人对战**: 在菜单中启用 **Player 1** 和 **Player 3**。
- **混战模式**: 启用所有三个玩家。

## 项目结构

- `main.py`: 游戏主入口。
- `debug_run.py`: 带调试输出的入口。
- `source/`: 源代码目录。
  - `ai/`: AI 逻辑实现 (`smart_ai.py`)。
  - `sites/`: 游戏场景 (菜单、竞技场等)。
  - `parts/`: 游戏实体 (玩家、子弹、墙壁等)。
    
