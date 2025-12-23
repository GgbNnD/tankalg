import matplotlib.pyplot as plt
import random
import sys

# 增加递归深度限制，以防迷宫过大导致递归错误
sys.setrecursionlimit(10000)

class MazeGenerator:
    def __init__(self, width, height, headless=False):
        self.width = width
        self.height = height
        self.headless = headless
        # 记录访问过的格子
        self.visited = [[False for _ in range(width)] for _ in range(height)]
        
        # 记录墙壁状态
        # 每个格子有四面墙：上(0), 右(1), 下(2), 左(3)
        # 使用集合存储存在的墙壁，格式为 ((x1, y1), (x2, y2))
        self.walls = set()
        self._init_walls()
        
        # 迷宫图结构，用于寻路算法 (Adjacency List)
        self.grid_graph = {}
        for y in range(height):
            for x in range(width):
                self.grid_graph[(x, y)] = []

        # 用于可视化的设置
        if not self.headless:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            self.title = self.ax.set_title("Maze Generation (DFS Algorithm)", fontsize=15)
            
            # 预先绘制所有墙壁（黑色）
            self.lines = []
            for y in range(self.height):
                for x in range(self.width):
                    # 只画右边和下边的墙，左边和上边由相邻格子负责（除了边界）
                    if x == self.width - 1: # 最右侧边界
                        self.ax.plot([x+1, x+1], [self.height-y, self.height-(y+1)], 'k-', lw=2)
                    if y == self.height - 1: # 最下侧边界
                        self.ax.plot([x, x+1], [self.height-(y+1), self.height-(y+1)], 'k-', lw=2)
                    
                    # 内部墙壁（初始全画）
                    # 右墙
                    line_v, = self.ax.plot([x+1, x+1], [self.height-y, self.height-(y+1)], 'k-', lw=2)
                    self.lines.append(((x, y, 'right'), line_v))
                    # 下墙
                    line_h, = self.ax.plot([x, x+1], [self.height-(y+1), self.height-(y+1)], 'k-', lw=2)
                    self.lines.append(((x, y, 'down'), line_h))
                    
                    # 左边界和上边界
                    if x == 0:
                        self.ax.plot([x, x], [self.height-y, self.height-(y+1)], 'k-', lw=2)
                    if y == 0:
                        self.ax.plot([x, x+1], [self.height-y, self.height-y], 'k-', lw=2)

            # 标记起点和终点
            # 起点 (0, 0) -> 对应绘图坐标 (0.5, height-0.5)
            self.start_patch = plt.Circle((0.5, self.height - 0.5), 0.3, color='lime', zorder=10)
            self.ax.add_patch(self.start_patch)
            self.ax.text(0.5, self.height - 0.5, 'S', ha='center', va='center', color='black', fontweight='bold')

            # 终点 (width-1, height-1)
            self.end_patch = plt.Circle((self.width - 0.5, 0.5), 0.3, color='red', zorder=10)
            self.ax.add_patch(self.end_patch)
            self.ax.text(self.width - 0.5, 0.5, 'E', ha='center', va='center', color='white', fontweight='bold')
            
            # 当前生成头部的标记
            self.head_patch = plt.Circle((0.5, self.height - 0.5), 0.2, color='blue', zorder=5)
            self.ax.add_patch(self.head_patch)

            plt.tight_layout()
            plt.ion() # 开启交互模式
            plt.show()

    def _init_walls(self):
        # 逻辑上初始化所有墙壁
        pass 

    def connect_cells(self, x1, y1, x2, y2, direction):
        """连接两个格子，更新图结构并移除视觉上的墙壁"""
        self.grid_graph[(x1, y1)].append((x2, y2))
        self.grid_graph[(x2, y2)].append((x1, y1))
        self.remove_wall_visual(x1, y1, direction)

    def remove_wall_visual(self, x, y, direction):
        if self.headless:
            return
        # 在图中移除墙壁（用白色覆盖或者隐藏线条）
        # 这里我们简单地查找对应的线条对象并将其设为不可见
        target_tag = None
        if direction == 'right':
            target_tag = (x, y, 'right')
        elif direction == 'down':
            target_tag = (x, y, 'down')
        elif direction == 'left':
            target_tag = (x-1, y, 'right')
        elif direction == 'up':
            target_tag = (x, y-1, 'down')
            
        for tag, line in self.lines:
            if tag == target_tag:
                line.set_visible(False)
                break

    def update_head(self, x, y):
        if self.headless:
            return
        self.head_patch.center = (x + 0.5, self.height - y - 0.5)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def generate(self, algo='dfs', block=True):
        if not self.headless:
            self.title.set_text(f"Maze Generation ({algo.upper()} Algorithm)")
        # 从起点 (0,0) 开始
        if algo == 'dfs':
            self._dfs(0, 0)
        elif algo == 'prim':
            self._prim(0, 0)
        
        # 生成结束
        if not self.headless:
            self.head_patch.set_visible(False)
            self.title.set_text("Maze Generation Complete!")
            self.fig.canvas.draw()
            
            if block:
                plt.ioff()
                plt.show()

    def _prim(self, start_x, start_y):
        self.visited[start_y][start_x] = True
        # frontier 存储待访问的邻居节点 (x, y)
        frontier = []
        
        directions = [(0, -1, 'up'), (0, 1, 'down'), (-1, 0, 'left'), (1, 0, 'right')]
        
        # 初始化 frontier
        for dx, dy, _ in directions:
            nx, ny = start_x + dx, start_y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                frontier.append((nx, ny))
        
        self.update_head(start_x, start_y)

        while frontier:
            # 随机选择一个 frontier 节点
            idx = random.randint(0, len(frontier) - 1)
            cx, cy = frontier.pop(idx)
            
            if self.visited[cy][cx]:
                continue
            
            self.visited[cy][cx] = True
            self.update_head(cx, cy)
            
            # 寻找该节点与已访问区域的连接
            potential_neighbors = []
            for dx, dy, direction in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.visited[ny][nx]:
                    potential_neighbors.append((dx, dy, direction))
            
            if potential_neighbors:
                # 随机选择一个已访问的邻居进行连接
                dx, dy, direction = random.choice(potential_neighbors)
                # remove_wall_visual 需要的是从当前格子出发去打通墙壁的方向
                # self.remove_wall_visual(cx, cy, direction)
                nx, ny = cx + dx, cy + dy
                self.connect_cells(cx, cy, nx, ny, direction)
            
            # 将新节点的未访问邻居加入 frontier
            for dx, dy, _ in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and not self.visited[ny][nx]:
                    frontier.append((nx, ny))

    def _dfs(self, x, y):
        self.visited[y][x] = True
        self.update_head(x, y)
        
        # 定义四个方向: (dx, dy, direction_name)
        directions = [
            (0, -1, 'up'), 
            (0, 1, 'down'), 
            (-1, 0, 'left'), 
            (1, 0, 'right')
        ]
        random.shuffle(directions)
        
        for dx, dy, direction in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < self.width and 0 <= ny < self.height and not self.visited[ny][nx]:
                # 移除墙壁
                # self.remove_wall_visual(x, y, direction)
                self.connect_cells(x, y, nx, ny, direction)
                # 递归访问
                self._dfs(nx, ny)
                # 回溯时更新头部位置，展示回溯过程
                self.update_head(x, y)

if __name__ == "__main__":
    print("欢迎使用迷宫生成器！")
    try:
        w = int(input("请输入迷宫宽度 (建议 10-30): "))
        h = int(input("请输入迷宫高度 (建议 10-30): "))
    except ValueError:
        print("输入无效，使用默认值 15x15")
        w, h = 15, 15

    print(f"正在生成 {w}x{h} 的迷宫...")
    
    print("请选择生成算法:")
    print("1. 递归回溯 (DFS)")
    print("2. Prim算法 (Prim)")
    algo_choice = input("请输入选项 (1 或 2): ")
    
    algo = 'dfs'
    if algo_choice == '2':
        algo = 'prim'

    print(f"使用的算法: {algo.upper()}")
    
    maze = MazeGenerator(w, h)
    maze.generate(algo)
