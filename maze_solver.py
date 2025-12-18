import matplotlib.pyplot as plt
import heapq
import sys
from maze_generator import MazeGenerator

class MazeSolver:
    def __init__(self, maze):
        self.maze = maze
        self.start = (0, 0)
        self.end = (maze.width - 1, maze.height - 1)

    def heuristic(self, a, b):
        # 曼哈顿距离
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self):
        print("开始使用 A* 算法解迷宫...")
        self.maze.title.set_text("Solving Maze with A* Algorithm...")
        
        start = self.start
        goal = self.end
        
        # 优先队列 (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        
        # g_score: 从起点到当前节点的实际代价
        g_score = {node: float('inf') for node in self.maze.grid_graph}
        g_score[start] = 0
        
        # f_score: g_score + heuristic
        f_score = {node: float('inf') for node in self.maze.grid_graph}
        f_score[start] = self.heuristic(start, goal)
        
        # 记录已绘制的节点，避免重复绘制
        visited_visuals = set()

        while open_set:
            # 获取 f_score 最小的节点
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                self.reconstruct_path(came_from, current)
                return

            # 可视化：标记当前探索的节点（浅蓝色）
            if current != start and current != goal and current not in visited_visuals:
                # 注意：绘图坐标系 y 轴是反的或者需要转换
                # MazeGenerator 中：y=0 在上，绘图 y = height - y
                # patch center: (x + 0.5, height - y - 0.5)
                patch = plt.Circle(
                    (current[0] + 0.5, self.maze.height - current[1] - 0.5), 
                    0.2, color='lightblue', alpha=0.6, zorder=4
                )
                self.maze.ax.add_patch(patch)
                visited_visuals.add(current)
                
                # 为了不让动画太慢，每隔一定步数刷新一次，或者每次都刷新
                # 这里每次刷新会比较慢，但能看清过程
                self.maze.fig.canvas.draw()
                self.maze.fig.canvas.flush_events()

            for neighbor in self.maze.grid_graph[current]:
                # 假设每一步代价为 1
                tentative_g_score = g_score[current] + 1
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print("未找到路径！")

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(self.start)
        path.reverse()
        
        print(f"找到路径！长度: {len(path)}")
        
        # 绘制最终路径
        x_coords = [p[0] + 0.5 for p in path]
        y_coords = [self.maze.height - p[1] - 0.5 for p in path]
        
        self.maze.ax.plot(x_coords, y_coords, 'r-', linewidth=3, alpha=0.8, zorder=6)
        self.maze.title.set_text("Maze Solved with A*!")
        self.maze.fig.canvas.draw()
        
        # 保持窗口打开
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    print("欢迎使用 A* 迷宫求解器！")
    try:
        w = int(input("请输入迷宫宽度 (建议 10-30): "))
        h = int(input("请输入迷宫高度 (建议 10-30): "))
    except ValueError:
        print("输入无效，使用默认值 15x15")
        w, h = 15, 15

    print("请选择生成算法:")
    print("1. 递归回溯 (DFS)")
    print("2. Prim算法 (Prim)")
    algo_choice = input("请输入选项 (1 或 2): ")
    
    algo = 'dfs'
    if algo_choice == '2':
        algo = 'prim'

    # 1. 生成迷宫 (block=False 以便继续运行求解)
    maze = MazeGenerator(w, h)
    maze.generate(algo, block=False)
    
    # 2. 求解迷宫
    solver = MazeSolver(maze)
    # 稍微暂停一下，让用户看清迷宫生成完毕的状态
    plt.pause(1)
    solver.solve()
