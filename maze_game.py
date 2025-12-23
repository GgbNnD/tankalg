import matplotlib.pyplot as plt
import matplotlib.patches as patches
from maze_generator import MazeGenerator
import random
import heapq
import sys

class Bullet:
    def __init__(self, x, y, dx, dy, color, ax, maze_height):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.active = True
        self.maze_height = maze_height
        # 初始绘制
        self.patch = patches.Circle(
            (self.x + 0.5, self.maze_height - self.y - 0.5), 
            0.15, color=color, zorder=20
        )
        ax.add_patch(self.patch)

    def move(self, grid_graph, opponents):
        if not self.active:
            return

        # 简单的碰撞检测：检查当前位置是否可以移动到下一个位置
        # 子弹速度设定为每帧移动1格，为了避免穿墙，我们先检查连通性
        next_x = self.x + self.dx
        next_y = self.y + self.dy
        
        # 检查墙壁碰撞
        if (next_x, next_y) not in grid_graph.get((self.x, self.y), []):
            self.active = False
            self.patch.set_visible(False)
            return

        # 更新位置
        self.x = next_x
        self.y = next_y
        self.patch.center = (self.x + 0.5, self.maze_height - self.y - 0.5)

        # 检查击中对手
        for op in opponents:
            if op.alive and op.x == self.x and op.y == self.y:
                op.die()
                self.active = False
                self.patch.set_visible(False)

class Entity:
    def __init__(self, x, y, color, ax, maze_height, maze_width):
        self.x = x
        self.y = y
        self.color = color
        self.ax = ax
        self.maze_height = maze_height
        self.maze_width = maze_width
        self.alive = True
        self.bullet = None
        self.facing = (1, 0) # 默认朝右
        
        # 绘制正方形个体
        # 逻辑坐标 (x, y) -> 绘图坐标 (x, height - y - 1) (左下角)
        self.patch = patches.Rectangle(
            (self.x + 0.2, self.maze_height - self.y - 0.8), 
            0.6, 0.6, color=color, zorder=15
        )
        ax.add_patch(self.patch)

    def move(self, dx, dy, grid_graph, opponents=None):
        if not self.alive:
            return
        
        self.facing = (dx, dy)
        nx, ny = self.x + dx, self.y + dy
        
        # 检查连通性
        if (nx, ny) in grid_graph.get((self.x, self.y), []):
            # 检查是否被占据
            if opponents:
                for op in opponents:
                    if op.alive and op.x == nx and op.y == ny:
                        return

            self.x, self.y = nx, ny
            self.patch.set_xy((self.x + 0.2, self.maze_height - self.y - 0.8))

    def shoot(self):
        if not self.alive:
            return
        # 只有当场上没有自己的子弹时才能发射
        if self.bullet is None or not self.bullet.active:
            # 移除旧的子弹图形（如果有）
            if self.bullet:
                self.bullet.patch.remove()
            
            self.bullet = Bullet(self.x, self.y, self.facing[0], self.facing[1], self.color, self.ax, self.maze_height)

    def die(self):
        self.alive = False
        self.patch.set_color('gray')
        self.patch.set_alpha(0.5)

class AI(Entity):
    def update(self, grid_graph, player, player_bullet):
        if not self.alive:
            return

        # 1. 躲避子弹逻辑
        if player_bullet and player_bullet.active:
            if self.is_in_danger(player_bullet, grid_graph):
                # 尝试移动到安全的位置
                safe_move = self.find_safe_move(grid_graph, player_bullet)
                if safe_move:
                    self.move(safe_move[0], safe_move[1], grid_graph, [player])
                    return # 躲避优先，本回合不进行其他移动

        # 2. 攻击逻辑
        if self.can_shoot_player(grid_graph, player):
            # 调整朝向并射击
            dx = 1 if player.x > self.x else -1 if player.x < self.x else 0
            dy = 1 if player.y > self.y else -1 if player.y < self.y else 0
            if dx != 0: dy = 0 # 确保只在一个方向上
            
            self.facing = (dx, dy)
            self.shoot()
            # 射击时不移动
            return

        # 3. 寻路逻辑 (Tactical Dijkstra)
        # 使用 Dijkstra 算法寻找一条既短又安全的路径，避开玩家的枪线
        path = self.find_tactical_path(grid_graph, player)
        if path and len(path) > 1:
            next_node = path[1]
            dx = next_node[0] - self.x
            dy = next_node[1] - self.y
            self.move(dx, dy, grid_graph, [player])

    def is_in_danger(self, bullet, grid_graph):
        # 简单预测：如果子弹在当前行或列，并且朝向我，且中间无阻挡
        if bullet.dx != 0 and bullet.y == self.y:
            # 水平方向
            if (bullet.dx > 0 and self.x > bullet.x) or (bullet.dx < 0 and self.x < bullet.x):
                return self.check_line_of_sight(grid_graph, (bullet.x, bullet.y), (self.x, self.y))
        if bullet.dy != 0 and bullet.x == self.x:
            # 垂直方向
            if (bullet.dy > 0 and self.y > bullet.y) or (bullet.dy < 0 and self.y < bullet.y):
                return self.check_line_of_sight(grid_graph, (bullet.x, bullet.y), (self.x, self.y))
        return False

    def find_safe_move(self, grid_graph, bullet):
        neighbors = grid_graph.get((self.x, self.y), [])
        random.shuffle(neighbors)
        
        # 寻找一个不在子弹轨迹上的邻居
        for nx, ny in neighbors:
            # 简单的判断：如果移动后不在子弹的行/列（假设子弹直线运动）
            # 或者即使在同行列，但不在子弹前方
            
            # 预测子弹下一步位置
            bullet_next_x = bullet.x + bullet.dx
            bullet_next_y = bullet.y + bullet.dy
            
            if (nx, ny) == (bullet_next_x, bullet_next_y):
                continue # 不要撞上子弹
            
            # 检查是否脱离了子弹的射击线
            if bullet.dx != 0: # 子弹水平运动
                if ny != bullet.y: # 移动到了不同行 -> 安全
                    return (nx - self.x, ny - self.y)
            elif bullet.dy != 0: # 子弹垂直运动
                if nx != bullet.x: # 移动到了不同列 -> 安全
                    return (nx - self.x, ny - self.y)
                    
        return None

    def can_shoot_player(self, grid_graph, player):
        # 检查是否在同一直线且无障碍
        if self.x == player.x or self.y == player.y:
            return self.check_line_of_sight(grid_graph, (self.x, self.y), (player.x, player.y))
        return False

    def check_line_of_sight(self, grid_graph, start, end):
        # 检查两点之间是否有墙
        x1, y1 = start
        x2, y2 = end
        
        if x1 == x2: # 垂直
            step = 1 if y2 > y1 else -1
            for y in range(y1, y2, step):
                if (x1, y + step) not in grid_graph.get((x1, y), []):
                    return False
        elif y1 == y2: # 水平
            step = 1 if x2 > x1 else -1
            for x in range(x1, x2, step):
                if (x + step, y1) not in grid_graph.get((x, y1), []):
                    return False
        else:
            return False # 不在同一直线
        return True

    def get_danger_cost(self, x, y, player, grid_graph):
        cost = 0
        # 1. 玩家射击线危险 (Pre-fire prevention)
        px, py = player.x, player.y
        pdx, pdy = player.facing
        
        # 检查是否在玩家的枪口朝向上
        is_in_line_of_fire = False
        if pdx != 0 and y == py: # 水平
            if (pdx > 0 and x > px) or (pdx < 0 and x < px):
                is_in_line_of_fire = True
        elif pdy != 0 and x == px: # 垂直
            if (pdy > 0 and y > py) or (pdy < 0 and y < py):
                is_in_line_of_fire = True
                
        if is_in_line_of_fire:
            # 检查是否有墙阻挡
            if self.check_line_of_sight(grid_graph, (px, py), (x, y)):
                cost += 20 # 极高的代价，尽量避开枪线

        return cost

    def find_tactical_path(self, grid_graph, player):
        start = (self.x, self.y)
        goal = (player.x, player.y)
        
        open_set = []
        # Dijkstra: (cost, current_node)
        heapq.heappush(open_set, (0, start))
        came_from = {}
        cost_so_far = {}
        cost_so_far[start] = 0
        
        while open_set:
            current_cost, current = heapq.heappop(open_set)
            
            if current == goal:
                break
            
            for neighbor in grid_graph.get(current, []):
                # 基础移动代价 = 1
                # 危险代价 = get_danger_cost
                danger = self.get_danger_cost(neighbor[0], neighbor[1], player, grid_graph)
                new_cost = cost_so_far[current] + 1 + danger
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current

        # 重建路径
        if goal not in came_from:
            return None
            
        path = []
        curr = goal
        while curr != start:
            path.append(curr)
            if curr not in came_from:
                return None
            curr = came_from[curr]
        path.append(start)
        path.reverse()
        return path

class MazeGame:
    def __init__(self, width, height):
        self.maze = MazeGenerator(width, height)
        # 生成迷宫但不阻塞
        self.maze.generate(algo='prim', block=False)
        self.maze.title.set_text("Maze Battle! You: Blue vs Smart AI: Red")
        
        self.player = Entity(0, 0, 'blue', self.maze.ax, height, width)
        self.ai = AI(width-1, height-1, 'red', self.maze.ax, height, width)
        
        self.maze.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.game_over = False
        self.restart_requested = False

        # 游戏循环定时器
        self.timer = self.maze.fig.canvas.new_timer(interval=200) # 200ms per tick
        self.timer.add_callback(self.update)
        self.timer.start()
        
    def run(self):
        # 关闭交互模式，确保 plt.show() 阻塞程序运行，防止窗口直接关闭
        plt.ioff()
        plt.show()

    def on_key(self, event):
        if self.game_over:
            if event.key == 'r':
                self.restart_requested = True
                plt.close(self.maze.fig)
            elif event.key == 'q':
                self.restart_requested = False
                plt.close(self.maze.fig)
            return

        if not self.player.alive:
            return
            
        if event.key == 'up':
            self.player.move(0, -1, self.maze.grid_graph, [self.ai])
        elif event.key == 'down':
            self.player.move(0, 1, self.maze.grid_graph, [self.ai])
        elif event.key == 'left':
            self.player.move(-1, 0, self.maze.grid_graph, [self.ai])
        elif event.key == 'right':
            self.player.move(1, 0, self.maze.grid_graph, [self.ai])
        elif event.key == ' ':
            self.player.shoot()
            
        self.maze.fig.canvas.draw()

    def update(self):
        if self.game_over:
            return

        if not self.player.alive or not self.ai.alive:
            self.game_over = True
            self.timer.stop()
            if not self.player.alive:
                self.maze.title.set_text("Game Over! You Died. Press 'r' to restart, 'q' to quit.")
            else:
                self.maze.title.set_text("Victory! You Killed the AI. Press 'r' to restart, 'q' to quit.")
            self.maze.fig.canvas.draw()
            return

        # 1. 更新子弹 (子弹速度快，可以更新两次)
        for _ in range(2):
            if self.player.bullet:
                self.player.bullet.move(self.maze.grid_graph, [self.ai])
            if self.ai.bullet:
                self.ai.bullet.move(self.maze.grid_graph, [self.player])

        # 2. AI 思考与行动
        self.ai.update(self.maze.grid_graph, self.player, self.player.bullet)

        self.maze.fig.canvas.draw()

if __name__ == "__main__":
    print("欢迎来到迷宫大逃杀！")
    try:
        w = int(input("请输入迷宫宽度 (建议 10-20): "))
        h = int(input("请输入迷宫高度 (建议 10-20): "))
    except ValueError:
        print("输入无效，使用默认值 15x15")
        w, h = 15, 15

    while True:
        game = MazeGame(w, h)
        game.run()
        if not game.restart_requested:
            break
    print("游戏结束")
