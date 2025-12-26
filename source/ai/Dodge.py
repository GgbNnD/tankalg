import math
import pygame
from .. import constants as C
from ..parts import cell

class DodgeStrategy:
    def __init__(self, player):
        self.player = player
        self.arena = player.arena
        self.threat_range = 600 # Increased detection range
        self.max_predict_steps = 150
        self.active_strategy = None # Current dodge command queue
        self.debug_timer = 0

    def get_dodge_action(self, all_trajectories=None):
        """
        判断 AI 是否需要闪避并返回相应动作。
        若需要闪避，返回包含 'steering' 和 'throttle' 的字典，否则返回 None。
        """
        # 1. 威胁检测与预测
        if all_trajectories is None:
            all_trajectories = self.get_all_bullet_trajectories()
            
        # 过滤出当前将要命中我们的威胁
        threats = []
        my_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        
        for dist, s, trajectory in all_trajectories:
             if self.check_trajectory_collision(trajectory, my_pos):
                threats.append((dist, s, trajectory))
        
        # 调试打印（1 秒间隔）
        current_time = pygame.time.get_ticks()
        if current_time - self.debug_timer > 1000:
            self.debug_timer = current_time
            if threats:
                print(f"[DODGE DEBUG] Threats: {len(threats)} | Closest Dist: {threats[0][0]:.2f}")
                # 检查 generate_dodge_strategy 是否能找到解
                action = self.generate_dodge_strategy(threats, all_trajectories)
                print(f"[DODGE DEBUG] Action: {action}")
            elif all_trajectories:
                # 即使没有威胁，打印最近子弹信息以检查检测范围
                print(f"[DODGE DEBUG] No Threats. Nearest Bullet: {all_trajectories[0][0]:.2f}")
            else:
                print("[DODGE DEBUG] No Bullets Detected")
        
        if not threats:
            self.active_strategy = None
            return None
            
        # 2. 策略生成
        # 传入所有轨迹以确保不会躲到另一颗子弹的路径上
        action = self.generate_dodge_strategy(threats, all_trajectories)
        return action

    def get_all_bullet_trajectories(self):
        """
        返回所有附近炮弹的列表，格式为 (距离, shell 对象, 预测轨迹)。
        """
        trajectories = []
        shells = self.arena.get_shells(normalize=False)
        
        # 收集实际的炮弹对象
        all_shells = []
        for p in self.arena.players:
            for s in p.rounds:
                all_shells.append(s)
                
        my_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        
        for s in all_shells:
            # 忽略刚发射并处于保护期的自有子弹
            if s.player == self.player and s.protection:
                continue
                
            # 距离检查
            s_pos = (s.x / C.MOTION_CALC_SCALE, s.y / C.MOTION_CALC_SCALE)
            dist = math.hypot(s_pos[0] - my_pos[0], s_pos[1] - my_pos[1])
            
            if dist > self.threat_range:
                continue
                
            # 预测轨迹
            trajectory = self.predict_trajectory(s)
            trajectories.append((dist, s, trajectory))
            
        # 按距离排序
        trajectories.sort(key=lambda x: x[0])
        return trajectories

    def get_threats(self):
        # 兼容旧接口的封装方法，当前推荐使用 get_all_bullet_trajectories
        trajectories = self.get_all_bullet_trajectories()
        threats = []
        my_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        for dist, s, trajectory in trajectories:
             if self.check_trajectory_collision(trajectory, my_pos):
                threats.append((dist, s, trajectory))
        return threats

    def predict_trajectory(self, shell):
        """
        预测炮弹在最多 max_predict_steps 步内的路径。
        返回线段列表 [(起点, 终点), ...]。
        """
        segments = []
        
        # Clone shell state for simulation
        curr_x = shell.x
        curr_y = shell.y
        curr_vx = shell.vx
        curr_vy = shell.vy
        
        # 模拟循环
        # 我们逐步模拟物理，但也可根据需要做加速优化
        
        sim_steps = self.max_predict_steps
        
        seg_start = (curr_x / C.MOTION_CALC_SCALE, curr_y / C.MOTION_CALC_SCALE)
        
        for _ in range(sim_steps):
            next_x = curr_x + curr_vx
            next_y = curr_y + curr_vy
            
            # 与墙体的碰撞检测（简化版的 shell.update_collisions）
            # 需要检测反弹
            
            # 转换为屏幕坐标以便墙体检测
            cx, cy = next_x / C.MOTION_CALC_SCALE, next_y / C.MOTION_CALC_SCALE
            
            # 检查墙体碰撞
            # 逐步精确检测较为昂贵，可用射线加速，但此处采用步进以获得更高准确性
            
            bounced = False
            
            # 获取附近的墙格索引
            cells_idx = cell.calculate_cell_num(cx, cy)
            
            # 创建用于碰撞检测的临时矩形，炮弹尺寸较小
            s_rect = pygame.Rect(0, 0, C.ROUND_PX, C.ROUND_PY)
            s_rect.center = (cx, cy)
            
            for i in range(4):
                if i >= len(cells_idx): break
                c = self.arena.cells[cells_idx[i]]
                for w in c.walls:
                    if s_rect.colliderect(w.rect):
                        # 反弹逻辑
                        if abs(w.rect.top - s_rect.bottom) <= C.ROUND_COLLITION_EPS + 2 and curr_vy > 0:
                            curr_vy *= -1
                            bounced = True
                        elif abs(w.rect.bottom - s_rect.top) <= C.ROUND_COLLITION_EPS + 2 and curr_vy < 0:
                            curr_vy *= -1
                            bounced = True
                        elif abs(w.rect.right - s_rect.left) <= C.ROUND_COLLITION_EPS + 2 and curr_vx < 0:
                            curr_vx *= -1
                            bounced = True
                        elif abs(w.rect.left - s_rect.right) <= C.ROUND_COLLITION_EPS + 2 and curr_vx > 0:
                            curr_vx *= -1
                            bounced = True
                        
                        if bounced: break
                if bounced: break
            
            curr_x = next_x
            curr_y = next_y
            
            if bounced:
                seg_end = (curr_x / C.MOTION_CALC_SCALE, curr_y / C.MOTION_CALC_SCALE)
                segments.append((seg_start, seg_end))
                seg_start = seg_end
                
        # 添加最终线段
        seg_end = (curr_x / C.MOTION_CALC_SCALE, curr_y / C.MOTION_CALC_SCALE)
        segments.append((seg_start, seg_end))
        
        return segments

    def check_trajectory_collision(self, trajectory, pos):
        # 检查任一线段是否与玩家的命中区域相交（近似为圆/矩形）
        # 玩家半径约 + 子弹半径约
        # C.PLAYER_PX 为 41，C.ROUND_PX 为 9。
        # 半径和 = (41 + 9) / 2 = 25。
        # 加上小安全边距 (+5) = 30。
        radius = 30 
        
        for start, end in trajectory:
            if self.line_intersects_circle(start, end, pos, radius):
                return True
        return False

    def line_intersects_circle(self, p1, p2, center, radius):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = center
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0: return False
        t = ((cx - x1) * dx + (cy - y1) * dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        dist_sq = (closest_x - cx)**2 + (closest_y - cy)**2
        return dist_sq <= radius**2

    def generate_dodge_strategy(self, threats, all_trajectories=None):
        # 优先策略 1：旋转
        # 尝试原地旋转以减小目标截面？
        # 在 Tank Trouble 中玩家近似圆形，但命中盒较复杂，旋转可能将部分命中区移开。
        
        # 优先策略 2：旋转并移动（侧移躲闪）
        # 根据子弹速度方向决定躲避方向
        
        # 我们对不同动作进行模拟并检查是否安全
        
        # 简化方法：
        # 尝试若干基础动作：前进、后退、顺时针旋转、逆时针旋转等
        
        best_action = None
        
        # Candidates: (steering, throttle)
        candidates = [
            (0, 0), # Stop
            (1, 0), # Rotate CW
            (-1, 0), # Rotate CCW
            (0, 1), # Forward
            (0, -1), # Backward
            (1, 1), # Forward Right
            (-1, 1), # Forward Left
            (1, -1), # Backward Right
            (-1, -1) # Backward Left
        ]
        
        # 若未提供 all_trajectories，则使用 threats（其为子集）
        # 理想情况应使用所有轨迹以避免躲入其他子弹路径
        check_against = all_trajectories if all_trajectories is not None else threats
        
        # 第一轮：寻找完全安全的动作
        for steering, throttle in candidates:
            if self.simulate_action_safety(steering, throttle, check_against):
                return {'steering': steering, 'throttle': throttle}
                
        # 第二轮：如果没有安全动作，找出"最不危险"的动作
        # 我们希望最大化与最近威胁的最小距离
        best_candidate = (0, 0)
        max_min_dist = -1.0
        
        for steering, throttle in candidates:
            # Simulate final position after N frames
            final_pos = self.simulate_movement(steering, throttle, steps=30)
            if final_pos is None: # 碰撞墙体
                continue
                
            # 计算到任一威胁轨迹的最小距离
            min_dist = float('inf')
            for _, _, trajectory in check_against:
                d = self.get_distance_to_trajectory(trajectory, final_pos)
                if d < min_dist:
                    min_dist = d
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_candidate = (steering, throttle)
                
        # 返回找到的最佳候选动作（比停止更好则返回该动作，否则返回停止）
        return {'steering': best_candidate[0], 'throttle': best_candidate[1]}

    def is_action_safe(self, steering, throttle, all_trajectories):
        """
        对外接口：检查某动作是否相对于所有轨迹是安全的。
        """
        return self.simulate_action_safety(steering, throttle, all_trajectories)

    def simulate_movement(self, steering, throttle, steps=30):
        """
        模拟移动并返回最终位置 (cx, cy)（像素坐标）。
        若发生与墙体碰撞则返回 None。
        """
        sim_x = self.player.x
        sim_y = self.player.y
        sim_theta = self.player.theta
        
        for _ in range(steps):
            if throttle != 0:
                v = C.BASE_MOVE_V if throttle > 0 else C.BACKWARD_V
                sim_x += throttle * v * math.cos(sim_theta)
                sim_y += throttle * v * math.sin(sim_theta)
            
            if steering != 0:
                sim_theta += steering * C.BASE_TURN_W
                
            # 墙体检测
            cx, cy = sim_x / C.MOTION_CALC_SCALE, sim_y / C.MOTION_CALC_SCALE
            cells_idx = cell.calculate_cell_num(cx, cy)
            p_rect = pygame.Rect(0, 0, C.PLAYER_PX, C.PLAYER_PY)
            p_rect.center = (cx, cy)
            
            for i in range(4):
                if i >= len(cells_idx): break
                c = self.arena.cells[cells_idx[i]]
                for w in c.walls:
                    if p_rect.colliderect(w.rect):
                        return None # 碰到墙体
                        
        return (sim_x / C.MOTION_CALC_SCALE, sim_y / C.MOTION_CALC_SCALE)

    def simulate_action_safety(self, steering, throttle, threats):
        # 对玩家动作进行若干帧的模拟
        # 检查模拟位置是否与任一威胁轨迹相交
        
        # Current state
        sim_x = self.player.x
        sim_y = self.player.y
        sim_theta = self.player.theta
        
        # 模拟 30 帧（约 0.5 秒，60fps），足以移动约 30 像素
        # 将步数从 10 增至 30，以便更有机会脱离碰撞半径
        for _ in range(30):
            # Apply physics (Simplified)
            if throttle != 0:
                v = C.BASE_MOVE_V if throttle > 0 else C.BACKWARD_V
                # 注意：此处不要再乘以 MOTION_CALC_SCALE。
                # BASE_MOVE_V 的单位已可直接用于更新 self.x
                sim_x += throttle * v * math.cos(sim_theta)
                sim_y += throttle * v * math.sin(sim_theta)
            
            if steering != 0:
                sim_theta += steering * C.BASE_TURN_W
                
            # 检查墙体碰撞（可行性判定）
            # 若撞墙，该动作无效
            cx, cy = sim_x / C.MOTION_CALC_SCALE, sim_y / C.MOTION_CALC_SCALE
            # Simple check: is center inside a wall?
            # (Full collision check is complex, assume center check is enough for feasibility)
            cells_idx = cell.calculate_cell_num(cx, cy)
            p_rect = pygame.Rect(0, 0, C.PLAYER_PX, C.PLAYER_PY)
            p_rect.center = (cx, cy)
            
            for i in range(4):
                if i >= len(cells_idx): break
                c = self.arena.cells[cells_idx[i]]
                for w in c.walls:
                    if p_rect.colliderect(w.rect):
                        return False # 碰到墙体，动作无效
            
            # 检查与威胁的碰撞（安全性）
            sim_pos = (cx, cy)
            for _, _, trajectory in threats:
                if self.check_trajectory_collision(trajectory, sim_pos):
                    return False # Still hits bullet
                    
        return True

    def get_distance_to_trajectory(self, trajectory, pos):
        """
        返回 pos 到轨迹中任意线段的最小距离。
        """
        min_dist_sq = float('inf')
        px, py = pos
        
        for p1, p2 in trajectory:
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1
            
            if dx == 0 and dy == 0:
                d_sq = (px - x1)**2 + (py - y1)**2
            else:
                t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                closest_x = x1 + t * dx
                closest_y = y1 + t * dy
                d_sq = (closest_x - px)**2 + (closest_y - py)**2
                
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                
        return math.sqrt(min_dist_sq)
