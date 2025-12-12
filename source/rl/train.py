import pygame
import numpy as np
import torch
import os
import sys
import math
import argparse
from collections import defaultdict, deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Argument parsing
parser = argparse.ArgumentParser(description='Train PPO Agent')
parser.add_argument('--mode', type=str, default='gui', choices=['gui', 'cli'], help='Training mode: gui or cli')
parser.add_argument('--load', type=str, default='auto', choices=['auto', 'latest', 'interrupted', 'none'], 
                    help='Load model strategy: auto (prefer interrupted), latest, interrupted, or none (fresh start)')
args, _ = parser.parse_known_args()

if args.mode == 'cli':
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    print("Running in CLI (Headless) mode - Max Speed")

from source.sites import arena
from source import constants as C
from source.rl.ppo_agent import PPOAgent, Memory
from source.parts import generate_maze

class TankEnv:
    def __init__(self, mode='gui'):
        self.mode = mode
        self.arena = arena.Arena()
        self.action_space_n = 6 # Idle, Fwd, Bwd, Left, Right, Fire
        
        # State Dimension Calculation:
        # Self (9) + 2 Enemies (7*2=14) + Shells (4*5=20) + Lidar (8) + Supply (3) + Danger (3) = 57
        self.state_dim = 57
        self.max_steps = 2000
        self.player_ids = [1, 2, 3]
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.get_surface()
        self.paused = False
        
    def reset(self):
        # Setup arena with 3 players
        self.arena.setup(player_num=3, score=(0, 0, 0))
        self.arena.manual_clock = True 
        self.steps = 0
        
        # Track positions for stagnation penalty
        self.prev_positions = {}
        for pid in self.player_ids:
            p = self.arena.get_player(pid)
            if p:
                self.prev_positions[pid] = (p.x, p.y)
            else:
                self.prev_positions[pid] = (0, 0)
                
        # Track nearest enemy distances (网格路径长度) and nearest own-shell-to-enemy distances
        # 用于判断是否在通过迷宫靠近敌人或子弹是否更接近敌人
        self.prev_enemy_distances = {}
        self.prev_shell_min_distances = {}
        self.prev_path_lengths = {}
        try:
            raw_state = self.arena.get_state(normalize=True)
            for pid in self.player_ids:
                # Euclidean fallback (kept for shells)
                self.prev_enemy_distances[pid] = self._nearest_enemy_distance(raw_state, pid)
                self.prev_shell_min_distances[pid] = self._min_shell_enemy_distance(raw_state, pid)
                # Grid-based shortest path length to nearest enemy (steps)
                self.prev_path_lengths[pid] = self._nearest_enemy_path_length(raw_state, pid)
        except Exception:
            # 如果在 reset 时计算失败，使用默认远距离
            for pid in self.player_ids:
                self.prev_enemy_distances[pid] = 1.0
                self.prev_shell_min_distances[pid] = 1.0
                self.prev_path_lengths[pid] = C.COLUMN_NUM * C.ROW_NUM

        return self.get_all_states()
        
    def get_all_states(self):
        states = {}
        raw_state = self.arena.get_state(normalize=True)
        # 预先计算所有墙壁的矩形，用于射线检测加速
        self.wall_rects = [pygame.Rect(w['x']*C.SCREEN_W, w['y']*C.SCREEN_H, w['w']*C.SCREEN_W, w['h']*C.SCREEN_H) 
                           for w in raw_state['walls']]
        
        for pid in self.player_ids:
            states[pid] = self.get_state_vector(raw_state, pid)
        return states

    def step(self, actions):
        # 1. Event Handling & Pause
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                    print(f"Training {'PAUSED' if self.paused else 'RESUMED'}")
                    pygame.display.set_caption(f"RL Training - {'PAUSED' if self.paused else 'RUNNING'}")

        while self.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    self.paused = False
            self.clock.tick(10)

        # 2. Run Game Logic
        if self.mode == 'gui':
            # 加快训练速度：提高帧率上限，并跳过部分渲染
            self.clock.tick(C.FRAME_RATE * 5) # 5x Speed
        else:
            # CLI 模式：全速运行，不限制帧率
            pass
            
        self.steps += 1
        
        keys = defaultdict(int)
        for pid, act in actions.items():
            self.map_action_to_keys(pid, act, keys)
            
        self.arena.update(self.screen, keys)
        
        # 每 5 帧才刷新一次屏幕，大幅减少渲染开销
        if self.mode == 'gui' and self.steps % 5 == 0:
            pygame.display.update()
        
        next_states = self.get_all_states()
        # 当前完整场景状态（归一化），用于距离计算和子弹近敌奖励
        raw_state = self.arena.get_state(normalize=True)
        rewards = {pid: 0 for pid in self.player_ids}
        dones = {pid: False for pid in self.player_ids}
        
        # 3. Reward Calculation
        events = self.arena.get_and_clear_events()
        for event in events:
            if event['type'] == 'hit':
                attacker = event['attacker']
                victim = event['victim']
                
                if attacker == victim:
                    # 自杀惩罚 (重罚)
                    if attacker in rewards: rewards[attacker] -= 100 
                else:
                    # 击杀奖励
                    if attacker in rewards: rewards[attacker] += 1000
                    if victim in rewards: rewards[victim] -= 50
            elif event['type'] == 'get_supply':
                # 拾取道具奖励
                pid = event['player']
                if pid in rewards: rewards[pid] += 50
        
        game_over = self.arena.finished
        
        for pid in self.player_ids:
            p = self.arena.get_player(pid)
            
            # 死亡判定
            if not p or p.dead:
                dones[pid] = True
                # 如果不是因为被击中而死（比如撞墙死或者其他判定），额外给点惩罚
                # 但主要是通过 event 里的 hit 来扣分
            
            if game_over:
                dones[pid] = True
                if p and not p.dead:
                    rewards[pid] += 50 # 胜利奖励
            
            # 存活/行为奖励
            if p and not p.dead:
                rewards[pid] += 0.01 # 降低纯存活奖励，迫使它们去寻找击杀
                
                # 撞墙/贴墙惩罚 + 前方非常近时的撞墙惩罚
                lidar = self.compute_lidar(p)
                front_dist = lidar[0] if len(lidar) > 0 else 1.0
                if min(lidar) < 0.1: # 距离墙壁非常近 (约10像素)
                    rewards[pid] -= 0.5 # 持续的贴墙惩罚

                # 检查是否停滞 (Stagnation Check)
                curr_pos = (p.x, p.y)
                prev_pos = self.prev_positions.get(pid, curr_pos)
                dist_moved = math.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                
                # 每 30 步更新一次位置记录，用于检测长期停滞（放宽触发条件）
                if self.steps % 30 == 0:
                    self.prev_positions[pid] = curr_pos
                    if dist_moved < 20.0: # 如果30步内移动距离小于20像素 (加大移动要求)
                        rewards[pid] -= 5.0 # 停滞惩罚

                # 如果向前且前方非常近，视为撞墙行为，重罚
                if actions.get(pid) == 1 and front_dist < 0.05:
                    rewards[pid] -= 2.0

                # 危险感知惩罚：如果有子弹正向我飞来且距离很近
                # 从 state vector 中提取 danger info (倒数3个)
                # 这里无法直接访问 state vector，重新计算简化版
                # 或者在 get_all_states 时计算并存起来，这里简化处理：
                # 只要有子弹在极近距离 (0.1) 内，就给惩罚
                nearest_shell_dist = self._min_shell_enemy_distance(raw_state, pid) # 这个函数其实是算子弹离敌人的距离...
                # 我们需要算子弹离自己的距离
                min_s_dist = 1.0
                for s in raw_state['shells']:
                     dx = s['x'] - p.x / (C.COLUMN_NUM * C.BLOCK_SIZE * C.MOTION_CALC_SCALE)
                     dy = s['y'] - p.y / (C.ROW_NUM * C.BLOCK_SIZE * C.MOTION_CALC_SCALE)
                     d = math.hypot(dx, dy)
                     if d < min_s_dist: min_s_dist = d
                
                if min_s_dist < 0.1: # 危险距离
                    rewards[pid] -= 0.5 # 处于危险中

                # 奖励：基于迷宫网格的路径长度变化来鼓励导航靠近敌人
                try:
                    # 网格路径长度（steps）——较小表示更接近，通过 delta 来奖励导航行为
                    curr_path_len = self._nearest_enemy_path_length(raw_state, pid)
                    prev_path_len = self.prev_path_lengths.get(pid, C.COLUMN_NUM * C.ROW_NUM)
                    delta_path = prev_path_len - curr_path_len
                    if delta_path > 0:
                        # 每缩短一步给予奖励（系数可调），并限制单步奖励上限
                        rewards[pid] += min(5.0 * delta_path, 10.0) # 加大导航奖励
                        # 如果已经到达同格，给额外小奖励以鼓励接近并尝试击杀
                        if curr_path_len == 0:
                            rewards[pid] += 5.0
                    self.prev_path_lengths[pid] = curr_path_len

                    # 保留子弹接近敌人的奖励，鼓励瞄准/射击
                    curr_shell_dist = self._min_shell_enemy_distance(raw_state, pid)
                    prev_shell = self.prev_shell_min_distances.get(pid, 1.0)
                    delta_shell = prev_shell - curr_shell_dist
                    if delta_shell > 0:
                        rewards[pid] += min(5.0 * delta_shell, 5.0) # 加大射击精度奖励
                    self.prev_shell_min_distances[pid] = curr_shell_dist
                except Exception:
                    pass

                # 危险操作惩罚：如果正前方很近有墙，还开火
                if actions.get(pid) == 5: # Fire
                    rewards[pid] -= 0.1 # 开火成本 (Ammo Cost)
                    if front_dist < 0.2: # 距离很近
                        rewards[pid] -= 2.0 # 惩罚贴脸开火
                    # 惩罚原地开火
                    if not p.moving:
                        rewards[pid] -= 2.0

                # 移动奖励
                # 缓和鼓励移动：小正奖励，未移动不再重罚
                if p.moving:
                    rewards[pid] += 0.1
                else:
                    rewards[pid] -= 0.1 # 稍微惩罚静止


        if self.steps >= self.max_steps:
            game_over = True
            for pid in self.player_ids: dones[pid] = True
            
        return next_states, rewards, dones, game_over

    def compute_lidar(self, p):
        """
        计算 8 个方向的射线距离 (Lidar Sensors)
        返回: 8个 float [0, 1]，1表示远处，0表示贴脸
        """
        sensors = []
        start_x, start_y = p.x / C.MOTION_CALC_SCALE, p.y / C.MOTION_CALC_SCALE
        
        # 8 directions: 0, 45, 90, 135, 180, 225, 270, 315 relative to player heading
        for angle_deg in range(0, 360, 45):
            angle_rad = p.theta + math.radians(angle_deg)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)
            
            dist = 1.0 # Max distance (normalized)
            
            # 简化的射线检测：沿射线采样几个点
            # 真实射线检测太慢，这里检测 5 个关键点: 20px, 50px, 100px, 150px, 200px
            check_points = [20, 50, 100, 150, 200]
            max_range = 200.0
            
            found_wall = False
            for r in check_points:
                cx = start_x + dx * r
                cy = start_y + dy * r
                
                # 边界检查
                if cx < 0 or cx > C.SCREEN_W or cy < 0 or cy > C.SCREEN_H:
                    dist = r / max_range
                    found_wall = True
                    break
                
                # 墙壁检查
                # 简单的点在矩形内检查
                test_rect = pygame.Rect(cx-2, cy-2, 4, 4)
                
                # 优化：只检查附近的墙 (这里为了简单遍历所有墙，性能可能略低，但Python里做空间划分太繁琐)
                # 实际上我们可以利用 self.wall_rects
                for w_rect in self.wall_rects:
                    if w_rect.colliderect(test_rect):
                        dist = r / max_range
                        found_wall = True
                        break
                if found_wall:
                    break
            
            if not found_wall:
                dist = 1.0
                
            sensors.append(dist)
            
        return sensors

    def _nearest_enemy_distance(self, raw_state, pid):
        """
        返回玩家 pid 与最近敌人的最小距离 (归一化坐标)，如果没有敌人返回 1.0
        """
        players = raw_state.get('players', [])
        me = None
        for pp in players:
            if pp.get('name') == pid:
                me = pp
                break
        if not me:
            return 1.0
        mx, my = float(me.get('x', 0.0)), float(me.get('y', 0.0))
        dists = []
        for pp in players:
            if pp.get('name') == pid:
                continue
            if pp.get('dead', False):
                continue
            ex, ey = float(pp.get('x', 0.0)), float(pp.get('y', 0.0))
            dists.append(math.hypot(mx-ex, my-ey))
        if not dists:
            return 1.0
        return min(dists)

    def _pos_to_cell_index(self, nx, ny):
        """
        将归一化坐标 (nx,ny) 映射到格子索引 (0..COLUMN_NUM*ROW_NUM-1)
        """
        try:
            cx = int(min(max(0, nx), 0.9999) * C.COLUMN_NUM)
            cy = int(min(max(0, ny), 0.9999) * C.ROW_NUM)
            return generate_maze.get_idx(cx, cy)
        except Exception:
            # 回退：简单取 (0,0)
            return 0

    def _is_passable_between(self, a_idx, b_idx):
        """
        判断两个相邻格子 a,b 之间是否可通行（即两格之间没有墙）
        只在相邻格子间调用。
        """
        if a_idx < 0 or b_idx < 0 or a_idx >= len(self.arena.cells) or b_idx >= len(self.arena.cells):
            return False
        a = self.arena.cells[a_idx]
        b = self.arena.cells[b_idx]
        # If either cell still has a wall sprite of the separating side, block
        # Check relative position
        ax = a_idx % C.COLUMN_NUM
        ay = a_idx // C.COLUMN_NUM
        bx = b_idx % C.COLUMN_NUM
        by = b_idx // C.COLUMN_NUM
        # Moving right
        if bx == ax + 1 and by == ay:
            # ensure neither has RIGHT/LEFT wall present
            for w in a.walls:
                if getattr(w, 'type', None) == C.RIGHT:
                    return False
            for w in b.walls:
                if getattr(w, 'type', None) == C.LEFT:
                    return False
            return True
        # Moving left
        if bx == ax - 1 and by == ay:
            for w in a.walls:
                if getattr(w, 'type', None) == C.LEFT:
                    return False
            for w in b.walls:
                if getattr(w, 'type', None) == C.RIGHT:
                    return False
            return True
        # Moving down
        if bx == ax and by == ay + 1:
            for w in a.walls:
                if getattr(w, 'type', None) == C.BOTTOM:
                    return False
            for w in b.walls:
                if getattr(w, 'type', None) == C.TOP:
                    return False
            return True
        # Moving up
        if bx == ax and by == ay - 1:
            for w in a.walls:
                if getattr(w, 'type', None) == C.TOP:
                    return False
            for w in b.walls:
                if getattr(w, 'type', None) == C.BOTTOM:
                    return False
            return True
        return False

    def _grid_path_length(self, start_idx, goal_idx):
        """
        使用 BFS 在格子图上搜索最短路径步数（4连通）。返回步数，找不到返回大数。
        """
        if start_idx == goal_idx:
            return 0
        n = C.COLUMN_NUM * C.ROW_NUM
        visited = [False] * n
        q = deque()
        q.append((start_idx, 0))
        visited[start_idx] = True
        while q:
            cur, d = q.popleft()
            cx = cur % C.COLUMN_NUM
            cy = cur // C.COLUMN_NUM
            # neighbors: left, right, up, down
            neighbors = []
            if cx > 0: neighbors.append(cur-1)
            if cx < C.COLUMN_NUM-1: neighbors.append(cur+1)
            if cy > 0: neighbors.append(cur-C.COLUMN_NUM)
            if cy < C.ROW_NUM-1: neighbors.append(cur+C.COLUMN_NUM)
            for nb in neighbors:
                if visited[nb]:
                    continue
                if not self._is_passable_between(cur, nb):
                    continue
                if nb == goal_idx:
                    return d+1
                visited[nb] = True
                q.append((nb, d+1))
        return C.COLUMN_NUM * C.ROW_NUM

    def _nearest_enemy_path_length(self, raw_state, pid):
        """
        返回玩家 pid 到最近活着敌人的网格最短路径步数（BFS），找不到返回大数
        """
        players = raw_state.get('players', [])
        me = None
        for pp in players:
            if pp.get('name') == pid:
                me = pp
                break
        if not me:
            return C.COLUMN_NUM * C.ROW_NUM
        start_idx = self._pos_to_cell_index(float(me.get('x', 0.0)), float(me.get('y', 0.0)))
        best = C.COLUMN_NUM * C.ROW_NUM
        for pp in players:
            if pp.get('name') == pid:
                continue
            if pp.get('dead', False):
                continue
            goal_idx = self._pos_to_cell_index(float(pp.get('x', 0.0)), float(pp.get('y', 0.0)))
            d = self._grid_path_length(start_idx, goal_idx)
            if d < best:
                best = d
        return best

    def _min_shell_enemy_distance(self, raw_state, pid):
        """
        返回该玩家所拥有的所有子弹到最近敌人的最小距离 (归一化)，没有则返回 1.0
        """
        shells = raw_state.get('shells', [])
        players = raw_state.get('players', [])
        enemy_positions = [(float(p['x']), float(p['y'])) for p in players if p.get('name') != pid and not p.get('dead', False)]
        if not enemy_positions or not shells:
            return 1.0
        min_dist = 1.0
        for s in shells:
            if int(s.get('owner', -1)) != pid:
                continue
            sx, sy = float(s.get('x', 0.0)), float(s.get('y', 0.0))
            for ex, ey in enemy_positions:
                d = math.hypot(sx-ex, sy-ey)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def get_state_vector(self, raw_state, pid):
        p_self = self.arena.get_player(pid)
        
        vec = []
        
        # 1. Self Info (9)
        if p_self:
            vec.extend([
                p_self.x / (C.COLUMN_NUM * C.BLOCK_SIZE * C.MOTION_CALC_SCALE),
                p_self.y / (C.ROW_NUM * C.BLOCK_SIZE * C.MOTION_CALC_SCALE),
                np.sin(p_self.theta), np.cos(p_self.theta),
                1.0 if p_self.can_fire else 0.0,
                p_self.fire_cool_down_timer / C.FIRE_COOL_DOWN, # Approx
                len(p_self.rounds) / 5.0,
                1.0 if p_self.moving else 0.0,
                1.0 if p_self.not_stuck else 0.0
            ])
        else:
            vec.extend([0]*9)

        # 2. Enemies Info (14) - Relative Position & Aiming Info
        enemies = []
        for p in raw_state['players']:
            if p['name'] != pid:
                enemies.append(p)
        
        # Sort by distance
        if p_self:
            enemies.sort(key=lambda e: (e['x'] - vec[0])**2 + (e['y'] - vec[1])**2)
            
        for i in range(2):
            if i < len(enemies):
                e = enemies[i]
                # Use Relative Position! Easier for AI to learn aiming
                dx = e['x'] - (vec[0] if p_self else 0)
                dy = e['y'] - (vec[1] if p_self else 0)
                
                # Calculate distance and relative angle for aiming
                dist = math.sqrt(dx**2 + dy**2)
                
                # Angle to enemy relative to self heading
                # atan2(dy, dx) gives angle of vector to enemy
                # p_self.theta is my current heading
                angle_to_enemy = math.atan2(dy, dx)
                # p_self.theta is in radians [0, 2PI] usually, but let's just subtract
                my_theta = p_self.theta if p_self else 0
                rel_angle = angle_to_enemy - my_theta
                
                # Normalize to [-PI, PI]
                while rel_angle > math.pi: rel_angle -= 2*math.pi
                while rel_angle < -math.pi: rel_angle += 2*math.pi
                
                vec.extend([dx, dy, dist, rel_angle, e['sin_theta'], e['cos_theta'], e['dead']])
            else:
                vec.extend([0]*7)

        # 3. Shells Info (20) - Relative Position & Velocity
        shells = list(raw_state['shells'])
        if p_self:
            # Sort by danger (distance)
            shells.sort(key=lambda s: (s['x'] - vec[0])**2 + (s['y'] - vec[1])**2)
            
        for i in range(5):
            if i < len(shells):
                s = shells[i]
                dx = s['x'] - (vec[0] if p_self else 0)
                dy = s['y'] - (vec[1] if p_self else 0)
                vec.extend([dx, dy, s['vx'], s['vy']])
            else:
                vec.extend([0]*4)

        # 4. Lidar Sensors (8)
        if p_self:
            lidar = self.compute_lidar(p_self)
            vec.extend(lidar)
        else:
            vec.extend([0]*8)

        # 5. Supply Info (3) - Nearest Supply
        supplies = raw_state.get('supplies', [])
        if p_self and supplies:
            # Find nearest supply
            supplies.sort(key=lambda s: (s['x'] - vec[0])**2 + (s['y'] - vec[1])**2)
            s = supplies[0]
            dx = s['x'] - vec[0]
            dy = s['y'] - vec[1]
            dist = math.sqrt(dx**2 + dy**2)
            # Type: 0 for normal, 1 for shotgun (just an example, depends on raw_state)
            # Assuming raw_state supplies has 'type' or similar, otherwise just pos
            vec.extend([dx, dy, dist])
        else:
            vec.extend([0, 0, 1.0]) # No supply visible/exist

        # 6. Danger Sensor (3) - Nearest Incoming Bullet
        # Find nearest bullet that is moving TOWARDS the player
        nearest_danger_dist = 1.0
        danger_angle = 0.0
        closing_speed = 0.0
        
        if p_self:
            my_x, my_y = vec[0], vec[1]
            for s in shells: # shells list from section 3
                # Check if bullet is moving towards me
                # Vector from bullet to me
                to_me_x = my_x - s['x']
                to_me_y = my_y - s['y']
                dist = math.sqrt(to_me_x**2 + to_me_y**2)
                if dist < 0.001: continue
                
                # Normalize
                to_me_x /= dist
                to_me_y /= dist
                
                # Bullet velocity (normalized roughly)
                v_dot = s['vx'] * to_me_x + s['vy'] * to_me_y
                
                if v_dot > 0: # Moving towards me
                    if dist < nearest_danger_dist:
                        nearest_danger_dist = dist
                        closing_speed = v_dot
                        # Relative angle
                        angle_to_bullet = math.atan2(s['y'] - my_y, s['x'] - my_x)
                        my_theta = p_self.theta
                        rel_angle = angle_to_bullet - my_theta
                        while rel_angle > math.pi: rel_angle -= 2*math.pi
                        while rel_angle < -math.pi: rel_angle += 2*math.pi
                        danger_angle = rel_angle
        
        vec.extend([nearest_danger_dist, danger_angle, closing_speed])

        return np.array(vec, dtype=np.float32)

    def map_action_to_keys(self, pid, action, keys):
        # 0: Idle, 1: Fwd, 2: Bwd, 3: Left, 4: Right, 5: Fire
        k_fwd, k_bwd, k_left, k_right, k_fire = 0, 0, 0, 0, 0
        
        if pid == 1:
            k_fwd, k_bwd, k_left, k_right, k_fire = pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_q
        elif pid == 2:
            k_fwd, k_bwd, k_left, k_right, k_fire = pygame.K_i, pygame.K_k, pygame.K_j, pygame.K_l, pygame.K_u
        elif pid == 3:
            k_fwd, k_bwd, k_left, k_right, k_fire = pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DELETE
            
        if action == 1: keys[k_fwd] = 1
        elif action == 2: keys[k_bwd] = 1
        elif action == 3: keys[k_left] = 1
        elif action == 4: keys[k_right] = 1
        elif action == 5: keys[k_fire] = 1
    
    def debug_draw_lidar(self):
        # 简单的调试绘制，画出每个玩家的射线
        for pid in self.player_ids:
            p = self.arena.get_player(pid)
            if p and not p.dead:
                lidar = self.compute_lidar(p)
                start_x, start_y = p.x / C.MOTION_CALC_SCALE, p.y / C.MOTION_CALC_SCALE
                for i, dist in enumerate(lidar):
                    angle_rad = p.theta + math.radians(i * 45)
                    end_x = start_x + math.cos(angle_rad) * (dist * 200)
                    end_y = start_y + math.sin(angle_rad) * (dist * 200)
                    color = (0, 255, 0) if dist > 0.2 else (255, 0, 0)
                    pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 1)

def train():
    env = TankEnv(mode=args.mode)
    state_dim = env.state_dim
    action_dim = env.action_space_n
    
    # 使用 3 个独立的 Agent，让它们发展出不同的性格/策略
    # 这样可以避免所有 AI 同时陷入同一个局部最优（比如都只会原地开火）
    agents = {pid: PPOAgent(state_dim, action_dim) for pid in env.player_ids}
    
    # 尝试加载旧模型
    print(f"Checking for existing models (Strategy: {args.load})...")
    for pid in env.player_ids:
        model_path = None
        
        if args.load == 'none':
            print(f"Player {pid}: Starting fresh (load=none)")
            continue
            
        path_interrupted = f'ppo_agent{pid}_interrupted.pth'
        path_latest = f'ppo_agent{pid}_latest.pth'
        
        if args.load == 'auto':
            if os.path.exists(path_interrupted):
                model_path = path_interrupted
            elif os.path.exists(path_latest):
                model_path = path_latest
        elif args.load == 'interrupted':
            if os.path.exists(path_interrupted):
                model_path = path_interrupted
        elif args.load == 'latest':
            if os.path.exists(path_latest):
                model_path = path_latest
            
        if model_path:
            print(f"Loading model for Player {pid} from {model_path}")
            try:
                # 即使保存时在CPU，加载时也会适配当前Agent的设备
                state_dict = torch.load(model_path)
                agents[pid].policy.load_state_dict(state_dict)
                agents[pid].policy_old.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading model for Player {pid}: {e}")
        else:
            print(f"No model found for Player {pid} with strategy '{args.load}', starting from scratch.")

    memories = {pid: Memory() for pid in env.player_ids}
    
    max_episodes = 5000
    update_timestep = 8000 # Increased for better GPU utilization and stable updates
    time_step = 0
    
    print("Starting MULTI-AGENT training (3 Separate Brains)...")
    
    try:
        for i_episode in range(1, max_episodes+1):
            states = env.reset()
            ep_rewards = {pid: 0 for pid in env.player_ids}
            active_players = set(env.player_ids)
            
            for t in range(env.max_steps):
                time_step += 1
                
                actions = {}
                log_probs = {}
                
                # 每个玩家使用自己的 Agent 决策
                for pid in active_players:
                    action, log_prob = agents[pid].select_action(states[pid])
                    actions[pid] = action
                    log_probs[pid] = log_prob
                    
                next_states, rewards, dones, game_over = env.step(actions)
                
                # 分别存储经验
                for pid in active_players:
                    memories[pid].states.append(torch.FloatTensor(states[pid]))
                    memories[pid].actions.append(torch.tensor(actions[pid]))
                    memories[pid].logprobs.append(torch.tensor(log_probs[pid]))
                    memories[pid].rewards.append(rewards[pid])
                    memories[pid].is_terminals.append(dones[pid])
                    
                    ep_rewards[pid] += rewards[pid]
                
                states = next_states
                
                for pid in list(active_players):
                    if dones[pid]:
                        active_players.remove(pid)
                
                # 更新所有 Agent
                if time_step % update_timestep == 0:
                    for pid in env.player_ids:
                        agents[pid].update(memories[pid])
                        memories[pid].clear_memory()
                    time_step = 0
                    
                if game_over or not active_players:
                    break
                    
            if i_episode % 10 == 0:
                print(f"Ep {i_episode} | Steps: {time_step} | Rewards: {ep_rewards}")
                
            if i_episode % 50 == 0:
                for pid in env.player_ids:
                    torch.save(agents[pid].policy.state_dict(), f'ppo_agent{pid}_latest.pth')

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving models...")
        for pid in env.player_ids:
            torch.save(agents[pid].policy.state_dict(), f'ppo_agent{pid}_interrupted.pth')
        print("Models saved successfully.")
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    train()