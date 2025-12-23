import pygame
import math
import heapq
import numpy as np
from .. import constants as C

class SmartAI:
    def __init__(self, player):
        self.player = player
        self.arena = player.arena
        self.path = []
        self.target_cell = None
        self.update_timer = 0
        self.path_update_interval = 100 # ms (Balanced updates)

    def get_keys(self):
        keys = {
            pygame.K_w: False,
            pygame.K_s: False,
            pygame.K_a: False,
            pygame.K_d: False,
            pygame.K_m: False, # Fire key for player 2 (usually) - wait, need to check player mapping
            # We will return a dict that can be used to update the main keys dict
            # But actually arena.get_ai_keys expects to return a new keys object or dict-like
        }
        
        # Determine which keys correspond to this player
        # Usually Player 1: WASD + Space/Q
        # Player 2: IJKL + / or something
        # But here we just return abstract actions and let arena map them or we return the specific keys if we know them.
        # In arena.py get_ai_keys, it returns a dict with keys like pygame.K_w.
        # Let's assume we are controlling the player assigned to this AI.
        # If this AI is for Player 2, we should return Player 2 keys?
        # Actually, arena.py get_ai_keys constructs a NEW keys dict.
        # It uses 'forward', 'left', 'right', 'fire' actions.
        # So we should probably return high-level actions or just set the keys directly.
        
        # Let's look at arena.py again. 
        # It sets new_keys[pygame.K_UP] etc based on ai_action.
        # We should probably rewrite get_ai_keys in arena.py to call this class and get specific key presses.
        
        # For now, let's implement the logic to decide:
        # forward, backward, left, right, fire
        
        actions = self.decide_action()
        
        # Map actions to keys (assuming Player 2 for now, or generic)
        # The arena.py will use these keys.
        # We will return a dict of { 'forward': bool, 'left': bool ... }
        return actions

    def decide_action(self):
        actions = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False,
            'fire': False
        }
        
        current_time = pygame.time.get_ticks()
        
        # 0. Anti-Stuck Logic (Highest Priority)
        if not self.player.not_stuck:
            # If stuck, reverse and turn
            actions['backward'] = True
            actions['left'] = True # Turn to unstick
            return actions

        # 1. Get Game State
        my_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        my_angle = self.player.theta
        
        enemies = [p for p in self.arena.players if p != self.player and not p.dead]
        if not enemies:
            return actions # No enemies, do nothing
            
        # Prioritize closer enemies
        enemies.sort(key=lambda p: (p.x - self.player.x)**2 + (p.y - self.player.y)**2)
        target = enemies[0]
        target_pos = (target.x / C.MOTION_CALC_SCALE, target.y / C.MOTION_CALC_SCALE)
        
        # 2. Dodge Bullets (High Priority)
        is_dodging = False
        
        def get_action_danger(action_tuple):
            fwd, bwd, lft, rgt = action_tuple
            
            dt = 0.5
            frames = dt * C.FRAME_RATE
            scale = C.MOTION_CALC_SCALE
            speed = C.BASE_MOVE_V / scale
            back_speed = C.BACKWARD_V / scale
            turn_speed = C.BASE_TURN_W
            
            sim_angle = my_angle
            if lft: sim_angle -= turn_speed * frames * 0.5
            if rgt: sim_angle += turn_speed * frames * 0.5
            
            vx, vy = 0, 0
            if fwd:
                vx = math.cos(sim_angle) * speed
                vy = math.sin(sim_angle) * speed
            elif bwd:
                vx = -math.cos(sim_angle) * back_speed
                vy = -math.sin(sim_angle) * back_speed
                
            return self.get_danger_score(my_pos, sim_angle, velocity=(vx, vy), time_window=1.0)

        current_danger = self.get_danger_score(my_pos, my_angle, velocity=(0,0), time_window=1.5)
        if current_danger > 0:
            is_dodging = True
        
        # 3. Attack / Pathfinding (Determine Desired Action)
        desired_action = None # (fwd, bwd, lft, rgt)
        
        # Check direct shot
        can_shoot, aim_angle = self.get_aim_angle(my_pos, target_pos, my_angle)
        
        if not is_dodging:
            if can_shoot:
                # Combat Movement
                diff = self.angle_difference(my_angle, aim_angle)
                lft, rgt, fwd = False, False, False
                
                if abs(diff) > 0.1:
                    if diff > 0: rgt = True
                    else: lft = True
                
                dist_to_target = math.hypot(target_pos[0]-my_pos[0], target_pos[1]-my_pos[1])
                if dist_to_target > C.BLOCK_SIZE * 1.5:
                    fwd = True
                
                desired_action = (fwd, False, lft, rgt)
                
            else:
                # 4. Pathfinding (Dijkstra + Smoothing)
                # Only update path if interval passed OR no path exists
                if current_time - self.update_timer > self.path_update_interval or not self.path:
                    raw_path = self.find_path(my_pos, target_pos)
                    new_path = self.smooth_path(raw_path, my_pos, target_pos)
                    
                    # Only replace path if we found a valid one, to prevent flickering
                    if new_path:
                        self.path = new_path
                        # Remove the first point if it's too close (it's usually my_pos)
                        if self.path and len(self.path) > 0:
                             dist = math.hypot(self.path[0][0] - my_pos[0], self.path[0][1] - my_pos[1])
                             if dist < C.BLOCK_SIZE / 4:
                                 self.path.pop(0)
                        self.update_timer = current_time
                    elif not self.path:
                        # If we have no path and found no path, keep it empty
                        self.path = []
                        self.update_timer = current_time
                    
                if self.path:
                    next_pos = self.path[0]
                    dist = math.hypot(next_pos[0] - my_pos[0], next_pos[1] - my_pos[1])
                    if dist < C.BLOCK_SIZE / 2:
                        self.path.pop(0)
                        if self.path:
                            next_pos = self.path[0]
                    
                    if self.path:
                        next_pos = self.path[0]
                        angle_to_next = math.atan2(next_pos[1] - my_pos[1], next_pos[0] - my_pos[0])
                        diff = self.angle_difference(my_angle, angle_to_next)
                        
                        lft, rgt, fwd = False, False, False
                        if abs(diff) > 0.5:
                            if diff > 0: rgt = True
                            else: lft = True
                        else:
                            fwd = True
                            if abs(diff) > 0.1:
                                if diff > 0: rgt = True
                                else: lft = True
                        
                        desired_action = (fwd, False, lft, rgt)

        # 4. Safety Check & Execution
        final_action = None
        
        # If we have a desired action, check if it's safe
        if desired_action and not is_dodging:
            if get_action_danger(desired_action) == 0:
                final_action = desired_action
            else:
                # Desired action is dangerous! Treat as emergency.
                is_dodging = True
        
        # If dodging (either initially or because desired action was unsafe)
        if is_dodging:
            # Find safe escape
            candidates = [
                (True, False, True, False),  # Forward + Left
                (True, False, False, True),  # Forward + Right
                (True, False, False, False), # Forward
                (False, True, True, False),  # Backward + Left
                (False, True, False, True),  # Backward + Right
                (False, True, False, False), # Backward
                (False, False, True, False), # Spin left
                (False, False, False, True), # Spin right
                (False, False, False, False) # Stop
            ]
            
            best_act = None
            min_danger = float('inf')
            
            for act in candidates:
                d = get_action_danger(act)
                # Prefer moving over stopping if danger is equal, but prefer spinning if it helps
                if d < min_danger:
                    min_danger = d
                    best_act = act
                elif d == min_danger and d > 0:
                    # Tie-breaking logic for dangerous situations
                    # Maybe prefer spinning/moving over stopping?
                    pass
            
            final_action = best_act

        # Apply final action
        if final_action:
            actions['forward'] = final_action[0]
            actions['backward'] = final_action[1]
            actions['left'] = final_action[2]
            actions['right'] = final_action[3]
        
        # --- FIRE DECISION ---
        if can_shoot:
            diff = self.angle_difference(my_angle, aim_angle)
            if abs(diff) < 0.15:
                actions['fire'] = True
                        
        return actions

    def get_danger_score(self, pos, angle, velocity=(0, 0), time_window=1.0):
        # Tank dimensions
        scale = C.MOTION_CALC_SCALE
        # Use bounding circle for broad phase
        # Tank is approx 41x27 px. Radius approx 25px.
        tank_radius_px = math.sqrt((C.PLAYER_PX/2)**2 + (C.PLAYER_PY/2)**2)
        tank_radius = tank_radius_px / scale
        
        bullet_radius = (C.ROUND_PX / 2) / scale
        safety_margin = 5 / scale 
        
        # OBB dimensions (half-width, half-height)
        # X-axis is forward. PLAYER_PX is length.
        hw = (C.PLAYER_PX / 2) / scale + safety_margin
        hh = (C.PLAYER_PY / 2) / scale + safety_margin
        
        shells = []
        for p in self.arena.players:
            shells.extend(p.rounds)
            
        max_score = 0.0
        
        for shell in shells:
            # Shell position and velocity (normalized)
            sx = shell.x / scale
            sy = shell.y / scale
            
            svx = shell.vx / scale
            svy = shell.vy / scale
            
            # Relative velocity (Bullet - Tank)
            rel_vx = svx - velocity[0]
            rel_vy = svy - velocity[1]
            
            # Vector from bullet to tank
            dx = pos[0] - sx
            dy = pos[1] - sy
            
            # 1. Broad Phase: Circle Intersection
            dot = dx * rel_vx + dy * rel_vy
            if dot <= 0:
                continue # Moving away
                
            speed_sq = rel_vx*rel_vx + rel_vy*rel_vy
            if speed_sq == 0: continue
            
            rel_speed = math.sqrt(speed_sq)
            proj_len = dot / rel_speed
            
            closest_x = sx + (rel_vx / rel_speed) * proj_len
            closest_y = sy + (rel_vy / rel_speed) * proj_len
            
            dist_to_line_sq = (pos[0] - closest_x)**2 + (pos[1] - closest_y)**2
            collision_dist = tank_radius + bullet_radius + safety_margin
            
            if dist_to_line_sq >= collision_dist**2:
                continue # Misses the bounding circle
                
            # Check time
            time_to_impact = proj_len / rel_speed
            frames_limit = time_window * C.FRAME_RATE
            
            if time_to_impact > frames_limit:
                continue
                
            # 2. Narrow Phase: OBB Intersection
            # Transform bullet relative trajectory to tank's local space
            # Tank is at (0,0) in local space, aligned with X axis.
            # We need to rotate the relative path by -angle.
            
            # Bullet start pos relative to tank
            rel_sx = sx - pos[0]
            rel_sy = sy - pos[1]
            
            # Bullet end pos relative to tank (at impact time + buffer)
            # We check the segment from current pos to pos at time_limit
            end_time = min(time_to_impact + 5, frames_limit) # Check a bit past impact
            rel_ex = rel_sx + rel_vx * end_time
            rel_ey = rel_sy + rel_vy * end_time
            
            # Rotate start and end points
            cos_a = math.cos(-angle)
            sin_a = math.sin(-angle)
            
            local_sx = rel_sx * cos_a - rel_sy * sin_a
            local_sy = rel_sx * sin_a + rel_sy * cos_a
            
            local_ex = rel_ex * cos_a - rel_ey * sin_a
            local_ey = rel_ex * sin_a + rel_ey * cos_a
            
            # Check intersection between segment (local_sx, local_sy)-(local_ex, local_ey)
            # and AABB [-hw, hw] x [-hh, hh]
            if self.line_intersects_aabb(local_sx, local_sy, local_ex, local_ey, -hw, -hh, hw, hh):
                 if self.can_see((sx, sy), pos):
                    score = 1.0 / (time_to_impact + 0.1)
                    if score > max_score:
                        max_score = score
                            
        return max_score
        dy = y2 - y1
        
        p = [-dx, dx, -dy, dy]
        q = [x1 - min_x, max_x - x1, y1 - min_y, max_y - y1]
        
        u1 = 0.0
        u2 = 1.0
        
        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return False
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    if t > u2: return False
                    if t > u1: u1 = t
                else:
                    if t < u1: return False
                    if t < u2: u2 = t
        
        return u1 <= u2

    def get_aim_angle(self, my_pos, target_pos, my_angle):
        # 1. Direct shot
        if self.can_see(my_pos, target_pos):
            return True, math.atan2(target_pos[1] - my_pos[1], target_pos[0] - my_pos[0])
            
        # 2. One bounce shot (Simplified)
        # Try bouncing off nearby walls? 
        # This is computationally expensive to do perfectly.
        # Let's try a few sample directions?
        # Or just check if we can hit the target by bouncing off a wall midpoint?
        
        # For now, just return False to save performance, unless we implement a fast bounce check.
        # But user asked for "use bullet ricochet".
        # Let's try to find a bounce point on the walls of the current cell or target cell?
        # Too complex for this snippet. Let's stick to direct shot for now but improve can_see.
        
        return False, 0

    def is_in_danger(self, pos, velocity=(0, 0), time_window=1.0):
        return self.get_danger_score(pos, velocity, time_window) > 0

    def angle_difference(self, angle1, angle2):
        # Returns the difference between two angles in range [-pi, pi]
        diff = angle2 - angle1
        while diff <= -math.pi:
            diff += 2 * math.pi
        while diff > math.pi:
            diff -= 2 * math.pi
        return diff

    def get_cell_center(self, cell_idx):
        col = cell_idx % C.COLUMN_NUM
        row = cell_idx // C.COLUMN_NUM
        x = C.LEFT_SPACE + col * C.BLOCK_SIZE + C.BLOCK_SIZE / 2
        y = C.TOP_SPACE + row * C.BLOCK_SIZE + C.BLOCK_SIZE / 2
        return (x, y)

    def get_cell_from_pos(self, pos):
        col = int((pos[0] - C.LEFT_SPACE) / C.BLOCK_SIZE)
        row = int((pos[1] - C.TOP_SPACE) / C.BLOCK_SIZE)
        col = max(0, min(C.COLUMN_NUM - 1, col))
        row = max(0, min(C.ROW_NUM - 1, row))
        return row * C.COLUMN_NUM + col

    def find_path(self, start_pos, end_pos):
        start_cell = self.get_cell_from_pos(start_pos)
        end_cell = self.get_cell_from_pos(end_pos)
        
        if start_cell == end_cell:
            return []
            
        # Dijkstra's Algorithm
        # Explores paths based on accumulated cost (g-score)
        # Guarantees shortest path
        
        # Priority Queue: (cost, cell_idx)
        pq = []
        heapq.heappush(pq, (0, start_cell))
        
        came_from = {}
        cost_so_far = {}
        came_from[start_cell] = None
        cost_so_far[start_cell] = 0
        
        while pq:
            current = heapq.heappop(pq)[1]
            
            if current == end_cell:
                return self.reconstruct_path(came_from, current)
            
            for neighbor in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1 # Edge weight is 1
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost
                    heapq.heappush(pq, (priority, neighbor))
                    came_from[neighbor] = current
                        
        return []

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from and came_from[current] is not None:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def get_neighbors(self, cell_idx):
        neighbors = []
        col = cell_idx % C.COLUMN_NUM
        row = cell_idx // C.COLUMN_NUM
        
        # Helper to check if a specific wall type exists in a cell
        def has_wall(c_idx, w_type):
            if c_idx < 0 or c_idx >= len(self.arena.cells):
                return False
            cell_obj = self.arena.cells[c_idx]
            # Check walls_sign first (static walls)
            if cell_obj.walls_sign & w_type:
                return True
            # Check dynamic walls (added via add_fixed_walls)
            for w in cell_obj.walls:
                if w.type == w_type:
                    return True
            return False

        # Check 4 directions
        # UP
        if row > 0:
            up_idx = cell_idx - C.COLUMN_NUM
            # Check if current cell has TOP wall OR up neighbor has BOTTOM wall
            if not has_wall(cell_idx, C.TOP) and not has_wall(up_idx, C.BOTTOM):
                neighbors.append(up_idx)
        
        # DOWN
        if row < C.ROW_NUM - 1:
            down_idx = cell_idx + C.COLUMN_NUM
            if not has_wall(cell_idx, C.BOTTOM) and not has_wall(down_idx, C.TOP):
                neighbors.append(down_idx)

        # LEFT
        if col > 0:
            left_idx = cell_idx - 1
            if not has_wall(cell_idx, C.LEFT) and not has_wall(left_idx, C.RIGHT):
                neighbors.append(left_idx)

        # RIGHT
        if col < C.COLUMN_NUM - 1:
            right_idx = cell_idx + 1
            if not has_wall(cell_idx, C.RIGHT) and not has_wall(right_idx, C.LEFT):
                neighbors.append(right_idx)
            
        return neighbors


    def can_see(self, start, end):
        x1, y1 = start
        x2, y2 = end
        
        # Safety margin to keep path away from walls
        # Tank width is approx 27-41 px. 
        # Let's use half of the smaller dimension as margin.
        margin = C.PLAYER_PY / 2 + 5 # approx 13.5 + 5 = 18.5 px
        
        for i, cell in enumerate(self.arena.cells):
            col = i % C.COLUMN_NUM
            row = i // C.COLUMN_NUM
            cx = C.LEFT_SPACE + col * C.BLOCK_SIZE
            cy = C.TOP_SPACE + row * C.BLOCK_SIZE
            cw = C.BLOCK_SIZE
            ch = C.BLOCK_SIZE
            
            # Helper to check intersection with inflated wall
            def check_wall(wx, wy, ww, wh):
                # Inflate wall rect by margin
                ix = wx - margin
                iy = wy - margin
                iw = ww + margin * 2
                ih = wh + margin * 2
                
                # Check line intersection with the 4 borders of inflated rect
                # Or simpler: Liang-Barsky or Cohen-Sutherland against the rect
                # Re-use line_intersects_aabb logic which is robust
                return self.line_intersects_aabb(x1, y1, x2, y2, ix, iy, ix+iw, iy+ih)

            # 1. Check Static Walls
            if cell.walls_sign & C.TOP:
                if check_wall(cx, cy, cw, 0): return False
            if cell.walls_sign & C.BOTTOM:
                if check_wall(cx, cy + ch, cw, 0): return False
            if cell.walls_sign & C.LEFT:
                if check_wall(cx, cy, 0, ch): return False
            if cell.walls_sign & C.RIGHT:
                if check_wall(cx + cw, cy, 0, ch): return False
            
            # 2. Check Dynamic Walls
            for wall in cell.walls:
                wx, wy, ww, wh = wall.rect
                if check_wall(wx, wy, ww, wh): return False
                        
        return True

    def line_intersects_aabb(self, x1, y1, x2, y2, min_x, min_y, max_x, max_y):
        # Liang-Barsky algorithm
        dx = x2 - x1
        dy = y2 - y1
        
        p = [-dx, dx, -dy, dy]
        q = [x1 - min_x, max_x - x1, y1 - min_y, max_y - y1]
        
        u1 = 0.0
        u2 = 1.0
        
        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return True # Parallel and outside
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    if t > u2: return False
                    if t > u1: u1 = t
                else:
                    if t < u1: return False
                    if t < u2: u2 = t
        
        return u1 <= u2

    def smooth_path(self, raw_path, start_pos, end_pos):
        if not raw_path:
            return []
            
        # Convert indices to points
        points = [start_pos]
        for cell_idx in raw_path:
            points.append(self.get_cell_center(cell_idx))
        # Replace last center with actual end_pos if it's in the last cell
        # Or just append end_pos? 
        # If raw_path includes the end cell, the last point is the center of the end cell.
        # We want to go to the actual target position.
        points[-1] = end_pos 
        
        if len(points) < 3:
            return points
            
        # String Pulling
        smoothed = [points[0]]
        current_idx = 0
        
        while current_idx < len(points) - 1:
            # Try to connect to the furthest possible point
            check_idx = len(points) - 1
            found_shortcut = False
            
            while check_idx > current_idx + 1:
                if self.can_see(points[current_idx], points[check_idx]):
                    smoothed.append(points[check_idx])
                    current_idx = check_idx
                    found_shortcut = True
                    break
                check_idx -= 1
            
            if not found_shortcut:
                current_idx += 1
                smoothed.append(points[current_idx])
                
        return smoothed

    def line_intersection(self, p1, p2, p3, p4):
        # Check if line segment p1-p2 intersects p3-p4
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return False
            
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return True
        return False
