import math
import pygame
from .. import constants as C
from ..parts import cell

class AttackStrategy:
    def __init__(self, player):
        self.player = player
        self.arena = player.arena
        # Attack range in pixels (screen coordinates)
        # Reduced to encourage movement/pathfinding when far away
        self.max_range = 500 
        self.ricochet_range = 300
        
    def get_attack_action(self, enemy):
        """
        Determines if the AI should attack the enemy.
        Returns a dictionary with action details if attacking, else None.
        """
        # 1. Check Range
        # Convert player/enemy positions to screen coordinates for calculation
        p_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        e_pos = (enemy.x / C.MOTION_CALC_SCALE, enemy.y / C.MOTION_CALC_SCALE)
        
        dist = math.hypot(e_pos[0] - p_pos[0], e_pos[1] - p_pos[1])
        if dist > self.max_range:
            return None
            
        # 2. Bullet Management (Ammo Conservation)
        # Consider walls: if there is no direct line of sight, we are effectively "far away"
        # or in a complex situation where spamming is dangerous/wasteful.
        has_los = self.check_path_clear(p_pos, e_pos)
        active_bullets = len(self.player.rounds)
        
        # Limit ammo if:
        # 1. Distance is greater than ricochet range
        # 2. OR there is no direct line of sight (wall in between)
        if dist > self.ricochet_range or not has_los:
            if active_bullets >= 5:
                return None # Stop attacking, move closer (Chase)
            
        # 3. Try Aiming
        aim_angle = self.try_aiming(p_pos, e_pos, dist)
        
        if aim_angle is not None:
            # Calculate steering to align with aim_angle
            current_theta = self.player.theta
            diff = aim_angle - current_theta
            
            # Normalize diff to [-PI, PI]
            while diff > math.pi: diff -= 2*math.pi
            while diff < -math.pi: diff += 2*math.pi
            
            action = {
                'aiming': True,
                'fire': False,
                'steering': 0,
                'throttle': 0
            }
            
            # 3. Execution: Rotate and Fire
            # If aligned within ~4 degrees (0.07 rad), fire
            if abs(diff) < 0.07:
                action['fire'] = True
                action['steering'] = 0
            else:
                # Rotate towards target
                action['steering'] = 1.0 if diff > 0 else -1.0
                
            return action
            
        return None

    def try_aiming(self, p_pos, e_pos, dist):
        # A. Direct Shot
        dx = e_pos[0] - p_pos[0]
        dy = e_pos[1] - p_pos[1]
        direct_angle = math.atan2(dy, dx)
        
        if self.check_path_clear(p_pos, e_pos):
            return direct_angle
            
        # B. Ricochet Shot
        # Only try ricochet if within effective range
        if dist > self.ricochet_range:
            return None

        # Scan 360 degrees in steps
        # Skip cardinal directions to avoid infinite bouncing loops or stuck bullets
        for i in range(0, 36):
            angle = i * 10 * math.pi / 180
            
            # Check if angle is close to cardinal directions (0, 90, 180, 270)
            deg = (angle * 180 / math.pi) % 360
            if any(abs(deg - x) < 5 for x in [0, 90, 180, 270]):
                continue
                
            if self.simulate_shot(p_pos, angle, e_pos):
                return angle
                
        return None

    def check_path_clear(self, start, end):
        # Check if line segment intersects any wall
        for c in self.arena.cells:
            for w in c.walls:
                if w.rect.clipline(start, end):
                    return False
        return True

    def simulate_shot(self, start_pos, angle, target_pos):
        # Raycast with bounces
        # Max bounces = 1 for efficiency (can be increased)
        max_bounces = 1
        current_pos = start_pos
        current_angle = angle
        
        # Target radius (approximate)
        target_radius = C.PLAYER_PX / 2
        
        for _ in range(max_bounces + 1):
            # Create a long ray
            ray_len = 1000
            end_x = current_pos[0] + ray_len * math.cos(current_angle)
            end_y = current_pos[1] + ray_len * math.sin(current_angle)
            
            closest_hit = None
            min_dist = float('inf')
            hit_wall = None
            
            # Find closest wall intersection
            for c in self.arena.cells:
                for w in c.walls:
                    clipped = w.rect.clipline(current_pos, (end_x, end_y))
                    if clipped:
                        # clipline returns ((x1, y1), (x2, y2)) inside the rect
                        p1, p2 = clipped
                        d1 = math.hypot(p1[0]-current_pos[0], p1[1]-current_pos[1])
                        d2 = math.hypot(p2[0]-current_pos[0], p2[1]-current_pos[1])
                        
                        # Ignore intersections very close to start (self-intersection after bounce)
                        dist = 0
                        pt = None
                        if d1 > 1:
                            dist = d1
                            pt = p1
                        elif d2 > 1:
                            dist = d2
                            pt = p2
                        else:
                            continue
                            
                        if dist < min_dist:
                            min_dist = dist
                            closest_hit = pt
                            hit_wall = w
            
            # Check Enemy Intersection along this segment
            segment_end = closest_hit if closest_hit else (end_x, end_y)
            
            if self.line_intersects_circle(current_pos, segment_end, target_pos, target_radius):
                return True
                
            if closest_hit and hit_wall:
                # Bounce Logic
                wx, wy, ww, wh = hit_wall.rect
                hx, hy = closest_hit
                
                # Determine normal based on hit position relative to wall bounds
                eps = 2.0
                normal_x, normal_y = 0, 0
                
                if abs(hx - wx) < eps: normal_x = -1       # Left face
                elif abs(hx - (wx+ww)) < eps: normal_x = 1 # Right face
                elif abs(hy - wy) < eps: normal_y = -1     # Top face
                elif abs(hy - (wy+wh)) < eps: normal_y = 1 # Bottom face
                else:
                    # Fallback if corner hit or precision issue: reverse velocity
                    normal_x, normal_y = -math.cos(current_angle), -math.sin(current_angle)

                # Reflect vector: R = V - 2(V.N)N
                vx = math.cos(current_angle)
                vy = math.sin(current_angle)
                
                dot = vx * normal_x + vy * normal_y
                rx = vx - 2 * dot * normal_x
                ry = vy - 2 * dot * normal_y
                
                current_angle = math.atan2(ry, rx)
                current_pos = closest_hit
                
                # Move slightly off wall to avoid immediate re-collision
                current_pos = (current_pos[0] + rx*2, current_pos[1] + ry*2)
            else:
                # No wall hit within ray length
                break
                
        return False

    def line_intersects_circle(self, p1, p2, center, radius):
        # Check if line segment p1-p2 intersects circle
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
