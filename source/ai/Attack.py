import math
import pygame
from .. import constants as C
from ..parts import cell

class AttackStrategy:
    def __init__(self, player):
        self.player = player
        self.arena = player.arena
        # 攻击范围
        self.max_range = 500 
        self.ricochet_range = 300
        
    def get_attack_action(self, enemy):
        """
        判断 AI 是否应当攻击敌人。
        若执行攻击，返回包含动作细节的字典，否则返回 None。
        """
        # 1. 距离检测
        # 将玩家与敌人的位置转换为屏幕坐标以便计算
        p_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        e_pos = (enemy.x / C.MOTION_CALC_SCALE, enemy.y / C.MOTION_CALC_SCALE)
        
        dist = math.hypot(e_pos[0] - p_pos[0], e_pos[1] - p_pos[1])
        if dist > self.max_range:
            return None
            
        # 2. 子弹管理
        # 考虑墙体遮挡
        has_los = self.check_path_clear(p_pos, e_pos)
        active_bullets = len(self.player.rounds)
        
        # 限制弹药
        # 距离大于反弹射程，或者没有直接视线时停止攻击 
        if dist > self.ricochet_range or not has_los:
            if active_bullets >= 5:
                return None 
            
        # 3. 尝试瞄准
        aim_angle = self.try_aiming(p_pos, e_pos, dist)
        
        if aim_angle is not None:
            # 计算转向以对准目标角度
            current_theta = self.player.theta
            diff = aim_angle - current_theta
            
            while diff > math.pi: diff -= 2*math.pi
            while diff < -math.pi: diff += 2*math.pi
            
            action = {
                'aiming': True,
                'fire': False,
                'steering': 0,
                'throttle': 0
            }
            
            # 3. 执行旋转并射击
            # 若角度对准在约 4 度内则开火
            if abs(diff) < 0.07:
                action['fire'] = True
                action['steering'] = 0
            else:
                # 向目标方向旋转
                action['steering'] = 1.0 if diff > 0 else -1.0
                
            return action
            
        return None

    def try_aiming(self, p_pos, e_pos, dist):
        # 直接射击
        dx = e_pos[0] - p_pos[0]
        dy = e_pos[1] - p_pos[1]
        direct_angle = math.atan2(dy, dx)
        
        if self.check_path_clear(p_pos, e_pos):
            return direct_angle
            
        # 反弹射击，仅在有效反弹范围内尝试
        if dist > self.ricochet_range:
            return None

        # 按步长扫描 360 度
        # 跳过正交方向以避免无限反弹或子弹陷入死循环
        for i in range(0, 36):
            angle = i * 10 * math.pi / 180
            
            # 检查角度是否接近基本方向（0、90、180、270）
            deg = (angle * 180 / math.pi) % 360
            if any(abs(deg - x) < 5 for x in [0, 90, 180, 270]):
                continue
                
            if self.simulate_shot(p_pos, angle, e_pos):
                return angle
                
        return None

    def check_path_clear(self, start, end):
        # 检查线段是否与任何墙相交
        for c in self.arena.cells:
            for w in c.walls:
                if w.rect.clipline(start, end):
                    return False
        return True

    def simulate_shot(self, start_pos, angle, target_pos):
        # 射线检测并考虑反弹
        max_bounces = 1
        current_pos = start_pos
        current_angle = angle
        
        target_radius = C.PLAYER_PX / 2
        
        for _ in range(max_bounces + 1):
            ray_len = 1000
            end_x = current_pos[0] + ray_len * math.cos(current_angle)
            end_y = current_pos[1] + ray_len * math.sin(current_angle)
            
            closest_hit = None
            min_dist = float('inf')
            hit_wall = None
            
            # 寻找最近的墙体相交点
            for c in self.arena.cells:
                for w in c.walls:
                    clipped = w.rect.clipline(current_pos, (end_x, end_y))
                    if clipped:
                        # 返回位于矩形内部的两点 ((x1, y1), (x2, y2))
                        p1, p2 = clipped
                        d1 = math.hypot(p1[0]-current_pos[0], p1[1]-current_pos[1])
                        d2 = math.hypot(p2[0]-current_pos[0], p2[1]-current_pos[1])
                        
                        # 忽略与起点非常接近的相交
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
            
            # 检查该线段上是否与敌人相交
            segment_end = closest_hit if closest_hit else (end_x, end_y)
            
            if self.line_intersects_circle(current_pos, segment_end, target_pos, target_radius):
                return True
                
            if closest_hit and hit_wall:
                # 反弹逻辑
                wx, wy, ww, wh = hit_wall.rect
                hx, hy = closest_hit
                
                # 根据碰撞点在墙体边界上的位置确定法线方向
                eps = 2.0
                normal_x, normal_y = 0, 0
                
                if abs(hx - wx) < eps: normal_x = -1       # Left face
                elif abs(hx - (wx+ww)) < eps: normal_x = 1 # Right face
                elif abs(hy - wy) < eps: normal_y = -1     # Top face
                elif abs(hy - (wy+wh)) < eps: normal_y = 1 # Bottom face
                else:
                    # 回退，若击中角点或数值精度问题，则反转速度向量
                    normal_x, normal_y = -math.cos(current_angle), -math.sin(current_angle)

                # 反射向量公式：R = V - 2(V·N)N
                vx = math.cos(current_angle)
                vy = math.sin(current_angle)
                
                dot = vx * normal_x + vy * normal_y
                rx = vx - 2 * dot * normal_x
                ry = vy - 2 * dot * normal_y
                
                current_angle = math.atan2(ry, rx)
                current_pos = closest_hit
                
                # 稍微将位置移离墙体，避免立刻再次碰撞
                current_pos = (current_pos[0] + rx*2, current_pos[1] + ry*2)
            else:
                # 在射线长度范围内未命中墙体
                break
                
        return False

    def line_intersects_circle(self, p1, p2, center, radius):
        # 检查线段 p1-p2 是否与圆相交
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
