import math
import pygame
from .. import constants as C

class ContactStrategy:
    def __init__(self, player):
        self.player = player
        self.arena = player.arena
        self.path = []
        self.next_idx = 0
        self.stuck_steps = 0
        self.prev_pos = (0, 0)
        self.prev_theta = 0
        
    def set_path(self, path):
        """
        为策略设置一条新的路径。
        path: 屏幕坐标（像素）的 (x, y) 元组列表。
        """
        self.path = path
        self.next_idx = 0
        self.stuck_steps = 0
        
        # 优化：路径的第一个点通常是玩家当前所在格子的中心。
        # 我们应跳过它以避免回溯，除非它是唯一的目标点。
        if len(self.path) > 1:
            self.next_idx = 1
            
            # 进一步优化：可以检查我们是否已经更接近第二个点（索引 2），
            # 或第二个点（索引 1）是否在我们身后。
            # 目前最重要的是跳过当前格子中心（索引 0）。

    def execute(self):
        """
        执行当前帧的接触/追踪策略。
        返回包含 'steering' 和 'throttle' 的字典。
        """
        if not self.path or self.next_idx >= len(self.path):
            return {'steering': 0, 'throttle': 0}
            
        target_pos = self.path[self.next_idx]
        # 将玩家位置转换为像素以便与路径（像素）比较
        current_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        
        # A. 路点管理
        dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
        if dist < 25: # 到达当前路点（稍微增加容差半径）
            self.next_idx += 1
            if self.next_idx >= len(self.path):
                return {'steering': 0, 'throttle': 0}
            target_pos = self.path[self.next_idx]
            
        # B. 转向与油门控制
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        desired_angle = math.atan2(dy, dx)
        
        diff = desired_angle - self.player.theta
        # 归一化角差到 [-PI, PI]
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        
        steering = 0.0
        throttle = 0.0
        
        # 前进逻辑：当角度误差小于约 45 度（约 0.78 弧度）时前进
        if abs(diff) < 0.78:
            throttle = 1.0
        else:
            throttle = 0.0 # 角度过大则停止前进以便原地转向
            
        # 旋转逻辑：当角度误差 >= 约 12 度（约 0.2 弧度）时转向
        if abs(diff) >= 0.2:
            if diff > 0:
                steering = 1.0
            else:
                steering = -1.0
                
        # C. 防卡住与碰撞处理
        # 检测是否被卡住（试图移动但位置没有变化）
        # 使用小的 epsilon 进行浮点比较
        # 使用未缩放坐标进行卡住检测（更高精度）
        raw_pos = (self.player.x, self.player.y)
        pos_diff = math.hypot(raw_pos[0] - self.prev_pos[0], raw_pos[1] - self.prev_pos[1])
        
        if throttle > 0 and pos_diff < 0.5: # Trying to move but stuck
            self.stuck_steps += 1
            
            # "滑动"逻辑：如果被墙卡住，尝试旋转
            # 这是对 C++ 逻辑的简化版本
            # 如果被卡住，尝试从墙侧旋转或左右摆动
            if self.stuck_steps <= 5:
                 # Try to rotate in the direction we want to go, even if we are stuck moving forward
                 # This helps sliding along walls
                 if steering == 0:
                     # If we were going straight, pick a direction (maybe based on previous rotation or random)
                     steering = 1.0 
            
            if self.stuck_steps > 5: # Stuck for too long
                # 长时间卡住：倒退
                throttle = -1.0
                steering = 0.0 # 直线后退
                
                # 如果成功后退（在后续帧位移变化），会重置 stuck_steps
                # 目前暂时强制倒退几帧；C++ 逻辑在位置改变后会重置计数。
                
        else:
            self.stuck_steps = 0
            
        self.prev_pos = raw_pos
        self.prev_theta = self.player.theta
        
        return {'steering': steering, 'throttle': throttle}
