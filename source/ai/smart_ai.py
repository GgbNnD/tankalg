import math
import pygame
from .. import constants as C
from .Dijkstra import Dijkstra
from .Attack import AttackStrategy
from .Contact import ContactStrategy
from .Dodge import DodgeStrategy

class SmartAI:
    def __init__(self, player):
        self.player = player
        self.arena = player.arena
        self.pathfinder = Dijkstra(self.arena)
        self.attack_strategy = AttackStrategy(player)
        self.contact_strategy = ContactStrategy(player)
        self.dodge_strategy = DodgeStrategy(player)
        self.path = []
        self.target_enemy = None
        self.update_timer = 0
        self.path_update_interval = 500 # ms
        self.last_attack_action = None

    def get_keys(self):
        # 更新逻辑
        self._update()
        
        # 返回动作（射击）
        # 移动通过直接设置 self.player.throttle/steering 来处理
        fire = False
        if self.last_attack_action and self.last_attack_action.get('fire'):
            fire = True
            
        actions = {'fire': fire}
        return actions

    def _update(self):
        current_time = pygame.time.get_ticks()
        
        # 0. 全局预测（先知）
        # 每帧获取所有未来子弹轨迹以为各决策提供信息
        all_trajectories = self.dodge_strategy.get_all_bullet_trajectories()
        
        # 1. 闪避（最高优先级是生存）
        dodge_action = self.dodge_strategy.get_dodge_action(all_trajectories=all_trajectories)
        if dodge_action:
            self.player.steering = dodge_action['steering']
            self.player.throttle = dodge_action['throttle']
            self.last_attack_action = None # 闪避时取消攻击动作
            return

        # 2. 寻找目标
        self.target_enemy = self._find_nearest_enemy()
        
        if not self.target_enemy:
            self.player.throttle = 0
            self.player.steering = 0
            self.last_attack_action = None
            return

        # 3. 尝试攻击策略
        attack_action = self.attack_strategy.get_attack_action(self.target_enemy)
        self.last_attack_action = attack_action
        
        if attack_action:
            # 检查此攻击动作是否安全，即使想击杀也不能送死
            if self.dodge_strategy.is_action_safe(attack_action['steering'], attack_action['throttle'], all_trajectories):
                # 安全，执行攻击动作
                self.player.steering = attack_action['steering']
                self.player.throttle = attack_action['throttle']
                # 开火在 get_keys 中处理
                return
            else:
                # 攻击动作不安全，放弃攻击移动。
                # 我们可能在停止后仍能开火，但目前让其回落到接触策略或停下。
                self.last_attack_action = None # 如果无法安全获得射击位置则不射击

        # 4. 如果没有安全的攻击选项，则向敌人移动
        
        # 定期更新路径
        if current_time - self.update_timer > self.path_update_interval:
            # 计算新路径
            start_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
            end_pos = (self.target_enemy.x / C.MOTION_CALC_SCALE, self.target_enemy.y / C.MOTION_CALC_SCALE)
            
            self.path = self.pathfinder.get_path(start_pos, end_pos)
            self.contact_strategy.set_path(self.path)
            self.update_timer = current_time
            
        # 执行寻路策略
        move_action = self.contact_strategy.execute()
        
        # 安全阀：检查此追逐动作是否安全
        if self.dodge_strategy.is_action_safe(move_action['steering'], move_action['throttle'], all_trajectories):
            self.player.steering = move_action['steering']
            self.player.throttle = move_action['throttle']
        else:
            # 追逐动作不安全，立即停止
            self.player.steering = 0
            self.player.throttle = 0

    def _find_nearest_enemy(self):
        nearest = None
        min_dist = float('inf')
        for p in self.arena.players:
            if p != self.player and not p.dead:
                dist = math.hypot(p.x - self.player.x, p.y - self.player.y)
                if dist < min_dist:
                    min_dist = dist
                    nearest = p
        return nearest

    def _steer_towards(self, target_pos):
        dx = target_pos[0] - self.player.x
        dy = target_pos[1] - self.player.y
        
        desired_angle = math.atan2(dy, dx)
        
        # 角度差
        diff = desired_angle - self.player.theta
        
        # 归一化到 [-PI, PI]
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        
        # 转向
        if abs(diff) > 0.1:
            if diff > 0:
                self.player.steering = 1.0 
            else:
                self.player.steering = -1.0
        else:
            self.player.steering = 0.0
            
        # 油门
        # 如果大角度转弯则减速
        if abs(diff) > 1.0:
            self.player.throttle = 0.0 # Turn in place
        else:
            self.player.throttle = 1.0
