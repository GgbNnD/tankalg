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
        # Update logic
        self._update()
        
        # Return actions (fire)
        # Movement is handled by setting self.player.throttle/steering directly
        fire = False
        if self.last_attack_action and self.last_attack_action.get('fire'):
            fire = True
            
        actions = {'fire': fire}
        return actions

    def _update(self):
        current_time = pygame.time.get_ticks()
        
        # 0. Global Prediction (The Oracle)
        # Get ALL future bullet paths once per frame to inform all decisions
        all_trajectories = self.dodge_strategy.get_all_bullet_trajectories()
        
        # 1. Dodge (Highest Priority - Survival)
        dodge_action = self.dodge_strategy.get_dodge_action(all_trajectories=all_trajectories)
        if dodge_action:
            self.player.steering = dodge_action['steering']
            self.player.throttle = dodge_action['throttle']
            self.last_attack_action = None # Cancel attack if dodging
            return

        # 2. Find target
        self.target_enemy = self._find_nearest_enemy()
        
        if not self.target_enemy:
            self.player.throttle = 0
            self.player.steering = 0
            self.last_attack_action = None
            return

        # 3. Try Attack Strategy (Kill)
        attack_action = self.attack_strategy.get_attack_action(self.target_enemy)
        self.last_attack_action = attack_action
        
        if attack_action:
            # SAFETY VALVE: Check if this attack move is safe
            # Even if we want to kill, we must not commit suicide
            if self.dodge_strategy.is_action_safe(attack_action['steering'], attack_action['throttle'], all_trajectories):
                # Execute Attack Action
                self.player.steering = attack_action['steering']
                self.player.throttle = attack_action['throttle']
                # Fire is handled in get_keys
                return
            else:
                # Attack move is unsafe. Abort attack movement.
                # We might still be able to fire if we stop? 
                # For now, just fall through to see if Contact strategy finds a safe path, 
                # or just stop.
                self.last_attack_action = None # Don't fire if we can't safely take the shot position

        # 4. If no safe attack possible, Move towards enemy (Chase)
        
        # Update path periodically
        if current_time - self.update_timer > self.path_update_interval:
            # Convert positions to screen coordinates (pixels) for Dijkstra
            start_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
            end_pos = (self.target_enemy.x / C.MOTION_CALC_SCALE, self.target_enemy.y / C.MOTION_CALC_SCALE)
            
            self.path = self.pathfinder.get_path(start_pos, end_pos)
            self.contact_strategy.set_path(self.path)
            self.update_timer = current_time
            
        # Execute Contact Strategy
        move_action = self.contact_strategy.execute()
        
        # SAFETY VALVE: Check if this chase move is safe
        if self.dodge_strategy.is_action_safe(move_action['steering'], move_action['throttle'], all_trajectories):
            self.player.steering = move_action['steering']
            self.player.throttle = move_action['throttle']
        else:
            # Chase move is unsafe! Stop immediately.
            # Better to stand still than walk into a bullet.
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
        
        # Angle difference
        diff = desired_angle - self.player.theta
        
        # Normalize to [-PI, PI]
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        
        # Steering
        if abs(diff) > 0.1:
            if diff > 0:
                self.player.steering = 1.0 
            else:
                self.player.steering = -1.0
        else:
            self.player.steering = 0.0
            
        # Throttle
        # Slow down if turning sharply
        if abs(diff) > 1.0:
            self.player.throttle = 0.0 # Turn in place
        else:
            self.player.throttle = 1.0
