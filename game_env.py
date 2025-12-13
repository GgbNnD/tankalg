import pygame
import numpy as np
import math
from mymap import *
from mycomp import *
from random import randint, random
import bullet, player, propreties
import os

# Set dummy video driver for headless training if needed
# os.environ["SDL_VIDEODRIVER"] = "dummy"

class TankGame:
    def __init__(self, render_mode=False, num_players=2):
        self.render_mode = render_mode
        self.num_players = num_players
        pygame.init()
        
        self.fps = propreties.data["fps"]
        self.timer = pygame.time.Clock()
        self.width = propreties.data["maxcols"] * SQSIZE + WALLWIDTH
        self.height = (propreties.data["maxrows"] + 1) * SQSIZE + WALLWIDTH
        
        if self.render_mode:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = pygame.Surface((self.width, self.height)) # Off-screen surface

        self.gene = make_generator()
        # Initialize players with random colors
        self.players = []
        for _ in range(num_players):
            color = (randint(50, 255), randint(50, 255), randint(50, 255))
            self.players.append(player.Player(color))
        
        # Initialize static enemy
        self.enemy = player.Player((255, 0, 0)) # Red enemy
            
        self.effect_bullets = []
        self.game_map = None
        self.count = 0
        # Limit game length to 60 seconds (60 * fps)
        self.max_steps = 100 * self.fps
        self.current_step = 0
        
                # Generate fixed map for training
        cols = 6
        rows = 4
        startx = (propreties.data["maxcols"] - cols) * SQSIZE // 2
        starty = (propreties.data["maxrows"] - rows) * SQSIZE // 2
        
        # PvP Map: Two vertical walls in the middle
        walls = []
        # Boundaries
        for c in range(cols):
            walls.append((c, 0, -1))      # Top
            walls.append((c, rows, -1))   # Bottom
        for r in range(rows):
            walls.append((0, r, 1))       # Left
            walls.append((cols, r, 1))    # Right
            
        # Two vertical walls
        # Left wall (x=2) from top (y=0) down to y=3 (length 3)
        for r in range(3):
            walls.append((2, r, 1))
            
        # Right wall (x=4) from bottom (y=4) up to y=1 (length 3)
        for r in range(1, 4):
            walls.append((4, r, 1))

        self.fixed_map_data = (walls, cols, rows, startx, starty)
        self.finished_players = []

    def reset(self):
        self.effect_bullets.clear()
        self.finished_players = [False] * self.num_players
        walls, cols, rows, startx, starty = self.fixed_map_data
        self.game_map = Map(walls, self.screen, cols, rows, startx, starty)
        
        # Reset players
        # P1: Top-Left (0, 0)
        # P2: Bottom-Right (5, 3)
        spawn_positions = [
            complex((0)*SQSIZE+70+startx, (0)*SQSIZE+70+starty),
            complex((5)*SQSIZE+70+startx, (3)*SQSIZE+70+starty)
        ]
        
        for i, ply in enumerate(self.players):
            if i < len(spawn_positions):
                pos = spawn_positions[i]
            else:
                # Fallback for more players
                pos = complex(randint(0, cols-1)*SQSIZE+70+startx, randint(0, rows-1)*SQSIZE+70+starty)
            
            angle = random()*2*pi
            ply.newgame(pos, angle)
        
        self.current_step = 0
        return [self.get_state(i) for i in range(self.num_players)]

    def step(self, actions):
        # Actions: List of [move, turn, shoot] for each player
        
        rewards = [0] * self.num_players
        infos = [{"hit": 0, "suicide": 0, "dead": 0, "win": 0} for _ in range(self.num_players)]
        
        for i, ply in enumerate(self.players):
            if self.finished_players[i]: continue # Skip finished players
            if ply.lives <= 0: continue
            
            act = actions[i]
            
            # Movement
            if act[0] > 0.3: ply.move_mode = player.Player.FORWARD
            elif act[0] < -0.3: ply.move_mode = player.Player.BACK
            else: ply.move_mode = 0
            
            # Turning
            if act[1] > 0.3: ply.turn_mode = player.Player.RIGHT
            elif act[1] < -0.3: ply.turn_mode = player.Player.LEFT
            else: ply.turn_mode = 0
            
            # Shooting
            if act[2] > 0.5:
                newb = ply.add_bullet()
                if newb:
                    self.effect_bullets.append(newb)
                    rewards[i] -= 0.002 # Very small penalty for shooting
        
        # Update bullets
        for b in self.effect_bullets.copy():
            if b.is_effect():
                # Check collision with ALL players
                # We pass all players to update, but we need to know who got hit
                # bullet.update modifies lives.
                
                # We need to check each player individually to assign rewards correctly
                # But bullet.update handles bounce and hit logic together.
                # Let's assume bullet.update returns True if it hit something? No.
                
                # We can check lives before and after.
                lives_before = [p.lives for p in self.players]
                
                # Pass all active players as targets
                targets = [p for p in self.players if p.lives > 0]
                b.update(self.game_map, tuple(targets))
                
                hit_occurred = False
                for i, ply in enumerate(self.players):
                    if ply.lives < lives_before[i]:
                        # Player i was hit
                        hit_occurred = True
                        self.finished_players[i] = True
                        rewards[i] -= 1.0 # Penalty for dying
                        infos[i]['dead'] = 1
                        
                        # Reward the shooter
                        if b.owner in self.players:
                            shooter_idx = self.players.index(b.owner)
                            if shooter_idx != i:
                                rewards[shooter_idx] += 1.0 # Reward for kill
                                infos[shooter_idx]['hit'] = 1
                            else:
                                rewards[shooter_idx] -= 0.5 # Penalty for suicide
                                infos[shooter_idx]['suicide'] = 1
                        break # Bullet removed after one hit usually
                
                if hit_occurred:
                    if b in self.effect_bullets:
                        self.effect_bullets.remove(b)
            else:
                self.effect_bullets.remove(b)

        # Update players
        active_count = 0
        for i, ply in enumerate(self.players):
            if not self.finished_players[i] and ply.lives > 0:
                active_count += 1
                ply.update(self.game_map)
                
                # Time penalty to encourage action
                rewards[i] -= 0.001
        
        self.current_step += 1
        
        # Done if 0 or 1 player left (Last Man Standing)
        done = (active_count <= 1) or self.current_step >= self.max_steps
        
        if done and active_count == 1:
             # Find the winner
             for i, ply in enumerate(self.players):
                 if ply.lives > 0:
                     infos[i]['win'] = 1
        
        return [self.get_state(i) for i in range(self.num_players)], rewards, done, infos

    def render(self):
        if not self.render_mode: return
        
        self.screen.fill((255, 255, 255))
        
        for b in self.effect_bullets:
            b.draw(self.screen)
            
        for ply in self.players:
            if ply.lives > 0:
                ply.draw(self.screen)
                
        self.game_map.draw()
        pygame.display.update()

    def get_state(self, player_idx):
        # If finished, return zeros
        if self.finished_players[player_idx]:
            return np.zeros(20)

        # Construct feature vector
        me = self.players[player_idx]
        
        if me.lives <= 0:
            return np.zeros(20) # Dead state
            
        # Normalize positions
        norm_x = self.width
        norm_y = self.height
        
        # 1. Self State (4)
        state = [
            me.position.real / norm_x,
            me.position.imag / norm_y,
            math.sin(me.angle),
            math.cos(me.angle)
        ]
        
        # 2. Enemy State (4) - Find nearest opponent
        # For 1v1, it's just the other player
        enemy = None
        min_dist = float('inf')
        
        for i, p in enumerate(self.players):
            if i != player_idx and p.lives > 0:
                dist = abs(p.position - me.position)
                if dist < min_dist:
                    min_dist = dist
                    enemy = p
        
        if enemy:
            rel_pos = enemy.position - me.position
            state.extend([
                rel_pos.real / norm_x,
                rel_pos.imag / norm_y,
                math.sin(enemy.angle),
                math.cos(enemy.angle)
            ])
        else:
            # No enemy alive
            state.extend([0, 0, 0, 0])
            
        # 3. Bullet State (4) - Nearest bullet
        # Find nearest bullet (that is not mine? or any bullet?)
        # Any bullet is dangerous.
        nearest_b = None
        min_b_dist = float('inf')
        
        for b in self.effect_bullets:
            # Only consider bullets that are NOT mine
            if b.owner == me:
                continue
                
            dist = abs(b.position - me.position)
            if dist < min_b_dist:
                min_b_dist = dist
                nearest_b = b
                
        if nearest_b:
            rel_b_pos = nearest_b.position - me.position
            # Velocity is speed * complex(cos, sin)
            # Bullet speed is constant usually.
            # Let's just use relative position and maybe velocity vector
            vx = bullet.Bullet.SPEED * math.cos(nearest_b.angle)
            vy = bullet.Bullet.SPEED * math.sin(nearest_b.angle)
            
            state.extend([
                rel_b_pos.real / norm_x,
                rel_b_pos.imag / norm_y,
                vx / bullet.Bullet.SPEED, # Normalize direction
                vy / bullet.Bullet.SPEED
            ])
        else:
            state.extend([0, 0, 0, 0])
        
        # 4. Wall Raycasts (8 directions) (8)
        rays = self.cast_rays(me.position, me.angle)
        state.extend(rays)
        
        return np.array(state)

    def cast_rays(self, pos, angle, num_rays=8, max_dist=500):
        distances = []
        for i in range(num_rays):
            ray_angle = angle + (i * 2 * math.pi / num_rays)
            ray_dir = complex(math.cos(ray_angle), math.sin(ray_angle))
            
            # Simple raymarching or line intersection
            # Since we have walls as lines, we can do line intersection
            min_dist = max_dist
            
            # Ray line segment
            ray_end = pos + ray_dir * max_dist
            
            # Let's try a different approach for Raycasting:
            # Iterate over all walls in the map and find intersection.
            # Map.walls contains (c, r, type).
            
            closest_dist = max_dist
            
            # Add map boundaries
            # Left, Right, Top, Bottom
            bounds = [
                ((self.game_map.startx, self.game_map.starty), (self.game_map.startx, self.height)), # Left
                ((self.width, self.game_map.starty), (self.width, self.height)), # Right
                ((self.game_map.startx, self.game_map.starty), (self.width, self.game_map.starty)), # Top
                ((self.game_map.startx, self.height), (self.width, self.height)) # Bottom
            ]
            
            all_lines = []
            # Convert map walls to lines
            for w in self.game_map.walls:
                c, r, t = w
                x = c * SQSIZE + self.game_map.startx
                y = r * SQSIZE + self.game_map.starty
                if t == -1: # Horizontal
                    all_lines.append(((x, y), (x + SQSIZE + WALLWIDTH, y)))
                elif t == 1: # Vertical
                    all_lines.append(((x, y), (x, y + SQSIZE + WALLWIDTH)))
            
            all_lines.extend(bounds)
            
            # Ray as a line
            x1, y1 = pos.real, pos.imag
            x2, y2 = ray_end.real, ray_end.imag
            
            for line in all_lines:
                x3, y3 = line[0]
                x4, y4 = line[1]
                
                # Line intersection formula
                denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
                if denom == 0: continue
                
                ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
                ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
                
                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    # Intersection
                    int_x = x1 + ua * (x2 - x1)
                    int_y = y1 + ua * (y2 - y1)
                    d = math.sqrt((int_x - x1)**2 + (int_y - y1)**2)
                    if d < closest_dist:
                        closest_dist = d
            
            distances.append(closest_dist / max_dist)
            
        return distances

