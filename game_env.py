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
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
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
        self.p1 = player.Player((255, 0, 0))
        self.p2 = player.Player((0, 255, 0))
        self.players = [self.p1, self.p2]
        self.effect_bullets = []
        self.game_map = None
        self.count = 0
        self.max_steps = 2000 # Limit game length
        self.current_step = 0

    def reset(self):
        self.effect_bullets.clear()
        cols = randint(propreties.data["mincols"], propreties.data["maxcols"])
        rows = randint(propreties.data["minrows"], propreties.data["maxrows"])
        startx = (propreties.data["maxcols"] - cols) * SQSIZE // 2
        starty = (propreties.data["maxrows"] - rows) * SQSIZE // 2
        
        walls = self.gene.send((cols, rows))
        self.game_map = Map(walls, self.screen, cols, rows, startx, starty)
        
        # Reset players
        # Ensure they don't spawn too close? The original code just randomizes.
        self.p1.newgame(complex(randint(0, cols-1)*SQSIZE+70+startx, randint(0, rows-1)*SQSIZE+70+starty), random()*2*pi)
        self.p2.newgame(complex(randint(0, cols-1)*SQSIZE+70+startx, randint(0, rows-1)*SQSIZE+70+starty), random()*2*pi)
        
        self.current_step = 0
        return self.get_state(0), self.get_state(1)

    def step(self, action1, action2):
        # Actions: [move, turn, shoot]
        # move: -1 to 1 (back/forward)
        # turn: -1 to 1 (left/right)
        # shoot: > 0.5 shoot
        
        actions = [action1, action2]
        rewards = [0, 0]
        
        for i, ply in enumerate(self.players):
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
                    rewards[i] -= 0.1 # Small penalty for shooting to prevent spamming
        
        # Update game state
        alive_players = []
        
        # Update bullets
        for b in self.effect_bullets.copy():
            if b.is_effect():
                # Check if bullet hit anyone
                # The original update method handles hitting players and reducing lives
                # But we need to know WHO got hit to assign rewards
                
                # We need to intercept the hit logic or check lives before/after
                prev_lives = [p.lives for p in self.players]
                b.update(self.game_map, tuple(self.players))
                
                # Check for hits
                for i, p in enumerate(self.players):
                    if p.lives < prev_lives[i]:
                        # Player i got hit
                        rewards[i] -= 100 # Penalty for getting hit
                        # Reward the shooter
                        if b.owner == self.players[1-i]:
                            rewards[1-i] += 100
            else:
                self.effect_bullets.remove(b)

        # Update players
        for ply in self.players:
            if ply.lives > 0:
                alive_players.append(ply)
                ply.update(self.game_map)
        
        self.current_step += 1
        
        done = False
        if len(alive_players) <= 1 or self.current_step >= self.max_steps:
            done = True
            # Survival reward?
            # Maybe not needed if we have win/loss rewards
        
        # Small survival reward
        for i in range(2):
            if self.players[i].lives > 0:
                rewards[i] += 0.1

        return self.get_state(0), self.get_state(1), rewards, done

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
        # Construct feature vector
        me = self.players[player_idx]
        enemy = self.players[1 - player_idx]
        
        if me.lives <= 0:
            return np.zeros(28) # Dead state
            
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
        
        # 2. Enemy State (4)
        # Relative position might be better
        rel_pos = enemy.position - me.position
        state.extend([
            rel_pos.real / norm_x,
            rel_pos.imag / norm_y,
            math.sin(enemy.angle),
            math.cos(enemy.angle)
        ])
        
        # 3. Bullets (3 closest) (3 * 4 = 12)
        bullets_info = []
        for b in self.effect_bullets:
            if not b.is_effect(): continue
            d = abs(b.position - me.position)
            # Relative pos and velocity
            rel_b = b.position - me.position
            vx = math.cos(b.angle)
            vy = math.sin(b.angle)
            bullets_info.append((d, rel_b.real/norm_x, rel_b.imag/norm_y, vx, vy))
            
        bullets_info.sort(key=lambda x: x[0])
        
        for i in range(3):
            if i < len(bullets_info):
                state.extend(bullets_info[i][1:])
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
            ray_line = player.SegmentLine(c2t(pos), c2t(ray_end))
            
            # Check intersection with all walls
            # This might be slow. Optimization: only check nearby walls?
            # For now, check all walls in current map
            
            # Get all wall lines from map
            # Map stores walls as grid cells. We need to convert to lines.
            # Actually map.getWalls(point) gets nearby walls.
            # But ray can go far.
            
            # Let's iterate through all walls in the map
            # This is expensive. 
            # Optimization: Step along the ray and check grid collision.
            
            dist = 0
            step_size = 20
            curr_pos = pos
            hit = False
            
            for _ in range(int(max_dist / step_size)):
                curr_pos += ray_dir * step_size
                dist += step_size
                
                # Check if inside a wall
                # Convert to grid coords
                col = int((curr_pos.real - self.game_map.startx) / SQSIZE)
                row = int((curr_pos.imag - self.game_map.starty) / SQSIZE)
                
                # Check bounds
                if col < 0 or col >= self.game_map.cols or row < 0 or row >= self.game_map.rows:
                    hit = True # Hit boundary
                    break
                    
                # Check specific walls
                # This is tricky because walls are between cells.
                # Let's use a simpler approach: Check collision with map boundaries and wall segments
                # Using the existing collision logic is hard for rays.
                
                # Let's use the `getWalls` method which returns walls near a point
                walls, center = self.game_map.getWalls(c2t(curr_pos))
                if walls:
                    # If we are close to a wall, do precise check
                    lines = self.game_map.getLines(walls, center)
                    for line in lines:
                        # Check intersection between ray segment (pos to curr_pos) and wall line
                        # Actually we just need distance
                        # Let's just say if we are close enough to a wall line, it's a hit
                        # This is an approximation
                        pass
                        
                    # Simplified: If we are in a cell, check if we crossed a wall to get here?
                    # Too complex for this step.
                    
                    # Alternative: Just check if the point is close to any wall line returned by getWalls
                    # If getWalls returns walls, it means we are near a grid intersection/edge.
                    pass

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

