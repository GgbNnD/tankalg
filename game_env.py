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
        cols = 10
        rows = 6
        startx = (propreties.data["maxcols"] - cols) * SQSIZE // 2
        starty = (propreties.data["maxrows"] - rows) * SQSIZE // 2
        
        # Simple map with one wall in the middle
        walls = []
        # Boundaries
        for c in range(cols):
            walls.append((c, 0, -1))      # Top
            walls.append((c, rows, -1))   # Bottom
        for r in range(rows):
            walls.append((0, r, 1))       # Left
            walls.append((cols, r, 1))    # Right
            
        # Internal wall (Middle)
        walls.append((cols//2, rows//2, 1)) # Vertical wall in center

        # Bunker around enemy at (0,0)
        # Protects enemy from direct fire but leaves an opening
        walls.append((1, 0, 1))   # Vertical wall right of (0,0)
        walls.append((0, 2, -1))  # Horizontal wall bottom of (0,1)

        self.fixed_map_data = (walls, cols, rows, startx, starty)
        self.finished_players = []

    def reset(self):
        self.effect_bullets.clear()
        self.finished_players = [False] * self.num_players
        walls, cols, rows, startx, starty = self.fixed_map_data
        self.game_map = Map(walls, self.screen, cols, rows, startx, starty)
        
        # Reset enemy (Static at top-left)
        enemy_pos = complex(0*SQSIZE+70+startx, 0*SQSIZE+70+starty)
        self.enemy.newgame(enemy_pos, 0)
        self.enemy.lives = 1
        
        # Reset players (Start at bottom-right)
        for ply in self.players:
            pos = complex((cols-1)*SQSIZE+70+startx, (rows-1)*SQSIZE+70+starty)
            angle = random()*2*pi
            ply.newgame(pos, angle)
        
        self.current_step = 0
        return [self.get_state(i) for i in range(self.num_players)]

    def step(self, actions):
        # Actions: List of [move, turn, shoot] for each player
        
        rewards = [0] * self.num_players
        
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
                # Check if bullet hit enemy
                # We use a copy of enemy to check collision without killing it for everyone
                # Actually, bullet.update modifies the target's lives.
                # We need to reset enemy lives after check or use a dummy.
                
                # Hack: Reset enemy lives before each check if we want it to survive?
                # No, bullet.update takes a tuple of players.
                # If we pass (self.enemy,), it modifies self.enemy.
                
                # Let's save enemy lives
                prev_lives = self.enemy.lives
                b.update(self.game_map, (self.enemy,))
                
                if self.enemy.lives < prev_lives:
                    # Enemy hit!
                    # Restore enemy lives so others can kill it too
                    self.enemy.lives = prev_lives
                    
                    # Find owner
                    for i, ply in enumerate(self.players):
                        if b.owner == ply and not self.finished_players[i]:
                            rewards[i] += 100.0 # Big reward for killing enemy
                            self.finished_players[i] = True
                            break
                    
                    # Remove bullet
                    if b in self.effect_bullets:
                        self.effect_bullets.remove(b)
            else:
                self.effect_bullets.remove(b)

        # Update players
        for i, ply in enumerate(self.players):
            if not self.finished_players[i] and ply.lives > 0:
                # Calculate distance before move
                dist_before = abs(ply.position - self.enemy.position)
                
                ply.update(self.game_map)
                
                # Calculate distance after move
                dist_after = abs(ply.position - self.enemy.position)
                
                # Reward for getting closer (Distance Shaping)
                # Scale: Moving 1 pixel closer gives 0.05 reward
                # rewards[i] += (dist_before - dist_after) * 0.05
                
                # Time penalty to encourage finishing the game
                rewards[i] -= 0.03
        
        self.current_step += 1
        
        # Done if all players finished or timeout
        done = all(self.finished_players) or self.current_step >= self.max_steps
        
        return [self.get_state(i) for i in range(self.num_players)], rewards, done

    def render(self):
        if not self.render_mode: return
        
        self.screen.fill((255, 255, 255))
        
        for b in self.effect_bullets:
            b.draw(self.screen)
            
        # Draw enemy
        if self.enemy.lives > 0:
            self.enemy.draw(self.screen)
            
        for ply in self.players:
            if ply.lives > 0:
                ply.draw(self.screen)
                
        self.game_map.draw()
        pygame.display.update()

    def get_state(self, player_idx):
        # If finished, return zeros (or last state, but zeros is fine as they don't act)
        if self.finished_players[player_idx]:
            return np.zeros(16)

        # Construct feature vector
        me = self.players[player_idx]
        
        if me.lives <= 0:
            return np.zeros(16) # Dead state
            
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
        
        # 2. Enemy State (4) - Static Enemy
        if self.enemy.lives > 0:
            rel_pos = self.enemy.position - me.position
            state.extend([
                rel_pos.real / norm_x,
                rel_pos.imag / norm_y,
                math.sin(self.enemy.angle),
                math.cos(self.enemy.angle)
            ])
        else:
            # Enemy dead
            state.extend([0, 0, 0, 0])
        
        # 3. Wall Raycasts (8 directions) (8)
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

