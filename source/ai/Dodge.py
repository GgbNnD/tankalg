import math
import pygame
from .. import constants as C
from ..parts import cell

class DodgeStrategy:
    def __init__(self, player):
        self.player = player
        self.arena = player.arena
        self.threat_range = 600 # Increased detection range
        self.max_predict_steps = 150
        self.active_strategy = None # Current dodge command queue
        self.debug_timer = 0

    def get_dodge_action(self, all_trajectories=None):
        """
        Determines if the AI needs to dodge and returns the action.
        Returns a dictionary with 'steering' and 'throttle' if dodging, else None.
        """
        # 1. Threat Detection & Prediction
        if all_trajectories is None:
            all_trajectories = self.get_all_bullet_trajectories()
            
        # Filter threats that are hitting us NOW
        threats = []
        my_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        
        for dist, s, trajectory in all_trajectories:
             if self.check_trajectory_collision(trajectory, my_pos):
                threats.append((dist, s, trajectory))
        
        # Debug Print (1s interval)
        current_time = pygame.time.get_ticks()
        if current_time - self.debug_timer > 1000:
            self.debug_timer = current_time
            if threats:
                print(f"[DODGE DEBUG] Threats: {len(threats)} | Closest Dist: {threats[0][0]:.2f}")
                # Check if generate_dodge_strategy finds a solution
                action = self.generate_dodge_strategy(threats, all_trajectories)
                print(f"[DODGE DEBUG] Action: {action}")
            elif all_trajectories:
                # Print info about nearest bullet even if not threatening, to check detection
                print(f"[DODGE DEBUG] No Threats. Nearest Bullet: {all_trajectories[0][0]:.2f}")
            else:
                print("[DODGE DEBUG] No Bullets Detected")
        
        if not threats:
            self.active_strategy = None
            return None
            
        # 2. Strategy Generation
        # Pass all trajectories to ensure we don't dodge into another bullet
        action = self.generate_dodge_strategy(threats, all_trajectories)
        return action

    def get_all_bullet_trajectories(self):
        """
        Returns a list of (distance, shell, trajectory) for all nearby shells.
        """
        trajectories = []
        shells = self.arena.get_shells(normalize=False)
        
        # We need actual shell objects
        all_shells = []
        for p in self.arena.players:
            for s in p.rounds:
                all_shells.append(s)
                
        my_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        
        for s in all_shells:
            # Ignore own bullets that are just fired (protection period)
            if s.player == self.player and s.protection:
                continue
                
            # Check distance
            s_pos = (s.x / C.MOTION_CALC_SCALE, s.y / C.MOTION_CALC_SCALE)
            dist = math.hypot(s_pos[0] - my_pos[0], s_pos[1] - my_pos[1])
            
            if dist > self.threat_range:
                continue
                
            # Predict trajectory
            trajectory = self.predict_trajectory(s)
            trajectories.append((dist, s, trajectory))
            
        # Sort by distance
        trajectories.sort(key=lambda x: x[0])
        return trajectories

    def get_threats(self):
        # Legacy method wrapper if needed, but we use get_all_bullet_trajectories now
        trajectories = self.get_all_bullet_trajectories()
        threats = []
        my_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        for dist, s, trajectory in trajectories:
             if self.check_trajectory_collision(trajectory, my_pos):
                threats.append((dist, s, trajectory))
        return threats

    def predict_trajectory(self, shell):
        """
        Predicts the shell's path for max_predict_steps.
        Returns a list of line segments [(start_pos, end_pos), ...].
        """
        segments = []
        
        # Clone shell state for simulation
        curr_x = shell.x
        curr_y = shell.y
        curr_vx = shell.vx
        curr_vy = shell.vy
        
        # Simulation loop
        # We simulate in larger steps or exact physics steps?
        # Let's simulate step-by-step but maybe optimize
        
        sim_steps = self.max_predict_steps
        
        seg_start = (curr_x / C.MOTION_CALC_SCALE, curr_y / C.MOTION_CALC_SCALE)
        
        for _ in range(sim_steps):
            next_x = curr_x + curr_vx
            next_y = curr_y + curr_vy
            
            # Collision check with walls (Simplified version of shell.update_collisions)
            # We need to detect bounce
            
            # Convert to screen coords for wall check
            cx, cy = next_x / C.MOTION_CALC_SCALE, next_y / C.MOTION_CALC_SCALE
            
            # Check walls
            # This is expensive to do exactly like the game loop for every step
            # We can use raycasting for optimization, but let's stick to step simulation for accuracy
            
            bounced = False
            
            # Get nearby walls
            cells_idx = cell.calculate_cell_num(cx, cy)
            
            # Create a dummy rect for collision
            # Shell size is small
            s_rect = pygame.Rect(0, 0, C.ROUND_PX, C.ROUND_PY)
            s_rect.center = (cx, cy)
            
            for i in range(4):
                if i >= len(cells_idx): break
                c = self.arena.cells[cells_idx[i]]
                for w in c.walls:
                    if s_rect.colliderect(w.rect):
                        # Bounce logic
                        if abs(w.rect.top - s_rect.bottom) <= C.ROUND_COLLITION_EPS + 2 and curr_vy > 0:
                            curr_vy *= -1
                            bounced = True
                        elif abs(w.rect.bottom - s_rect.top) <= C.ROUND_COLLITION_EPS + 2 and curr_vy < 0:
                            curr_vy *= -1
                            bounced = True
                        elif abs(w.rect.right - s_rect.left) <= C.ROUND_COLLITION_EPS + 2 and curr_vx < 0:
                            curr_vx *= -1
                            bounced = True
                        elif abs(w.rect.left - s_rect.right) <= C.ROUND_COLLITION_EPS + 2 and curr_vx > 0:
                            curr_vx *= -1
                            bounced = True
                        
                        if bounced: break
                if bounced: break
            
            curr_x = next_x
            curr_y = next_y
            
            if bounced:
                seg_end = (curr_x / C.MOTION_CALC_SCALE, curr_y / C.MOTION_CALC_SCALE)
                segments.append((seg_start, seg_end))
                seg_start = seg_end
                
        # Add final segment
        seg_end = (curr_x / C.MOTION_CALC_SCALE, curr_y / C.MOTION_CALC_SCALE)
        segments.append((seg_start, seg_end))
        
        return segments

    def check_trajectory_collision(self, trajectory, pos):
        # Check if any segment intersects with player's hitbox (approximated as circle/rect)
        # Player radius approx + Bullet radius approx
        # C.PLAYER_PX is 41, C.ROUND_PX is 9.
        # Radius sum = (41 + 9) / 2 = 25.
        # Adding a small safety margin (+5) = 30.
        radius = 30 
        
        for start, end in trajectory:
            if self.line_intersects_circle(start, end, pos, radius):
                return True
        return False

    def line_intersects_circle(self, p1, p2, center, radius):
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

    def generate_dodge_strategy(self, threats, all_trajectories=None):
        # Priority 1: Rotation
        # Try rotating in place to minimize profile? 
        # In Tank Trouble, player is somewhat circular but hitboxes are complex.
        # Rotating might move hitboxes out of way.
        
        # Priority 2: Rotate and Move (Side Dodge)
        # Determine dodge direction based on bullet velocity
        
        # We simulate different actions and check if they result in safety
        
        # Simplified approach:
        # Try 4 directions: Forward, Backward, Rotate CW, Rotate CCW
        
        best_action = None
        
        # Candidates: (steering, throttle)
        candidates = [
            (0, 0), # Stop
            (1, 0), # Rotate CW
            (-1, 0), # Rotate CCW
            (0, 1), # Forward
            (0, -1), # Backward
            (1, 1), # Forward Right
            (-1, 1), # Forward Left
            (1, -1), # Backward Right
            (-1, -1) # Backward Left
        ]
        
        # If all_trajectories is not provided, use threats (which is a subset)
        # But ideally we should use all_trajectories to avoid dodging into other bullets
        check_against = all_trajectories if all_trajectories is not None else threats
        
        # First pass: Find a completely safe action
        for steering, throttle in candidates:
            if self.simulate_action_safety(steering, throttle, check_against):
                return {'steering': steering, 'throttle': throttle}
                
        # Second pass: If no safe action, find the "least dangerous" action
        # We want to maximize the distance to the closest threat
        best_candidate = (0, 0)
        max_min_dist = -1.0
        
        for steering, throttle in candidates:
            # Simulate final position after N frames
            final_pos = self.simulate_movement(steering, throttle, steps=30)
            if final_pos is None: # Hit wall
                continue
                
            # Calculate min distance to any threat trajectory
            min_dist = float('inf')
            for _, _, trajectory in check_against:
                d = self.get_distance_to_trajectory(trajectory, final_pos)
                if d < min_dist:
                    min_dist = d
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_candidate = (steering, throttle)
                
        # If we found a better action than stopping (or stopping is best)
        return {'steering': best_candidate[0], 'throttle': best_candidate[1]}

    def is_action_safe(self, steering, throttle, all_trajectories):
        """
        Public method to check if an action is safe against all trajectories.
        """
        return self.simulate_action_safety(steering, throttle, all_trajectories)

    def simulate_movement(self, steering, throttle, steps=30):
        """
        Simulates movement and returns final position (cx, cy) in pixels.
        Returns None if wall collision occurs.
        """
        sim_x = self.player.x
        sim_y = self.player.y
        sim_theta = self.player.theta
        
        for _ in range(steps):
            if throttle != 0:
                v = C.BASE_MOVE_V if throttle > 0 else C.BACKWARD_V
                sim_x += throttle * v * math.cos(sim_theta)
                sim_y += throttle * v * math.sin(sim_theta)
            
            if steering != 0:
                sim_theta += steering * C.BASE_TURN_W
                
            # Wall Check
            cx, cy = sim_x / C.MOTION_CALC_SCALE, sim_y / C.MOTION_CALC_SCALE
            cells_idx = cell.calculate_cell_num(cx, cy)
            p_rect = pygame.Rect(0, 0, C.PLAYER_PX, C.PLAYER_PY)
            p_rect.center = (cx, cy)
            
            for i in range(4):
                if i >= len(cells_idx): break
                c = self.arena.cells[cells_idx[i]]
                for w in c.walls:
                    if p_rect.colliderect(w.rect):
                        return None # Hit wall
                        
        return (sim_x / C.MOTION_CALC_SCALE, sim_y / C.MOTION_CALC_SCALE)

    def simulate_action_safety(self, steering, throttle, threats):
        # Simulate player movement for a few frames
        # Check if new position collides with ANY threat trajectory
        
        # Current state
        sim_x = self.player.x
        sim_y = self.player.y
        sim_theta = self.player.theta
        
        # Simulate 30 frames (approx 0.5s at 60fps, enough to move ~30px)
        # Increased from 10 to 30 to allow escaping the collision radius
        for _ in range(30):
            # Apply physics (Simplified)
            if throttle != 0:
                v = C.BASE_MOVE_V if throttle > 0 else C.BACKWARD_V
                # FIX: Do NOT multiply by MOTION_CALC_SCALE here. 
                # BASE_MOVE_V is already in the correct units for adding to self.x
                sim_x += throttle * v * math.cos(sim_theta)
                sim_y += throttle * v * math.sin(sim_theta)
            
            if steering != 0:
                sim_theta += steering * C.BASE_TURN_W
                
            # Check wall collision (Feasibility)
            # If hits wall, this action is invalid
            cx, cy = sim_x / C.MOTION_CALC_SCALE, sim_y / C.MOTION_CALC_SCALE
            # Simple check: is center inside a wall?
            # (Full collision check is complex, assume center check is enough for feasibility)
            cells_idx = cell.calculate_cell_num(cx, cy)
            p_rect = pygame.Rect(0, 0, C.PLAYER_PX, C.PLAYER_PY)
            p_rect.center = (cx, cy)
            
            for i in range(4):
                if i >= len(cells_idx): break
                c = self.arena.cells[cells_idx[i]]
                for w in c.walls:
                    if p_rect.colliderect(w.rect):
                        return False # Hit wall, invalid action
            
            # Check threat collision (Safety)
            sim_pos = (cx, cy)
            for _, _, trajectory in threats:
                if self.check_trajectory_collision(trajectory, sim_pos):
                    return False # Still hits bullet
                    
        return True

    def get_distance_to_trajectory(self, trajectory, pos):
        """
        Returns the minimum distance from pos to any segment in the trajectory.
        """
        min_dist_sq = float('inf')
        px, py = pos
        
        for p1, p2 in trajectory:
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1
            
            if dx == 0 and dy == 0:
                d_sq = (px - x1)**2 + (py - y1)**2
            else:
                t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                closest_x = x1 + t * dx
                closest_y = y1 + t * dy
                d_sq = (closest_x - px)**2 + (closest_y - py)**2
                
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                
        return math.sqrt(min_dist_sq)
