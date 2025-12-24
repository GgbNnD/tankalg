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
        Set a new path for the strategy to follow.
        path: List of (x, y) tuples in SCREEN COORDINATES (pixels).
        """
        self.path = path
        self.next_idx = 0
        self.stuck_steps = 0
        
        # Optimization: The first point in the path is usually the center of the cell 
        # the player is currently in. We should skip it to avoid backtracking,
        # unless it's the only point (target).
        if len(self.path) > 1:
            self.next_idx = 1
            
            # Further optimization: Check if we are already closer to the second point (index 2)
            # or if the second point (index 1) is "behind" us? 
            # For now, just skipping the current cell center (index 0) is the most important fix.

    def execute(self):
        """
        Execute the contact strategy for the current frame.
        Returns a dictionary with 'steering' and 'throttle' values.
        """
        if not self.path or self.next_idx >= len(self.path):
            return {'steering': 0, 'throttle': 0}
            
        target_pos = self.path[self.next_idx]
        # Convert player pos to pixels for comparison with path (which is in pixels)
        current_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
        
        # A. Waypoint Management
        dist = math.hypot(target_pos[0] - current_pos[0], target_pos[1] - current_pos[1])
        if dist < 25: # Reached current waypoint (increased radius slightly)
            self.next_idx += 1
            if self.next_idx >= len(self.path):
                return {'steering': 0, 'throttle': 0}
            target_pos = self.path[self.next_idx]
            
        # B. Steering & Throttle
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        desired_angle = math.atan2(dy, dx)
        
        diff = desired_angle - self.player.theta
        # Normalize to [-PI, PI]
        while diff > math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        
        steering = 0.0
        throttle = 0.0
        
        # Forward logic: if angle error < 45 degrees (approx 0.78 rad)
        if abs(diff) < 0.78:
            throttle = 1.0
        else:
            throttle = 0.0 # Stop to turn if angle is too large
            
        # Rotation logic: if angle error >= 12 degrees (approx 0.2 rad)
        if abs(diff) >= 0.2:
            if diff > 0:
                steering = 1.0
            else:
                steering = -1.0
                
        # C. Anti-Stuck & Collision Handling
        # Check if we are stuck (trying to move but position not changing)
        # Use a small epsilon for float comparison
        # Use scaled coordinates for stuck detection (more precision)
        raw_pos = (self.player.x, self.player.y)
        pos_diff = math.hypot(raw_pos[0] - self.prev_pos[0], raw_pos[1] - self.prev_pos[1])
        
        if throttle > 0 and pos_diff < 0.5: # Trying to move but stuck
            self.stuck_steps += 1
            
            # "Slide" logic: if stuck against wall, try rotating
            # This is a simplified version of the C++ logic
            # If we are stuck, try to rotate away from the wall or just wiggle
            if self.stuck_steps <= 5:
                 # Try to rotate in the direction we want to go, even if we are stuck moving forward
                 # This helps sliding along walls
                 if steering == 0:
                     # If we were going straight, pick a direction (maybe based on previous rotation or random)
                     steering = 1.0 
            
            if self.stuck_steps > 5: # Stuck for too long
                # Reverse
                throttle = -1.0
                steering = 0.0 # Back up straight
                
                # Reset stuck counter if we successfully backed up a bit (handled in next frames)
                # For now, just force reverse for a few frames? 
                # The C++ logic keeps backing up as long as stuckSteps > 5 condition is met?
                # No, it resets stuckSteps eventually or the position changes.
                
        else:
            self.stuck_steps = 0
            
        self.prev_pos = raw_pos
        self.prev_theta = self.player.theta
        
        return {'steering': steering, 'throttle': throttle}
