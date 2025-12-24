import heapq
import math
from .. import constants as C

class Dijkstra:
    def __init__(self, arena):
        self.arena = arena
        self.cols = C.COLUMN_NUM
        self.rows = C.ROW_NUM
        
    def get_path(self, start_pos, end_pos):
        """
        Calculate a path from start_pos to end_pos.
        start_pos, end_pos: (x, y) tuples in world coordinates.
        Returns: List of (x, y) tuples representing the path.
        """
        start_cell = self._get_cell_idx(start_pos)
        end_cell = self._get_cell_idx(end_pos)
        
        if start_cell is None or end_cell is None:
            return []
            
        if start_cell == end_cell:
            return [end_pos]
            
        # Priority Queue: (cost, current_cell_idx, path_list)
        pq = [(0, start_cell, [start_cell])]
        visited = set()
        min_dists = {start_cell: 0}
        
        final_path_indices = None
        
        while pq:
            cost, curr, path = heapq.heappop(pq)
            
            if curr == end_cell:
                final_path_indices = path
                break
            
            if curr in visited:
                continue
            visited.add(curr)
            
            neighbors = self._get_neighbors(curr)
            for neighbor, move_cost in neighbors:
                new_cost = cost + move_cost
                if neighbor not in min_dists or new_cost < min_dists[neighbor]:
                    min_dists[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
                    
        if not final_path_indices:
            return []
            
        # Convert to world coordinates
        waypoints = []
        for idx in final_path_indices:
            cell = self.arena.cells[idx]
            waypoints.append(cell.rect.center)
            
        # Post-processing:
        # Replace the first waypoint with start_pos? 
        # No, usually we want to steer towards the first waypoint (center of start cell) 
        # or the second one if we are close to the first.
        # Let's just return the full path of centers, but replace the last one with exact end_pos.
        
        waypoints[-1] = end_pos
        
        # Simple smoothing: Remove the first waypoint if we are already "past" it or very close
        # But this depends on the vehicle dynamics. 
        # We'll leave the raw path (centers) and let the controller handle the "seek" logic.
        
        return waypoints

    def _get_cell_idx(self, pos):
        x, y = pos
        col = int((x - C.LEFT_SPACE) / C.BLOCK_SIZE)
        row = int((y - C.TOP_SPACE) / C.BLOCK_SIZE)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            return row * self.cols + col
        return None

    def _get_neighbors(self, cell_idx):
        neighbors = []
        r = cell_idx // self.cols
        c = cell_idx % self.cols
        current_cell = self.arena.cells[cell_idx]
        
        # Directions: (dr, dc, direction_name, wall_type, opp_wall_type)
        directions = [
            (-1, 0, 'UP', C.TOP, C.BOTTOM),
            (1, 0, 'DOWN', C.BOTTOM, C.TOP),
            (0, -1, 'LEFT', C.LEFT, C.RIGHT),
            (0, 1, 'RIGHT', C.RIGHT, C.LEFT)
        ]
        
        for dr, dc, name, wall_type, opp_wall_type in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbor_idx = nr * self.cols + nc
                neighbor_cell = self.arena.cells[neighbor_idx]
                
                # Check walls
                blocked = False
                # Check current cell's wall
                for w in current_cell.walls:
                    if w.type == wall_type:
                        blocked = True
                        break
                if not blocked:
                    # Check neighbor cell's wall
                    for w in neighbor_cell.walls:
                        if w.type == opp_wall_type:
                            blocked = True
                            break
                
                if not blocked:
                    neighbors.append((neighbor_idx, 1)) # Cost 1
                    
        return neighbors
