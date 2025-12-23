import pygame
import random
from collections import defaultdict
from ..parts import player, cell, generate_maze, supply
from .. import tools, setup, constants as C


class Arena:
    def setup(self, player_num, score):
        self.finished = False
        self.player_num = player_num
        self.score = score
        self.next = 'arena'
        self.random = random.random()

        self.setup_states()
        self.setup_map()
        self.setup_players()
        self.setup_supplies()

    def setup_states(self):
        self.clock = 0
        self.can_pause = True
        self.pause = False
        self.ending = False
        self.ending_timer = 0
        self.celebrating = False
        self.celebrating_timer = 0
        # 事件队列（供外部训练/逻辑读取并清空）
        self._events = []

    def setup_map(self):
        self.cells = []
        self.map_surface = pygame.Surface((C.SCREEN_W, C.SCREEN_H)).convert()
        self.map_surface.fill(C.SCREEN_COLOR)

        # 不使用递归迷宫生成器，创建固定网格（无内部墙），地图固定
        for i in range(0, C.COLUMN_NUM*C.ROW_NUM):
            col = i % C.COLUMN_NUM
            row = int(i / C.COLUMN_NUM)
            walls_sign = 0
            if row == 0:
                walls_sign |= C.TOP
            if row == C.ROW_NUM-1:
                walls_sign |= C.BOTTOM
            if col == 0:
                walls_sign |= C.LEFT
            if col == C.COLUMN_NUM-1:
                walls_sign |= C.RIGHT

            self.cells.append(cell.Cell(
                C.LEFT_SPACE + col * C.BLOCK_SIZE,
                C.TOP_SPACE + row * C.BLOCK_SIZE,
                walls_sign=walls_sign))
            self.cells[i].draw_cell(self.map_surface)

        # 添加固定墙：在 x=1/3 和 x=2/3 处，各一面竖直墙
        # 第一面从上方向下覆盖高度的 2/3，第二面从下方向上覆盖高度的 2/3
        self.add_fixed_walls()

        for i in range(0, C.COLUMN_NUM*C.ROW_NUM):
            self.cells[i].draw_walls(self.map_surface)

    def add_fixed_walls(self):
        """
        在格子间添加两面固定竖直墙：分别位于归一化 x=1/3 和 x=2/3 处。
        第一面（x=1/3）覆盖顶部 2/3 行；第二面（x=2/3）覆盖底部 2/3 行。
        """
        # 计算目标列（取整到格子索引）
        col1 = int(C.COLUMN_NUM * (1.0/3.0))
        col2 = int(C.COLUMN_NUM * (2.0/3.0))
        col1 = max(0, min(C.COLUMN_NUM-1, col1))
        col2 = max(0, min(C.COLUMN_NUM-1, col2))

        rows_cover = int(round(C.ROW_NUM * (2.0/3.0)))

        # 在列边界处添加 RIGHT（放在左侧单元格的右侧）或 LEFT（若在最左边）的墙
        for row in range(0, rows_cover):
            if col1 > 0:
                idx = generate_maze.get_idx(col1-1, row)
                self.cells[idx].walls.add(cell.Wall(self.cells[idx].rect.x + C.BLOCK_SIZE, self.cells[idx].rect.y, C.RIGHT))
            else:
                idx = generate_maze.get_idx(col1, row)
                self.cells[idx].walls.add(cell.Wall(self.cells[idx].rect.x, self.cells[idx].rect.y, C.LEFT))

        start_row = C.ROW_NUM - rows_cover
        for row in range(start_row, C.ROW_NUM):
            if col2 > 0:
                idx = generate_maze.get_idx(col2-1, row)
                self.cells[idx].walls.add(cell.Wall(self.cells[idx].rect.x + C.BLOCK_SIZE, self.cells[idx].rect.y, C.RIGHT))
            else:
                idx = generate_maze.get_idx(col2, row)
                self.cells[idx].walls.add(cell.Wall(self.cells[idx].rect.x, self.cells[idx].rect.y, C.LEFT))

    def setup_supplies(self):
        self.supply_num = 0
        self.supply_timer = 0
        self.last_supply_idx = -1
        self.supplies = pygame.sprite.Group()

    def setup_players(self):
        self.rest_player_num = self.player_num
        self.players = pygame.sprite.Group()

        vised = {}
        for i in range(3):
            if self.score[i] != -1:
                aplayer = player.player(i+1, self)
                x = int(random.random()*C.COLUMN_NUM)
                y = int(random.random()*C.ROW_NUM)
                while vised.get(generate_maze.get_idx(x, y), False):
                    x = int(random.random()*C.COLUMN_NUM)
                    y = int(random.random()*C.ROW_NUM)
                vised[generate_maze.get_idx(x, y)] = True
                aplayer.x = self.cells[generate_maze.get_idx(
                    x, y)].rect.centerx*C.MOTION_CALC_SCALE
                aplayer.y = self.cells[generate_maze.get_idx(
                    x, y)].rect.centery*C.MOTION_CALC_SCALE
                aplayer.theta = 2*C.PI*random.random()
                self.players.add(aplayer)

    def update_ending(self):
        if not self.ending and self.rest_player_num <= 1:
            self.ending = True
            self.ending_timer = self.clock
        if self.ending and self.clock-self.ending_timer >= C.ENDING_TIME:
            self.ending = False
            self.celebrating = True
            self.celebrating_timer = self.clock
            for aplayer in self.players:
                if not aplayer.dead:
                    aplayer.celebrate()

        if self.celebrating and self.clock-self.celebrating_timer >= C.CELEBRATING_TIME:
            for aplayer in self.players:
                if not aplayer.dead:
                    self.score = tuple(
                        self.score[i]+(aplayer.name-1 == i) for i in range(3))
            self.finished = True

    def draw_info(self, surface):
        surface.blit(tools.create_label('ESC  /  P', 28),
                     (C.SCREEN_W*0.11, C.SCREEN_H*0.9))
        if self.pause:
            surface.blit(tools.create_label('pause...', 28),
                         (C.SCREEN_W*0.215, C.SCREEN_H*0.9))
        if self.score[0] != -1:
            surface.blit(tools.create_label(str(self.score[0]), 32),
                         (C.SCREEN_W*0.46, C.SCREEN_H*0.9))
            surface.blit(tools.create_label('PLAYER1', 32),
                         (C.SCREEN_W*0.33, C.SCREEN_H*0.9))
        if self.score[1] != -1:
            surface.blit(tools.create_label(str(self.score[1]), 32),
                         (C.SCREEN_W*0.66, C.SCREEN_H*0.9))
            surface.blit(tools.create_label('PLAYER2', 32),
                         (C.SCREEN_W*0.53, C.SCREEN_H*0.9))
        if self.score[2] != -1:
            surface.blit(tools.create_label(str(self.score[2]), 32),
                         (C.SCREEN_W*0.86, C.SCREEN_H*0.9))
            surface.blit(tools.create_label('PLAYER3', 32),
                         (C.SCREEN_W*0.73, C.SCREEN_H*0.9))

    def update_states(self, keys):
        if keys[pygame.K_ESCAPE]:
            self.next = 'main_menu'
            self.finished = True

        if self.can_pause and keys[pygame.K_p]:
            self.pause = not self.pause

        if keys[pygame.K_p]:
            self.can_pause = False
        else:
            self.can_pause = True

    def update_supply(self):
        if self.supply_timer == 0:
            self.supply_timer = self.clock
        elif self.clock-self.supply_timer > C.SUPPLY_TIME and self.supply_num < 2:
            idx = int(self.random*C.COLUMN_NUM*C.ROW_NUM)
            while idx == self.last_supply_idx:
                idx = int(random.random()*C.COLUMN_NUM*C.ROW_NUM)
            self.last_supply_idx = idx
            self.supplies.add(supply.Supply(self.cells[idx].rect.center))
            self.supply_num += 1
            self.supply_timer = self.clock

        for asupply in self.supplies:
            for player in self.players:
                hitbox = pygame.sprite.spritecollide(
                    asupply, player.hitboxes, False)
                if hitbox:
                    setup.SOUNDS['pick'].play()
                    asupply.kill()
                    self.supply_num -= 1
                    self.supply_timer = self.clock
                    if asupply.type:
                        player.shotgun = True
                        player.biground = False
                    else:
                        player.biground = True
                        player.shotgun = False
                    return

    def draw_map(self, surface):
        surface.blit(self.map_surface, (0, 0))

        # Visualize AI Path
        for player in self.players:
            if hasattr(player, 'ai_controller') and hasattr(player.ai_controller, 'path'):
                path = player.ai_controller.path
                if path and len(path) > 0:
                    points = []
                    # Start from player center
                    # player.x and player.y are already center coordinates in scaled space?
                    # Let's check player.py update_position.
                    # It seems player.x/y are high precision coordinates.
                    # And they seem to be the center based on rotation logic.
                    # But we need to divide by C.MOTION_CALC_SCALE to get screen coordinates?
                    # Wait, in smart_ai.py: my_pos = (self.player.x / C.MOTION_CALC_SCALE, self.player.y / C.MOTION_CALC_SCALE)
                    # So player.x is scaled up.
                    
                    start_x = player.x / C.MOTION_CALC_SCALE
                    start_y = player.y / C.MOTION_CALC_SCALE
                    points.append((start_x, start_y))
                    
                    for item in path:
                        if isinstance(item, int):
                            col = item % C.COLUMN_NUM
                            row = item // C.COLUMN_NUM
                            cx = C.LEFT_SPACE + col * C.BLOCK_SIZE + C.BLOCK_SIZE / 2
                            cy = C.TOP_SPACE + row * C.BLOCK_SIZE + C.BLOCK_SIZE / 2
                            points.append((cx, cy))
                        elif isinstance(item, (tuple, list)) and len(item) == 2:
                            points.append(item)
                    
                    if len(points) > 1:
                        # Draw path lines (Green)
                        pygame.draw.lines(surface, (0, 255, 0), False, points, 2)
                        # Draw target point (Red dot)
                        pygame.draw.circle(surface, (255, 0, 0), points[-1], 4)

        self.draw_info(surface)
        self.supplies.draw(surface)

    def get_ai_keys(self, player, original_keys):
        # 智能 AI 逻辑
        # original_keys 可能是 pygame.key.get_pressed()（序列）
        # 也可能是训练逻辑传入的 dict/defaultdict（mapping）
        if hasattr(original_keys, 'get'):
            # 创建一个 defaultdict(int) 以便对缺失键返回 0
            new_keys = defaultdict(int)
            new_keys.update(original_keys)
        else:
            new_keys = list(original_keys)

        # 清除玩家2的手动按键，防止干扰
        new_keys[pygame.K_i] = 0
        new_keys[pygame.K_k] = 0
        new_keys[pygame.K_j] = 0
        new_keys[pygame.K_l] = 0
        new_keys[pygame.K_u] = 0
        
        # 初始化 AI 控制器
        if not hasattr(player, 'ai_controller'):
            from ..ai.smart_ai import SmartAI
            player.ai_controller = SmartAI(player)
            
        # 获取 AI 决策
        actions = player.ai_controller.get_keys()
            
        if actions['forward']:
            new_keys[pygame.K_i] = 1
        if actions['backward']: # 虽然 SmartAI 目前没用到 backward，但预留
            new_keys[pygame.K_k] = 1
        if actions['left']:
            new_keys[pygame.K_j] = 1
        if actions['right']:
            new_keys[pygame.K_l] = 1
        if actions['fire']:
            new_keys[pygame.K_u] = 1
            
        return new_keys

    def update(self, surface, keys):
        self.clock = pygame.time.get_ticks()
        self.random = random.random()
        self.draw_map(surface)

        if not self.pause and not self.celebrating:
            for aplayer in self.players:
                if aplayer.name == 2:
                    # 玩家2 使用 AI 控制
                    ai_keys = self.get_ai_keys(aplayer, keys)
                    aplayer.update(ai_keys)
                else:
                    aplayer.update(keys)

            self.update_supply()

        if not self.pause:
            self.update_ending()

        if not self.celebrating:
            self.update_states(keys)

        for aplayer in self.players:
            aplayer.draw(surface)

    def get_player(self, name):
        for p in self.players:
            if p.name == name:
                return p
        return None

    def get_shells(self, max_shells=200, normalize=True):
        shells = []
        # 遍历所有玩家的子弹
        for p in self.players:
            for s in p.rounds:
                shells.append({
                    'x': float(s.x),
                    'y': float(s.y),
                    'vx': float(s.vx),
                    'vy': float(s.vy),
                    'owner': int(s.player.name),
                    'size': float(getattr(s, 'size', 1.0))
                })
                if len(shells) >= max_shells:
                    break
            if len(shells) >= max_shells:
                break

        if normalize:
            sx = float(C.MOTION_CALC_SCALE * C.COLUMN_NUM * C.BLOCK_SIZE)
            sy = float(C.MOTION_CALC_SCALE * C.ROW_NUM * C.BLOCK_SIZE)
            for sh in shells:
                sh['x'] = sh['x'] / max(1.0, sx)
                sh['y'] = sh['y'] / max(1.0, sy)
                sh['vx'] = sh['vx'] / max(1.0, sx)
                sh['vy'] = sh['vy'] / max(1.0, sy)
        return shells

    # --- 新增: 获取墙壁信息 ---
    def get_walls(self, normalize=True):
        walls = []
        # 遍历所有格子获取墙壁
        for c in self.cells:
            for w in c.walls:
                # w is a sprite with .rect
                cx, cy = w.rect.center
                wh, hh = w.rect.w, w.rect.h
                if normalize:
                    cx = float(cx) / max(1.0, C.SCREEN_W)
                    cy = float(cy) / max(1.0, C.SCREEN_H)
                    wh = float(wh) / max(1.0, C.SCREEN_W)
                    hh = float(hh) / max(1.0, C.SCREEN_H)
                walls.append({'x': cx, 'y': cy, 'w': wh, 'h': hh})
        return walls

    def get_state(self, normalize=True):
        """
        返回场景低维观测字典：
         - players: list of player.get_state()
         - shells: list of shells
         - supplies: list of supplies (归一化)
         - walls: list of walls (归一化) [新增]
         - meta: clock, rest_player_num, finished, score
        """
        players = [p.get_state(normalize=normalize) for p in self.players]
        supplies = []
        for s in self.supplies:
            cx, cy = s.rect.center
            if normalize:
                cx = float(cx) / max(1.0, C.SCREEN_W)
                cy = float(cy) / max(1.0, C.SCREEN_H)
            supplies.append({'x': float(cx), 'y': float(cy), 'type': int(s.type)})
            
        return {
            'players': players,
            'shells': self.get_shells(normalize=normalize),
            'supplies': supplies,
            'walls': self.get_walls(normalize=normalize), # 新增
            'meta': {
                'clock': int(self.clock),
                'rest_player_num': int(self.rest_player_num),
                'finished': bool(self.finished),
                'ending': bool(self.ending),
                'score': tuple(self.score)
            }
        }

    def get_and_clear_events(self):
        """返回当前事件列表并清空。事件的产生点可在后续需要时加入到 `self._events` 中。
        目前返回的事件格式由调用方决定；这里提供最小兼容性实现以配合训练脚本。
        """
        events = list(self._events)
        self._events.clear()
        return events
