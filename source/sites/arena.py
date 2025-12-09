import pygame
import random
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

    def setup_map(self):
        self.cells = []
        self.map_surface = pygame.Surface((C.SCREEN_W, C.SCREEN_H)).convert()
        self.map_surface.fill(C.SCREEN_COLOR)

        generate_maze.predeal(self)
        generate_maze.Generage_Maze(self, 0, 0, C.COLUMN_NUM-1, C.ROW_NUM-1)

        for i in range(0, C.COLUMN_NUM*C.ROW_NUM):
            self.cells[i].draw_walls(self.map_surface)

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
        self.draw_info(surface)
        self.supplies.draw(surface)

    def get_ai_keys(self, player, original_keys):
        # 简单的随机 AI 逻辑
        new_keys = list(original_keys)
        
        # 清除玩家2的手动按键，防止干扰
        new_keys[pygame.K_i] = 0
        new_keys[pygame.K_k] = 0
        new_keys[pygame.K_j] = 0
        new_keys[pygame.K_l] = 0
        new_keys[pygame.K_u] = 0
        
        if not hasattr(player, 'ai_action_timer'):
            player.ai_action_timer = 0
            player.ai_action = 'idle'
            
        # 每 200ms 改变一次动作
        if self.clock - player.ai_action_timer > 200:
            player.ai_action_timer = self.clock
            # 动作概率：前进 40%，左转 20%，右转 20%，开火 10%，不动 10%
            actions = ['forward', 'left', 'right', 'fire', 'idle']
            weights = [0.4, 0.2, 0.2, 0.1, 0.1]
            player.ai_action = random.choices(actions, weights=weights)[0]
            
        if player.ai_action == 'forward':
            new_keys[pygame.K_i] = 1
        elif player.ai_action == 'left':
            new_keys[pygame.K_j] = 1
        elif player.ai_action == 'right':
            new_keys[pygame.K_l] = 1
        elif player.ai_action == 'fire':
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
