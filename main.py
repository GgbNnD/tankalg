from mymap import *
from mycomp import *
from random import randint,random
import bullet,player,propreties,time
import json
class GameState:
    """游戏状态暴露类，用于AI决策树获取游戏信息"""
    def __init__(self):
        self.players = []
        self.bullets = []
        self.game_map = None

    def update_state(self, players, bullets, game_map):
        """更新游戏状态"""
        self.players = players
        self.bullets = bullets
        self.game_map = game_map

    def get_tank_positions(self):
        """获取所有坦克的位置信息"""
        positions = {}
        for i, tank in enumerate(self.players):
            if tank.lives > 0:
                positions[f"player_{i+1}"] = {
                    "position": (tank.position.real, tank.position.imag),
                    "angle": tank.angle,
                    "color": tank.color,
                    "lives": tank.lives,
                    "bullets": tank.bullets,
                    "score": tank.score
                }
        return positions

    def get_bullet_positions(self):
        """获取所有子弹的位置信息"""
        bullets_info = []
        for bullet_obj in self.bullets:
            if bullet_obj.is_effect():
                bullets_info.append({
                    "position": (bullet_obj.position.real, bullet_obj.position.imag),
                    "angle": bullet_obj.angle,
                    "owner": bullet_obj.owner.color if bullet_obj.owner else None
                })
        return bullets_info

    def get_map_info(self):
        """获取地图信息"""
        if self.game_map is None:
            return {}
        
        # 将墙壁坐标转换为实际像素坐标
        actual_walls = []
        for wall in self.game_map.walls:
            col, row, direction = wall
            if direction == -1:  # 横向墙
                x = col * SQSIZE + self.game_map.startx
                y = row * SQSIZE + self.game_map.starty
                actual_walls.append({
                    "type": "horizontal",
                    "start": (x, y),
                    "end": (x + SQSIZE + WALLWIDTH, y),
                    "raw": wall
                })
            elif direction == 1:  # 纵向墙
                x = col * SQSIZE + self.game_map.startx
                y = row * SQSIZE + self.game_map.starty
                actual_walls.append({
                    "type": "vertical",
                    "start": (x, y),
                    "end": (x, y + SQSIZE + WALLWIDTH),
                    "raw": wall
                })
        
        return {
            "walls": self.game_map.walls,
            "actual_walls": actual_walls,  # 实际像素坐标墙壁
            "rows": self.game_map.rows,
            "cols": self.game_map.cols,
            "startx": self.game_map.startx,
            "starty": self.game_map.starty
        }

    def get_full_state(self):
        """获取完整的游戏状态"""
        return {
            "tanks": self.get_tank_positions(),
            "bullets": self.get_bullet_positions(),
            "map": self.get_map_info(),
            "timestamp": time.time()
        }

    def to_json(self):
        """将游戏状态转换为JSON格式"""
        state = self.get_full_state()
        return json.dumps(state, indent=2)

pygame.init()
font = pygame.font.Font("propreties/simkai.ttf",SQSIZE - 40)
font2 = pygame.font.Font("propreties/simkai.ttf",SQSIZE - 80)
count = 0
timer = pygame.time.Clock()
fps = propreties.data["fps"]
waittime = 0
gene = make_generator()
width,height = propreties.data["maxcols"]*SQSIZE+WALLWIDTH,(propreties.data["maxrows"]+1)*SQSIZE+WALLWIDTH
screen = pygame.display.set_mode((width,height))
p1 = player.Player((255,0,0))
p2 = player.Player((0,255,0))
players = (p1,p2)
keepgoing = True
pause = False
effect_bullets = []
timetest = time.perf_counter()
real_fps = 60

game_state = GameState()

while keepgoing:
    count = (count+1)%60
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            keepgoing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                pause = not pause
            
            if event.key == pygame.K_s:
                p1.turn_mode = player.Player.LEFT
            elif event.key == pygame.K_f:
                p1.turn_mode = player.Player.RIGHT
            elif event.key == pygame.K_e:
                p1.move_mode = player.Player.FORWARD
            elif event.key == pygame.K_d:
                p1.move_mode = player.Player.BACK
            if event.key == pygame.K_q:
                if newb := p1.add_bullet():
                    effect_bullets.append(newb)
                    
            if event.key == pygame.K_LEFT:
                p2.turn_mode = player.Player.LEFT
            elif event.key == pygame.K_RIGHT:
                p2.turn_mode = player.Player.RIGHT
            elif event.key == pygame.K_UP:
                p2.move_mode = player.Player.FORWARD
            elif event.key == pygame.K_DOWN:
                p2.move_mode = player.Player.BACK
            if event.key == pygame.K_m:
                if newb := p2.add_bullet():
                    effect_bullets.append(newb)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_s:
                p1.turn_mode = 0
            elif event.key == pygame.K_f:
                p1.turn_mode = 0
            elif event.key == pygame.K_e:
                p1.move_mode = 0
            elif event.key == pygame.K_d:
                p1.move_mode = 0
            elif event.key == pygame.K_LEFT:
                p2.turn_mode = 0
            elif event.key == pygame.K_RIGHT:
                p2.turn_mode = 0
            elif event.key == pygame.K_UP:
                p2.move_mode = 0
            elif event.key == pygame.K_DOWN:
                p2.move_mode = 0
    screen.fill((255,255,255))
    alive_players = []
    for b in effect_bullets.copy():
        if b.is_effect():
            b.update(game_map,(p1,p2))
            b.draw(screen)
        else:
            effect_bullets.remove(b)
    for ply in players:
        if ply.lives > 0:
            alive_players.append(ply)
            ply.update(game_map)
            ply.draw(screen)
    if len(alive_players) <= 1 and waittime == -1:
        waittime = fps*3
    if waittime > 0:
        waittime -= 1
    if waittime == 0:
        waittime = -1
        if len(alive_players) == 1:
            alive_players[0].score += 1
        effect_bullets.clear()
        cols = randint(propreties.data["mincols"],propreties.data["maxcols"])
        rows = randint(propreties.data["minrows"],propreties.data["maxrows"])
        startx = (propreties.data["maxcols"] - cols)*SQSIZE//2
        starty = (propreties.data["maxrows"] - rows)*SQSIZE//2
        walls = gene.send((cols,rows))
        game_map = Map(walls,screen,cols,rows,startx,starty)
        p1.newgame(complex(randint(0,cols-1)*SQSIZE+70+startx,randint(0,rows-1)*SQSIZE+70+starty),random()*2*pi)
        p2.newgame(complex(randint(0,cols-1)*SQSIZE+70+startx,randint(0,rows-1)*SQSIZE+70+starty),random()*2*pi)
        
    game_state.update_state(players, effect_bullets, game_map)
    # if p1.lives > 0:  # 只有红色坦克存活时才打印
    #     red_tank_pos = (p1.position.real, p1.position.imag)
    #     red_tank_angle = p1.angle
    #     print(f"红色坦克位置: ({red_tank_pos[0]:.2f}, {red_tank_pos[1]:.2f}), 角度: {red_tank_angle:.2f}")
    # # 打印地图信息
    # map_info = game_state.get_map_info()
    # if map_info:
    #     walls_count = len(map_info['actual_walls'])
    #     print(f"地图墙壁数量: {walls_count}")
        
    #     # 计算中间位置的墙壁索引
    #     mid_start = max(0, walls_count // 2 - 2)
    #     mid_end = min(walls_count, walls_count // 2 + 3)
        
    #     print(f"中间墙壁坐标 (索引 {mid_start} 到 {mid_end-1}):")
    #     for i in range(mid_start, mid_end):
    #         if i < len(map_info['actual_walls']):
    #             wall = map_info['actual_walls'][i]
    #             print(f"  墙壁{i}: {wall['type']} - 从{wall['start']}到{wall['end']}")
    
    screen.blit(font.render(f"红方:{players[0].score}",True,(255,0,0)),(width//8,height-SQSIZE+20))
    screen.blit(font.render(f"绿方:{players[1].score}",True,(0,255,0)),(width*5//8,height-SQSIZE+20))
    if count == 0:
        real_fps = 1/(time.perf_counter()-timetest)
    timetest = time.perf_counter()
    blit_center(screen,font2.render(f"fps:{int(real_fps)}",True,(0,0,0)),(width//2,height-SQSIZE+20))
    game_map.draw()
    timer.tick(fps)
    pygame.display.update()
pygame.quit()