import pygame
import numpy as np
import torch
import os
import sys
import math
import argparse
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Argument parsing
parser = argparse.ArgumentParser(description='Train PPO Agent')
parser.add_argument('--mode', type=str, default='gui', choices=['gui', 'cli'], help='Training mode: gui or cli')
parser.add_argument('--load', type=str, default='auto', choices=['auto', 'latest', 'interrupted', 'none'], 
                    help='Load model strategy: auto (prefer interrupted), latest, interrupted, or none (fresh start)')
args, _ = parser.parse_known_args()

if args.mode == 'cli':
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    print("Running in CLI (Headless) mode - Max Speed")

from source.sites import arena
from source import constants as C
from source.rl.ppo_agent import PPOAgent, Memory

class TankEnv:
    def __init__(self, mode='gui'):
        self.mode = mode
        self.arena = arena.Arena()
        self.action_space_n = 6 # Idle, Fwd, Bwd, Left, Right, Fire
        
        # State Dimension Calculation:
        # Self (9) + 2 Enemies (5*2=10) + Shells (4*5=20) + Lidar (8) = 47
        self.state_dim = 47 
        self.max_steps = 2000
        self.player_ids = [1, 2, 3]
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.get_surface()
        self.paused = False
        
    def reset(self):
        # Setup arena with 3 players
        self.arena.setup(player_num=3, score=(0, 0, 0))
        self.arena.manual_clock = True 
        self.steps = 0
        
        # Track positions for stagnation penalty
        self.prev_positions = {}
        for pid in self.player_ids:
            p = self.arena.get_player(pid)
            if p:
                self.prev_positions[pid] = (p.x, p.y)
            else:
                self.prev_positions[pid] = (0, 0)
                
        return self.get_all_states()
        
    def get_all_states(self):
        states = {}
        raw_state = self.arena.get_state(normalize=True)
        # 预先计算所有墙壁的矩形，用于射线检测加速
        self.wall_rects = [pygame.Rect(w['x']*C.SCREEN_W, w['y']*C.SCREEN_H, w['w']*C.SCREEN_W, w['h']*C.SCREEN_H) 
                           for w in raw_state['walls']]
        
        for pid in self.player_ids:
            states[pid] = self.get_state_vector(raw_state, pid)
        return states

    def step(self, actions):
        # 1. Event Handling & Pause
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                    print(f"Training {'PAUSED' if self.paused else 'RESUMED'}")
                    pygame.display.set_caption(f"RL Training - {'PAUSED' if self.paused else 'RUNNING'}")

        while self.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    self.paused = False
            self.clock.tick(10)

        # 2. Run Game Logic
        if self.mode == 'gui':
            # 加快训练速度：提高帧率上限，并跳过部分渲染
            self.clock.tick(C.FRAME_RATE * 5) # 5x Speed
        else:
            # CLI 模式：全速运行，不限制帧率
            pass
            
        self.steps += 1
        
        keys = defaultdict(int)
        for pid, act in actions.items():
            self.map_action_to_keys(pid, act, keys)
            
        self.arena.update(self.screen, keys)
        
        # 每 5 帧才刷新一次屏幕，大幅减少渲染开销
        if self.mode == 'gui' and self.steps % 5 == 0:
            pygame.display.update()
        
        next_states = self.get_all_states()
        rewards = {pid: 0 for pid in self.player_ids}
        dones = {pid: False for pid in self.player_ids}
        
        # 3. Reward Calculation
        events = self.arena.get_and_clear_events()
        for event in events:
            if event['type'] == 'hit':
                attacker = event['attacker']
                victim = event['victim']
                
                if attacker == victim:
                    # 自杀惩罚 (重罚)
                    if attacker in rewards: rewards[attacker] -= 50 
                else:
                    # 击杀奖励
                    if attacker in rewards: rewards[attacker] += 30
                    if victim in rewards: rewards[victim] -= 10
        
        game_over = self.arena.finished
        
        for pid in self.player_ids:
            p = self.arena.get_player(pid)
            
            # 死亡判定
            if not p or p.dead:
                dones[pid] = True
                # 如果不是因为被击中而死（比如撞墙死或者其他判定），额外给点惩罚
                # 但主要是通过 event 里的 hit 来扣分
            
            if game_over:
                dones[pid] = True
                if p and not p.dead:
                    rewards[pid] += 20 # 胜利奖励
            
            # 存活/行为奖励
            if p and not p.dead:
                rewards[pid] += 0.01 # 降低纯存活奖励，迫使它们去寻找击杀
                
                # 检查是否停滞 (Stagnation Check)
                curr_pos = (p.x, p.y)
                prev_pos = self.prev_positions.get(pid, curr_pos)
                dist_moved = math.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                
                # 每 10 步更新一次位置记录，用于检测长期停滞
                if self.steps % 10 == 0:
                    self.prev_positions[pid] = curr_pos
                    if dist_moved < 5.0: # 如果10步内移动距离小于5像素
                        rewards[pid] -= 20.0 # 停滞惩罚 (极刑，强迫移动)
                
                # 危险操作惩罚：如果正前方很近有墙，还开火
                if actions[pid] == 5: # Fire
                    rewards[pid] -= 0.1 # 开火成本 (Ammo Cost)，防止无限开火
                    
                    lidar = self.compute_lidar(p)
                    front_dist = lidar[0] # 0度是正前方
                    if front_dist < 0.3: # 距离很近
                        rewards[pid] -= 2.0 # 惩罚贴脸开火
                    
                    # 惩罚原地开火
                    if not p.moving:
                        rewards[pid] -= 30.0

                # 移动奖励
                if p.moving:
                    rewards[pid] += 1.0 # 移动奖励 (极大增加)
                else:
                    rewards[pid] -= 1.0 # 每一帧不移动都重罚


        if self.steps >= self.max_steps:
            game_over = True
            for pid in self.player_ids: dones[pid] = True
            
        return next_states, rewards, dones, game_over

    def compute_lidar(self, p):
        """
        计算 8 个方向的射线距离 (Lidar Sensors)
        返回: 8个 float [0, 1]，1表示远处，0表示贴脸
        """
        sensors = []
        start_x, start_y = p.x / C.MOTION_CALC_SCALE, p.y / C.MOTION_CALC_SCALE
        
        # 8 directions: 0, 45, 90, 135, 180, 225, 270, 315 relative to player heading
        for angle_deg in range(0, 360, 45):
            angle_rad = p.theta + math.radians(angle_deg)
            dx = math.cos(angle_rad)
            dy = math.sin(angle_rad)
            
            dist = 1.0 # Max distance (normalized)
            
            # 简化的射线检测：沿射线采样几个点
            # 真实射线检测太慢，这里检测 5 个关键点: 20px, 50px, 100px, 150px, 200px
            check_points = [20, 50, 100, 150, 200]
            max_range = 200.0
            
            found_wall = False
            for r in check_points:
                cx = start_x + dx * r
                cy = start_y + dy * r
                
                # 边界检查
                if cx < 0 or cx > C.SCREEN_W or cy < 0 or cy > C.SCREEN_H:
                    dist = r / max_range
                    found_wall = True
                    break
                
                # 墙壁检查
                # 简单的点在矩形内检查
                test_rect = pygame.Rect(cx-2, cy-2, 4, 4)
                
                # 优化：只检查附近的墙 (这里为了简单遍历所有墙，性能可能略低，但Python里做空间划分太繁琐)
                # 实际上我们可以利用 self.wall_rects
                for w_rect in self.wall_rects:
                    if w_rect.colliderect(test_rect):
                        dist = r / max_range
                        found_wall = True
                        break
                if found_wall:
                    break
            
            if not found_wall:
                dist = 1.0
                
            sensors.append(dist)
            
        return sensors

    def get_state_vector(self, raw_state, pid):
        p_self = self.arena.get_player(pid)
        
        vec = []
        
        # 1. Self Info (9)
        if p_self:
            vec.extend([
                p_self.x / (C.COLUMN_NUM * C.BLOCK_SIZE * C.MOTION_CALC_SCALE),
                p_self.y / (C.ROW_NUM * C.BLOCK_SIZE * C.MOTION_CALC_SCALE),
                np.sin(p_self.theta), np.cos(p_self.theta),
                1.0 if p_self.can_fire else 0.0,
                p_self.fire_cool_down_timer / C.FIRE_COOL_DOWN, # Approx
                len(p_self.rounds) / 5.0,
                1.0 if p_self.moving else 0.0,
                1.0 if p_self.not_stuck else 0.0
            ])
        else:
            vec.extend([0]*9)

        # 2. Enemies Info (10) - Relative Position
        enemies = []
        for p in raw_state['players']:
            if p['name'] != pid:
                enemies.append(p)
        
        # Sort by distance
        if p_self:
            enemies.sort(key=lambda e: (e['x'] - vec[0])**2 + (e['y'] - vec[1])**2)
            
        for i in range(2):
            if i < len(enemies):
                e = enemies[i]
                # Use Relative Position! Easier for AI to learn aiming
                dx = e['x'] - (vec[0] if p_self else 0)
                dy = e['y'] - (vec[1] if p_self else 0)
                vec.extend([dx, dy, e['sin_theta'], e['cos_theta'], e['dead']])
            else:
                vec.extend([0]*5)

        # 3. Shells Info (20) - Relative Position & Velocity
        shells = list(raw_state['shells'])
        if p_self:
            # Sort by danger (distance)
            shells.sort(key=lambda s: (s['x'] - vec[0])**2 + (s['y'] - vec[1])**2)
            
        for i in range(5):
            if i < len(shells):
                s = shells[i]
                dx = s['x'] - (vec[0] if p_self else 0)
                dy = s['y'] - (vec[1] if p_self else 0)
                vec.extend([dx, dy, s['vx'], s['vy']])
            else:
                vec.extend([0]*4)

        # 4. Lidar Sensors (8) - NEW!
        if p_self:
            lidar = self.compute_lidar(p_self)
            vec.extend(lidar)
        else:
            vec.extend([0]*8)

        return np.array(vec, dtype=np.float32)

    def map_action_to_keys(self, pid, action, keys):
        # 0: Idle, 1: Fwd, 2: Bwd, 3: Left, 4: Right, 5: Fire
        k_fwd, k_bwd, k_left, k_right, k_fire = 0, 0, 0, 0, 0
        
        if pid == 1:
            k_fwd, k_bwd, k_left, k_right, k_fire = pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_q
        elif pid == 2:
            k_fwd, k_bwd, k_left, k_right, k_fire = pygame.K_i, pygame.K_k, pygame.K_j, pygame.K_l, pygame.K_u
        elif pid == 3:
            k_fwd, k_bwd, k_left, k_right, k_fire = pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DELETE
            
        if action == 1: keys[k_fwd] = 1
        elif action == 2: keys[k_bwd] = 1
        elif action == 3: keys[k_left] = 1
        elif action == 4: keys[k_right] = 1
        elif action == 5: keys[k_fire] = 1
    
    def debug_draw_lidar(self):
        # 简单的调试绘制，画出每个玩家的射线
        for pid in self.player_ids:
            p = self.arena.get_player(pid)
            if p and not p.dead:
                lidar = self.compute_lidar(p)
                start_x, start_y = p.x / C.MOTION_CALC_SCALE, p.y / C.MOTION_CALC_SCALE
                for i, dist in enumerate(lidar):
                    angle_rad = p.theta + math.radians(i * 45)
                    end_x = start_x + math.cos(angle_rad) * (dist * 200)
                    end_y = start_y + math.sin(angle_rad) * (dist * 200)
                    color = (0, 255, 0) if dist > 0.2 else (255, 0, 0)
                    pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), 1)

def train():
    env = TankEnv(mode=args.mode)
    state_dim = env.state_dim
    action_dim = env.action_space_n
    
    # 使用 3 个独立的 Agent，让它们发展出不同的性格/策略
    # 这样可以避免所有 AI 同时陷入同一个局部最优（比如都只会原地开火）
    agents = {pid: PPOAgent(state_dim, action_dim) for pid in env.player_ids}
    
    # 尝试加载旧模型
    print(f"Checking for existing models (Strategy: {args.load})...")
    for pid in env.player_ids:
        model_path = None
        
        if args.load == 'none':
            print(f"Player {pid}: Starting fresh (load=none)")
            continue
            
        path_interrupted = f'ppo_agent{pid}_interrupted.pth'
        path_latest = f'ppo_agent{pid}_latest.pth'
        
        if args.load == 'auto':
            if os.path.exists(path_interrupted):
                model_path = path_interrupted
            elif os.path.exists(path_latest):
                model_path = path_latest
        elif args.load == 'interrupted':
            if os.path.exists(path_interrupted):
                model_path = path_interrupted
        elif args.load == 'latest':
            if os.path.exists(path_latest):
                model_path = path_latest
            
        if model_path:
            print(f"Loading model for Player {pid} from {model_path}")
            try:
                # 即使保存时在CPU，加载时也会适配当前Agent的设备
                state_dict = torch.load(model_path)
                agents[pid].policy.load_state_dict(state_dict)
                agents[pid].policy_old.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading model for Player {pid}: {e}")
        else:
            print(f"No model found for Player {pid} with strategy '{args.load}', starting from scratch.")

    memories = {pid: Memory() for pid in env.player_ids}
    
    max_episodes = 5000
    update_timestep = 2000 
    time_step = 0
    
    print("Starting MULTI-AGENT training (3 Separate Brains)...")
    
    try:
        for i_episode in range(1, max_episodes+1):
            states = env.reset()
            ep_rewards = {pid: 0 for pid in env.player_ids}
            active_players = set(env.player_ids)
            
            for t in range(env.max_steps):
                time_step += 1
                
                actions = {}
                log_probs = {}
                
                # 每个玩家使用自己的 Agent 决策
                for pid in active_players:
                    action, log_prob = agents[pid].select_action(states[pid])
                    actions[pid] = action
                    log_probs[pid] = log_prob
                    
                next_states, rewards, dones, game_over = env.step(actions)
                
                # 分别存储经验
                for pid in active_players:
                    memories[pid].states.append(torch.FloatTensor(states[pid]))
                    memories[pid].actions.append(torch.tensor(actions[pid]))
                    memories[pid].logprobs.append(torch.tensor(log_probs[pid]))
                    memories[pid].rewards.append(rewards[pid])
                    memories[pid].is_terminals.append(dones[pid])
                    
                    ep_rewards[pid] += rewards[pid]
                
                states = next_states
                
                for pid in list(active_players):
                    if dones[pid]:
                        active_players.remove(pid)
                
                # 更新所有 Agent
                if time_step % update_timestep == 0:
                    for pid in env.player_ids:
                        agents[pid].update(memories[pid])
                        memories[pid].clear_memory()
                    time_step = 0
                    
                if game_over or not active_players:
                    break
                    
            if i_episode % 10 == 0:
                print(f"Ep {i_episode} | Steps: {time_step} | Rewards: {ep_rewards}")
                
            if i_episode % 50 == 0:
                for pid in env.player_ids:
                    torch.save(agents[pid].policy.state_dict(), f'ppo_agent{pid}_latest.pth')

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving models...")
        for pid in env.player_ids:
            torch.save(agents[pid].policy.state_dict(), f'ppo_agent{pid}_interrupted.pth')
        print("Models saved successfully.")
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    train()