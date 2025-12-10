"""
遗传算法训练器 - 用于训练三个AI坦克
不使用外部库，仅使用Python标准库和numpy

遗传算法流程：
1. 初始化种群（随机生成神经网络权重）
2. 评估适应度（运行游戏，根据表现计算分数）
3. 选择（轮盘赌或锦标赛选择）
4. 交叉（单点交叉或均匀交叉）
5. 变异（随机改变部分权重）
6. 重复2-5直到满足终止条件
"""

import pygame
import sys
import math
import random
import pickle
import os
import time
import logging
import argparse
from collections import defaultdict

from source import tools, setup, constants as C
from source.sites import main_menu, load_screen, arena as arena_module


# ============================================
# 神经网络实现 (Pure Python)
# ============================================

class SimpleNeuralNetwork:
    """
    纯 Python 前馈神经网络
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization approximation
            limit = math.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            w = [[random.uniform(-limit, limit) for _ in range(layer_sizes[i+1])] for _ in range(layer_sizes[i])]
            b = [0.0 for _ in range(layer_sizes[i+1])]
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0 if x < 0 else 1
            
    def relu(self, x):
        return max(0, x)
    
    def forward(self, inputs):
        """前向传播"""
        current_activations = inputs
        
        for i in range(len(self.weights)):
            next_activations = []
            w = self.weights[i]
            b = self.biases[i]
            
            for j in range(len(b)): 
                activation = b[j]
                for k in range(len(current_activations)):
                    activation += current_activations[k] * w[k][j]
                
                if i == len(self.weights) - 1:
                    next_activations.append(self.sigmoid(activation))
                else:
                    next_activations.append(self.relu(activation))
            
            current_activations = next_activations
            
        return current_activations
    
    def get_weights(self):
        """获取所有权重作为一维列表"""
        flat = []
        for w_matrix in self.weights:
            for row in w_matrix:
                flat.extend(row)
        for b_vec in self.biases:
            flat.extend(b_vec)
        return flat
    
    def set_weights(self, flat_weights):
        """从一维列表设置权重"""
        idx = 0
        for i in range(len(self.weights)):
            # Set weights
            rows = len(self.weights[i])
            cols = len(self.weights[i][0])
            for r in range(rows):
                for c in range(cols):
                    self.weights[i][r][c] = flat_weights[idx]
                    idx += 1
            # Set biases
            cols = len(self.biases[i])
            for c in range(cols):
                self.biases[i][c] = flat_weights[idx]
                idx += 1
    
    def get_weight_count(self):
        """获取权重总数"""
        count = 0
        for w in self.weights:
            count += len(w) * len(w[0])
        for b in self.biases:
            count += len(b)
        return count
    
    def copy(self):
        """复制神经网络"""
        new_net = SimpleNeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)
        # Deep copy
        new_net.weights = [[row[:] for row in w] for w in self.weights]
        new_net.biases = [b[:] for b in self.biases]
        return new_net



# ============================================
# AI坦克控制器（使用神经网络）
# ============================================

class NeuralNetworkTankController:
    """
    使用神经网络控制坦克的AI
    输入：游戏状态（归一化）
    输出：5个动作概率（前进、后退、左转、右转、射击）
    """
    def __init__(self, player_name, neural_network=None):
        self.player_name = player_name
        
        # 输入特征数量：
        # - 自己的位置(2) + 朝向(2) + 能否射击(1) = 5
        # - 最近敌人的相对位置(2) + 相对角度(1) + 距离(1) = 4
        # - 第二近敌人的相对位置(2) + 相对角度(1) + 距离(1) = 4
        # - 最近墙壁距离(8个方向) = 8
        # - 最近补给相对位置(2) = 2
        # - 最近子弹相对位置(2) + 速度(2) + 距离(1) = 5
        # 总计: 28
        self.input_size = 28
        self.hidden_sizes = [64, 32] # 增加网络深度和宽度
        self.output_size = 5  # forward, backward, left, right, shoot
        
        if neural_network:
            self.nn = neural_network
        else:
            self.nn = SimpleNeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)
        
        # 适应度跟踪
        self.fitness = 0
        self.kills = 0
        self.damage_dealt = 0
        self.survival_time = 0
        self.distance_traveled = 0
        self.last_x = None
        self.last_y = None
        self.total_rotation = 0
        self.last_angle = None
        self.stuck_time = 0
    
    def extract_features(self, game_state):
        """从游戏状态提取特征"""
        features = []
        
        # 获取自己的状态
        my_state = None
        enemies = []
        for p in game_state.get('players', []):
            if p.get('name') == self.player_name:
                my_state = p
            elif not p.get('dead', True):
                enemies.append(p)
        
        if not my_state or my_state.get('dead', True):
            return [0.0] * self.input_size
        
        # 自己的位置和状态
        my_x = my_state.get('x', 0.5)
        my_y = my_state.get('y', 0.5)
        my_sin = my_state.get('sin_theta', 0)
        my_cos = my_state.get('cos_theta', 1)
        can_fire = float(my_state.get('can_fire', 0))
        
        features.extend([my_x, my_y, my_sin, my_cos, can_fire])
        
        # 更新移动距离和旋转
        if self.last_x is not None:
            self.distance_traveled += math.sqrt((my_x - self.last_x)**2 + (my_y - self.last_y)**2)
        self.last_x = my_x
        self.last_y = my_y
        
        my_angle = math.atan2(my_sin, my_cos)
        if self.last_angle is not None:
            diff = abs(my_angle - self.last_angle)
            if diff > math.pi:
                diff = 2*math.pi - diff
            self.total_rotation += diff
        self.last_angle = my_angle
        
        # 更新卡住时间
        if not my_state.get('not_stuck', 1):
            self.stuck_time += 1
        
        # 计算到敌人的相对位置和角度
        enemy_features = []
        for enemy in enemies:
            ex = enemy.get('x', 0.5)
            ey = enemy.get('y', 0.5)
            dx = ex - my_x
            dy = ey - my_y
            dist = math.sqrt(dx*dx + dy*dy) + 0.001
            
            # 相对角度
            angle_to_enemy = math.atan2(dy, dx)
            my_angle = math.atan2(my_sin, my_cos)
            relative_angle = angle_to_enemy - my_angle
            # 归一化到[-pi, pi]
            while relative_angle > math.pi:
                relative_angle -= 2 * math.pi
            while relative_angle < -math.pi:
                relative_angle += 2 * math.pi
            relative_angle /= math.pi  # 归一化到[-1, 1]
            
            enemy_features.append((dist, dx, dy, relative_angle))
        
        # 排序，取最近的两个敌人
        enemy_features.sort(key=lambda x: x[0])
        
        for i in range(2):
            if i < len(enemy_features):
                dist, dx, dy, rel_angle = enemy_features[i]
                features.extend([dx, dy, rel_angle, min(dist, 1.0)])
            else:
                features.extend([0.0, 0.0, 0.0, 1.0])
        
        # 计算到墙壁的距离（8个方向）
        walls = game_state.get('walls', [])
        my_angle = math.atan2(my_sin, my_cos)
        
        directions = [
            my_angle,                    # 前
            my_angle + math.pi/4,        # 右前
            my_angle + math.pi/2,        # 右
            my_angle + 3*math.pi/4,      # 右后
            my_angle + math.pi,          # 后
            my_angle - 3*math.pi/4,      # 左后
            my_angle - math.pi/2,        # 左
            my_angle - math.pi/4         # 左前
        ]
        
        for direction in directions:
            min_dist = 1.0
            for wall in walls:
                wx = wall.get('x', 0)
                wy = wall.get('y', 0)
                dx = wx - my_x
                dy = wy - my_y
                dist = math.sqrt(dx*dx + dy*dy)
                
                # 检查墙是否在这个方向
                angle_to_wall = math.atan2(dy, dx)
                angle_diff = abs(angle_to_wall - direction)
                while angle_diff > math.pi:
                    angle_diff = 2*math.pi - angle_diff
                
                if angle_diff < math.pi/8 and dist < min_dist:
                    min_dist = dist
            
            features.append(min_dist)
        
        # 最近补给的相对位置
        supplies = game_state.get('supplies', [])
        if supplies:
            nearest_supply = min(supplies, 
                                key=lambda s: (s.get('x', 0.5) - my_x)**2 + (s.get('y', 0.5) - my_y)**2)
            features.extend([nearest_supply.get('x', 0.5) - my_x, 
                           nearest_supply.get('y', 0.5) - my_y])
        else:
            features.extend([0.0, 0.0])
            
        # 最近子弹 (新增)
        shells = game_state.get('shells', [])
        if shells:
            # 找到最近的子弹
            nearest_shell = min(shells, 
                               key=lambda s: (s.get('x', 0) - my_x)**2 + (s.get('y', 0) - my_y)**2)
            
            sdx = nearest_shell.get('x', 0) - my_x
            sdy = nearest_shell.get('y', 0) - my_y
            sdist = math.sqrt(sdx*sdx + sdy*sdy)
            svx = nearest_shell.get('vx', 0)
            svy = nearest_shell.get('vy', 0)
            
            features.extend([sdx, sdy, svx, svy, min(sdist, 1.0)])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 1.0])
        
        return features
    
    def get_action(self, game_state):
        """根据游戏状态获取动作"""
        features = self.extract_features(game_state)
        outputs = self.nn.forward(features)
        
        # 输出：[forward, backward, left, right, shoot]
        # 使用阈值0.5判断是否执行动作
        action = {
            'move_forward': outputs[0] > 0.5,
            'move_backward': outputs[1] > 0.5,
            'turn_left': outputs[2] > 0.5,
            'turn_right': outputs[3] > 0.5,
            'shoot': outputs[4] > 0.5
        }
        
        # 防止同时前进和后退
        if action['move_forward'] and action['move_backward']:
            action['move_backward'] = False
        
        # 防止同时左转和右转
        if action['turn_left'] and action['turn_right']:
            action['turn_right'] = False
        
        return action
    
    def reset(self):
        """重置适应度追踪"""
        self.fitness = 0
        self.kills = 0
        self.damage_dealt = 0
        self.survival_time = 0
        self.distance_traveled = 0
        self.last_x = None
        self.last_y = None
        self.total_rotation = 0
        self.last_angle = None
        self.stuck_time = 0


# ============================================
# 遗传算法实现
# ============================================

class GeneticAlgorithm:
    """
    遗传算法类
    """
    def __init__(self, population_size=20, mutation_rate=0.1, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
        
        # 创建初始种群（每个个体包含3个坦克控制器）
        self.population = []
        for _ in range(population_size):
            individual = {
                'controllers': [
                    NeuralNetworkTankController(1),
                    NeuralNetworkTankController(2),
                    NeuralNetworkTankController(3)
                ],
                'fitness': 0
            }
            self.population.append(individual)
        
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
    
    def evaluate_fitness(self, controllers, game_result):
        """
        评估适应度
        game_result: 包含游戏结果的字典
        """
        total_fitness = 0
        
        for i, controller in enumerate(controllers):
            player_name = i + 1
            
            # 基础适应度组成：
            # 1. 存活时间奖励 (提高权重)
            survival_bonus = controller.survival_time / 1000.0  # 每1秒1分
            survival_bonus = 0.0
            # 2. 击杀奖励 (大幅提高，鼓励进攻)
            kill_bonus = controller.kills * 500  # 每次击杀500分
            
            # 3. 移动奖励（鼓励探索，但惩罚原地打转）
            # 如果旋转过多，移动奖励减少
            # rotation_penalty = max(0, controller.total_rotation - 50) * 0.5 # 放宽旋转限制
            movement_bonus = max(0, min(controller.distance_traveled * 5, 100))
            
            # 4. 胜利奖励
            winner_bonus = 0
            if game_result.get('winner') == player_name:
                winner_bonus = 1000
            
            # 5. 撞墙惩罚
            stuck_penalty = controller.stuck_time * 1.0 # 每帧撞墙扣1分
            
            # 6. 最后存活奖励
            if not game_result.get(f'player_{player_name}_dead', True):
                survival_bonus += 100
            
            controller.fitness = survival_bonus + kill_bonus + movement_bonus + winner_bonus - stuck_penalty
            # 确保适应度不为负
            controller.fitness = max(0, controller.fitness)
            
            total_fitness += controller.fitness
        
        return total_fitness
    
    def selection(self):
        """
        锦标赛选择
        """
        tournament_size = min(3, len(self.population))
        selected = []
        
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1, parent2):
        """
        单点交叉
        对每个控制器的权重进行交叉
        """
        if random.random() > self.crossover_rate:
            # 不交叉，直接复制 (需要深拷贝)
            child1 = {
                'controllers': [NeuralNetworkTankController(i+1, p.nn.copy()) for i, p in enumerate(parent1['controllers'])],
                'fitness': 0
            }
            child2 = {
                'controllers': [NeuralNetworkTankController(i+1, p.nn.copy()) for i, p in enumerate(parent2['controllers'])],
                'fitness': 0
            }
            return child1, child2
        
        child1 = {
            'controllers': [],
            'fitness': 0
        }
        child2 = {
            'controllers': [],
            'fitness': 0
        }
        
        for i in range(3):
            # 获取父代权重 (list)
            weights1 = parent1['controllers'][i].nn.get_weights()
            weights2 = parent2['controllers'][i].nn.get_weights()
            
            # 单点交叉
            crossover_point = random.randint(1, len(weights1) - 1)
            
            child1_weights = weights1[:crossover_point] + weights2[crossover_point:]
            child2_weights = weights2[:crossover_point] + weights1[crossover_point:]
            
            # 创建子代控制器
            child1_controller = NeuralNetworkTankController(i + 1)
            child1_controller.nn.set_weights(child1_weights)
            child1['controllers'].append(child1_controller)
            
            child2_controller = NeuralNetworkTankController(i + 1)
            child2_controller.nn.set_weights(child2_weights)
            child2['controllers'].append(child2_controller)
        
        return child1, child2
    
    def mutate(self, individual):
        """
        变异操作
        随机改变部分权重
        """
        for controller in individual['controllers']:
            weights = controller.nn.get_weights()
            
            # 列表变异
            for i in range(len(weights)):
                if random.random() < self.mutation_rate:
                    weights[i] += random.gauss(0, 0.3)
                    # 限制范围
                    weights[i] = max(-2, min(2, weights[i]))
            
            controller.nn.set_weights(weights)
        
        return individual
    
    def evolve(self):
        """
        进化一代
        """
        # 选择
        selected = self.selection()
        
        # 交叉和变异
        new_population = []
        
        # 精英保留（保留最好的2个个体）
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        for i in range(2):
            elite = {
                'controllers': [NeuralNetworkTankController(j + 1, sorted_pop[i]['controllers'][j].nn.copy()) for j in range(3)],
                'fitness': sorted_pop[i]['fitness']
            }
            new_population.append(elite)
        
        # 生成其余个体
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
        self.generation += 1
        
        # 更新最佳个体
        best_in_gen = max(self.population, key=lambda x: x['fitness'])
        if best_in_gen['fitness'] > self.best_fitness:
            self.best_fitness = best_in_gen['fitness']
            # 深拷贝最佳个体
            self.best_individual = {
                'controllers': [NeuralNetworkTankController(j + 1, best_in_gen['controllers'][j].nn.copy()) for j in range(3)],
                'fitness': best_in_gen['fitness']
            }
        
        self.fitness_history.append(self.best_fitness)
    
    def save(self, filename, extra_data=None):
        """保存最佳个体"""
        if self.best_individual:
            data = {
                'generation': self.generation,
                'best_fitness': self.best_fitness,
                'weights': [c.nn.get_weights() for c in self.best_individual['controllers']],
                'fitness_history': self.fitness_history,
                'extra_data': extra_data
            }
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved best individual to {filename}")
    
    def load(self, filename):
        """加载最佳个体"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.generation = data['generation']
            self.best_fitness = data['best_fitness']
            self.fitness_history = data.get('fitness_history', [])
            extra_data = data.get('extra_data', None)
            
            # 恢复最佳个体
            self.best_individual = {
                'controllers': [],
                'fitness': self.best_fitness
            }
            for i, weights in enumerate(data['weights']):
                controller = NeuralNetworkTankController(i + 1)
                controller.nn.set_weights(weights)
                self.best_individual['controllers'].append(controller)
            
            print(f"Loaded from {filename}, generation {self.generation}, best fitness {self.best_fitness}")
            return extra_data
        return None


# ============================================
# 训练环境
# ============================================

class GATrainingEnvironment:
    """
    遗传算法训练环境
    """
    def __init__(self, render=True, max_game_time=30000, speed=1.0, fixed_maze=False):
        """
        render: 是否渲染游戏画面
        max_game_time: 最大游戏时间（毫秒）
        speed: 游戏速度倍率
        fixed_maze: 是否使用固定迷宫
        """
        self.render = render
        self.max_game_time = max_game_time
        self.speed = speed
        self.fixed_maze = fixed_maze
        self.map_seed = None
        
        if fixed_maze:
            self.map_seed = random.randint(0, 100000)
            print(f"使用固定地图种子: {self.map_seed}")
        
        # 初始化pygame（如果还没有）
        if not pygame.get_init():
            pygame.init()
        
        # 设置显示
        if render:
            self.screen = pygame.display.set_mode(C.SCREEN_SIZE)
            pygame.display.set_caption('Tank Trouble - GA Training')
        else:
            # 无头模式
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            self.screen = pygame.display.set_mode(C.SCREEN_SIZE)
        
        self.clock = pygame.time.Clock()
    
    def run_game(self, controllers, fast_mode=False):
        """
        运行一局游戏
        controllers: 3个AI控制器列表
        fast_mode: 是否快速模式（不渲染，不限帧率）
        返回: 游戏结果字典
        """
        # 重置控制器
        for c in controllers:
            c.reset()
        
        # 创建游戏状态
        arena = arena_module.Arena()
        # 设置3个玩家，传入地图种子
        arena.setup(3, (0, 0, 0), map_seed=self.map_seed)
        
        start_time = pygame.time.get_ticks()
        game_start_clock = arena.clock
        
        # 追踪每个玩家的初始状态
        initial_positions = {}
        for p in arena.players:
            initial_positions[p.name] = {'x': p.x, 'y': p.y}
        
        result = {
            'winner': None,
            'player_1_dead': False,
            'player_2_dead': False,
            'player_3_dead': False,
            'game_time': 0
        }
        
        # 记录已死亡玩家，防止重复计算
        dead_players = set()
        
        running = True
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 检查游戏是否结束
            current_time = pygame.time.get_ticks()
            game_time = current_time - start_time
            
            if game_time > self.max_game_time:
                running = False
                continue
            
            if arena.finished or arena.rest_player_num <= 1:
                running = False
                # 找到获胜者
                for p in arena.players:
                    if not p.dead:
                        result['winner'] = p.name
                continue
            
            # 检查玩家死亡情况并更新击杀数
            for p in arena.players:
                if p.dead and p.name not in dead_players:
                    dead_players.add(p.name)
                    result[f'player_{p.name}_dead'] = True
                    
                    # 检查是否有击杀者
                    if hasattr(p, 'killer') and p.killer:
                        killer_name = p.killer.name
                        # 找到对应的控制器并增加击杀数
                        # controllers索引是 name-1
                        if 1 <= killer_name <= 3:
                            controllers[killer_name-1].kills += 1
            
            # 获取游戏状态
            game_state = arena.get_state(normalize=True)
            
            # 为每个玩家生成按键 - 使用 defaultdict 避免索引越界
            keys = defaultdict(int)
            
            # 玩家1的按键 (WASD + Q)
            action1 = controllers[0].get_action(game_state)
            if action1['move_forward']:
                keys[pygame.K_w] = 1
            if action1['move_backward']:
                keys[pygame.K_s] = 1
            if action1['turn_left']:
                keys[pygame.K_a] = 1
            if action1['turn_right']:
                keys[pygame.K_d] = 1
            if action1['shoot']:
                keys[pygame.K_q] = 1
            
            # 玩家2的按键 (IJKL + U)
            action2 = controllers[1].get_action(game_state)
            if action2['move_forward']:
                keys[pygame.K_i] = 1
            if action2['move_backward']:
                keys[pygame.K_k] = 1
            if action2['turn_left']:
                keys[pygame.K_j] = 1
            if action2['turn_right']:
                keys[pygame.K_l] = 1
            if action2['shoot']:
                keys[pygame.K_u] = 1
            
            # 玩家3的按键 (方向键 + DELETE)
            action3 = controllers[2].get_action(game_state)
            if action3['move_forward']:
                keys[pygame.K_UP] = 1
            if action3['move_backward']:
                keys[pygame.K_DOWN] = 1
            if action3['turn_left']:
                keys[pygame.K_LEFT] = 1
            if action3['turn_right']:
                keys[pygame.K_RIGHT] = 1
            if action3['shoot']:
                keys[pygame.K_DELETE] = 1
            
            # 更新游戏
            arena.update(self.screen, keys)
            
            # 更新控制器的存活时间
            for i, controller in enumerate(controllers):
                player_name = i + 1
                for p in arena.players:
                    if p.name == player_name:
                        if not p.dead:
                            controller.survival_time = game_time
                        else:
                            result[f'player_{player_name}_dead'] = True
            
            # 渲染
            if self.render and not fast_mode:
                pygame.display.update()
                self.clock.tick(C.FRAME_RATE * self.speed)
            elif fast_mode:
                # 快速模式：不限帧率，但偶尔处理事件
                if game_time % 1000 < 20:
                    pygame.display.update()
        
        result['game_time'] = game_time
        return result


def train_ga(generations=100, population_size=20, games_per_eval=3, 
             render=True, save_interval=10, load_checkpoint=True, speed=1.0, fixed_maze=False):
    """
    训练遗传算法
    
    generations: 训练代数
    population_size: 种群大小
    games_per_eval: 每个个体评估时运行的游戏数
    render: 是否渲染
    save_interval: 保存间隔
    load_checkpoint: 是否加载检查点
    speed: 游戏速度倍率
    fixed_maze: 是否使用固定迷宫
    """
    # 配置日志
    logging.basicConfig(filename='training.log', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', filemode='a')
    
    print("="*50)
    print("遗传算法训练器 - Tank Trouble (Pure Python)")
    print("="*50)
    print(f"种群大小: {population_size}")
    print(f"训练代数: {generations}")
    print(f"每个体评估游戏数: {games_per_eval}")
    print(f"游戏速度: {speed}x")
    print(f"固定迷宫: {fixed_maze}")
    print("="*50)
    
    logging.info(f"Starting training: pop={population_size}, gens={generations}, fixed_maze={fixed_maze}")
    
    # 初始化
    ga = GeneticAlgorithm(population_size=population_size)
    env = GATrainingEnvironment(render=render, speed=speed, fixed_maze=fixed_maze)
    
    # 尝试加载检查点
    checkpoint_file = 'ga_checkpoint.pkl'
    if load_checkpoint:
        extra_data = ga.load(checkpoint_file)
        if extra_data is not None or ga.generation > 0:
            print(f"从检查点继续训练，当前代数: {ga.generation}")
            if fixed_maze and isinstance(extra_data, dict) and 'map_seed' in extra_data:
                env.map_seed = extra_data['map_seed']
                print(f"恢复地图种子: {env.map_seed}")
    
    try:
        for gen in range(ga.generation, generations):
            start_time = time.time()
            print(f"\n--- 第 {gen + 1}/{generations} 代 ---")
            
            # 评估每个个体
            for i, individual in enumerate(ga.population):
                total_fitness = 0
                
                for game_idx in range(games_per_eval):
                    # 重置控制器
                    for c in individual['controllers']:
                        c.reset()
                    
                    # 运行游戏
                    result = env.run_game(individual['controllers'], fast_mode=(not render))
                    
                    # 计算适应度
                    fitness = ga.evaluate_fitness(individual['controllers'], result)
                    total_fitness += fitness
                
                # 平均适应度
                individual['fitness'] = total_fitness / games_per_eval
                
                print(f"  个体 {i + 1}/{population_size}: 适应度 = {individual['fitness']:.2f}")
            
            # 打印统计信息
            avg_fitness = sum(ind['fitness'] for ind in ga.population) / len(ga.population)
            best_fitness = max(ind['fitness'] for ind in ga.population)
            duration = time.time() - start_time
            
            print(f"  平均适应度: {avg_fitness:.2f}, 最佳适应度: {best_fitness:.2f}")
            print(f"  历史最佳适应度: {ga.best_fitness:.2f}")
            print(f"  耗时: {duration:.2f}s")
            
            logging.info(f"Gen {gen+1}: Avg={avg_fitness:.2f}, Best={best_fitness:.2f}, HistBest={ga.best_fitness:.2f}, Time={duration:.2f}s")
            
            # 进化
            ga.evolve()
            
            # 定期保存
            if (gen + 1) % save_interval == 0:
                extra_data = {'map_seed': env.map_seed} if fixed_maze else None
                ga.save(checkpoint_file, extra_data=extra_data)
        
        # 训练完成，保存最终结果
        extra_data = {'map_seed': env.map_seed} if fixed_maze else None
        ga.save('ga_final.pkl', extra_data=extra_data)
        print("\n训练完成！")
        
    except KeyboardInterrupt:
        print("\n训练被中断，保存当前进度...")
        extra_data = {'map_seed': env.map_seed} if fixed_maze else None
        ga.save(checkpoint_file, extra_data=extra_data)
    
    return ga


def demo_best(checkpoint_file='ga_checkpoint.pkl'):
    """
    演示最佳个体
    """
    ga = GeneticAlgorithm()
    if not ga.load(checkpoint_file):
        print(f"无法加载检查点文件: {checkpoint_file}")
        return
    
    print("演示最佳个体...")
    env = GATrainingEnvironment(render=True)
    
    while True:
        result = env.run_game(ga.best_individual['controllers'], fast_mode=False)
        print(f"游戏结束，获胜者: 玩家{result['winner']}")
        
        # 等待用户按键继续
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    else:
                        waiting = False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='遗传算法训练器')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'demo'],
                        help='运行模式: train(训练) 或 demo(演示)')
    parser.add_argument('--generations', type=int, default=50,
                        help='训练代数')
    parser.add_argument('--population', type=int, default=15,
                        help='种群大小')
    parser.add_argument('--games', type=int, default=2,
                        help='每个体评估游戏数')
    parser.add_argument('--no-render', action='store_true',
                        help='不渲染游戏画面（加速训练）')
    parser.add_argument('--checkpoint', type=str, default='ga_checkpoint.pkl',
                        help='检查点文件路径')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='游戏速度倍率 (仅在渲染模式下有效)')
    parser.add_argument('--fixed-maze', action='store_true',
                        help='使用固定迷宫进行训练')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ga(
            generations=args.generations,
            population_size=args.population,
            games_per_eval=args.games,
            render=not args.no_render,
            load_checkpoint=True,
            speed=args.speed,
            fixed_maze=args.fixed_maze
        )
    elif args.mode == 'demo':
        demo_best(args.checkpoint)
