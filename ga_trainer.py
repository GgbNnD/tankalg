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
from collections import defaultdict
from source import tools, setup, constants as C
from source.sites import main_menu, load_screen, arena as arena_module


# ============================================
# 简单神经网络实现（不使用外部库）
# ============================================

def sigmoid(x):
    """Sigmoid激活函数"""
    # 防止溢出
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh(x):
    """Tanh激活函数"""
    x = max(-500, min(500, x))
    return math.tanh(x)


def relu(x):
    """ReLU激活函数"""
    return max(0, x)


class SimpleNeuralNetwork:
    """
    简单的前馈神经网络
    输入层 -> 隐藏层 -> 输出层
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 随机初始化权重
        self.weights_ih = [[random.uniform(-1, 1) for _ in range(input_size)] 
                           for _ in range(hidden_size)]
        self.bias_h = [random.uniform(-1, 1) for _ in range(hidden_size)]
        
        self.weights_ho = [[random.uniform(-1, 1) for _ in range(hidden_size)] 
                           for _ in range(output_size)]
        self.bias_o = [random.uniform(-1, 1) for _ in range(output_size)]
    
    def forward(self, inputs):
        """前向传播"""
        # 输入层 -> 隐藏层
        hidden = []
        for i in range(self.hidden_size):
            sum_val = self.bias_h[i]
            for j in range(self.input_size):
                sum_val += inputs[j] * self.weights_ih[i][j]
            hidden.append(tanh(sum_val))
        
        # 隐藏层 -> 输出层
        outputs = []
        for i in range(self.output_size):
            sum_val = self.bias_o[i]
            for j in range(self.hidden_size):
                sum_val += hidden[j] * self.weights_ho[i][j]
            outputs.append(sigmoid(sum_val))
        
        return outputs
    
    def get_weights(self):
        """获取所有权重作为一维列表"""
        weights = []
        for row in self.weights_ih:
            weights.extend(row)
        weights.extend(self.bias_h)
        for row in self.weights_ho:
            weights.extend(row)
        weights.extend(self.bias_o)
        return weights
    
    def set_weights(self, weights):
        """从一维列表设置权重"""
        idx = 0
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.weights_ih[i][j] = weights[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.bias_h[i] = weights[idx]
            idx += 1
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.weights_ho[i][j] = weights[idx]
                idx += 1
        for i in range(self.output_size):
            self.bias_o[i] = weights[idx]
            idx += 1
    
    def get_weight_count(self):
        """获取权重总数"""
        return (self.input_size * self.hidden_size + self.hidden_size +
                self.hidden_size * self.output_size + self.output_size)
    
    def copy(self):
        """复制神经网络"""
        new_nn = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_nn.set_weights(self.get_weights())
        return new_nn


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
        self.hidden_size = 24
        self.output_size = 5  # forward, backward, left, right, shoot
        
        if neural_network:
            self.nn = neural_network
        else:
            self.nn = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        
        # 适应度跟踪
        self.fitness = 0
        self.kills = 0
        self.suicides = 0
        self.damage_dealt = 0
        self.survival_time = 0
        self.distance_traveled = 0
        self.last_x = None
        self.last_y = None
        self.total_rotation = 0
        self.last_angle = None
        self.stuck_time = 0
        self.sum_distance_to_enemy = 0
        self.steps = 0
    
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
        
        # Map aspect ratio for correct angle calculation
        # Width = 12 * 94 = 1128, Height = 6 * 94 = 564. Ratio = 2.0
        aspect_ratio = 2.0
        
        for enemy in enemies:
            ex = enemy.get('x', 0.5)
            ey = enemy.get('y', 0.5)
            dx = (ex - my_x) * aspect_ratio # Correct for aspect ratio
            dy = (ey - my_y) * 1.0
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
        
        if enemy_features:
            self.sum_distance_to_enemy += enemy_features[0][0]
            self.steps += 1
        
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
                dx = (wx - my_x) * aspect_ratio # Correct for aspect ratio
                dy = (wy - my_y) * 1.0
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
                                key=lambda s: ((s.get('x', 0.5) - my_x)*aspect_ratio)**2 + (s.get('y', 0.5) - my_y)**2)
            features.extend([(nearest_supply.get('x', 0.5) - my_x)*aspect_ratio, 
                           (nearest_supply.get('y', 0.5) - my_y)])
        else:
            features.extend([0.0, 0.0])
            
        # 最近子弹 (新增)
        shells = game_state.get('shells', [])
        if shells:
            # 找到最近的子弹
            nearest_shell = min(shells, 
                               key=lambda s: ((s.get('x', 0) - my_x)*aspect_ratio)**2 + (s.get('y', 0) - my_y)**2)
            
            sdx = (nearest_shell.get('x', 0) - my_x) * aspect_ratio
            sdy = (nearest_shell.get('y', 0) - my_y)
            sdist = math.sqrt(sdx*sdx + sdy*sdy)
            svx = nearest_shell.get('vx', 0) * aspect_ratio
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
        
        # --- Heuristic Safety Layer (预定义策略层) ---
        
        # 1. Anti-Stuck (防卡死) - 已移除，避免抽搐
        # if self.stuck_time > 20: ...
        
        # 2. Wall Avoidance (防撞墙)
        # features indices: 13=Front, 14=RF, 15=R, 16=RB, 17=Back, 18=LB, 19=L, 20=LF
        SAFE_DIST = 0.01  # 降低到0.01，避免过早干预
        
        # 前方防撞
        front_dist = features[13]
        left_front = features[20]
        right_front = features[14]
        
        if action['move_forward'] and front_dist < SAFE_DIST:
             action['move_forward'] = False
             # 向空旷处转弯
             if left_front > right_front:
                 action['turn_left'] = True
                 action['turn_right'] = False
             else:
                 action['turn_right'] = True
                 action['turn_left'] = False

        # 后方防撞 (防止倒车撞墙)
        back_dist = features[17]
        left_back = features[18]
        right_back = features[16]
        
        if action['move_backward'] and back_dist < SAFE_DIST:
             action['move_backward'] = False
             # 停止倒车
        
        # 3. Aim Assist (辅助瞄准)
        # 如果敌人大致在正前方，强制射击
        # Enemy 1 relative angle is at index 7
        # Enemy 2 relative angle is at index 11
        # Angle is normalized [-1, 1]
        SHOOT_THRESHOLD = 0.05 # ~9 degrees
        
        rel_angle_1 = features[7]
        rel_angle_2 = features[11]
        
        # 检查是否有敌人在射击角度内且距离不是太远(features[8] is dist)
        if (abs(rel_angle_1) < SHOOT_THRESHOLD and features[8] < 0.8) or \
           (abs(rel_angle_2) < SHOOT_THRESHOLD and features[12] < 0.8):
            action['shoot'] = True
        
        return action
    
    def reset(self):
        """重置适应度追踪"""
        self.fitness = 0
        self.kills = 0
        self.suicides = 0
        self.damage_dealt = 0
        self.survival_time = 0
        self.distance_traveled = 0
        self.last_x = None
        self.last_y = None
        self.total_rotation = 0
        self.last_angle = None
        self.stuck_time = 0
        self.sum_distance_to_enemy = 0
        self.steps = 0


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
        
        # Initialize log file
        self.log_file = 'training_log.csv'
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("Generation,BestFitness,AvgFitness,MaxKills,MaxSurvival\n")
    
    def log_stats(self, best_ind, avg_fitness):
        """Log training statistics to CSV"""
        max_kills = 0
        max_survival = 0
        for c in best_ind['controllers']:
            max_kills = max(max_kills, c.kills)
            max_survival = max(max_survival, c.survival_time)
            
        with open(self.log_file, 'a') as f:
            f.write(f"{self.generation},{self.best_fitness:.2f},{avg_fitness:.2f},{max_kills},{max_survival}\n")
    
    def evaluate_fitness(self, controllers, game_result):
        """
        评估适应度
        game_result: 包含游戏结果的字典
        """
        total_fitness = 0
        
        for i, controller in enumerate(controllers):
            player_name = i + 1
            
            # 基础适应度组成：
            # 1. 存活时间奖励 (降低权重，防止只为了生存而躲避)
            survival_bonus = controller.survival_time / 2000.0  # 每2秒1分
            
            # 2. 击杀奖励 (大幅提高，鼓励进攻)
            kill_bonus = controller.kills * 500  # 每次击杀500分
            
            # 3. 移动奖励（鼓励探索，但惩罚原地打转）
            # 如果旋转过多，移动奖励减少
            rotation_penalty = max(0, controller.total_rotation - 20) * 2 # 允许一定旋转，超过后扣分
            movement_bonus = max(0, controller.distance_traveled * 10 - rotation_penalty)
            
            # 4. 胜利奖励
            winner_bonus = 0
            if game_result.get('winner') == player_name:
                winner_bonus = 500
            
            # 5. 撞墙惩罚
            stuck_penalty = controller.stuck_time * 2.0 # 每帧撞墙扣2分
            
            # 6. 最后存活奖励
            if not game_result.get(f'player_{player_name}_dead', True):
                survival_bonus += 100
            
            # 7. 自杀惩罚
            suicide_penalty = controller.suicides * 500
            
            # 8. 不活跃惩罚
            inactivity_penalty = 0
            if controller.distance_traveled < 2.0:
                inactivity_penalty = 200
            
            controller.fitness = survival_bonus + kill_bonus + movement_bonus + winner_bonus - stuck_penalty - inactivity_penalty - suicide_penalty
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
            return parent1, parent2
        
        child1 = {
            'controllers': [],
            'fitness': 0
        }
        child2 = {
            'controllers': [],
            'fitness': 0
        }
        
        for i in range(3):
            # 获取父代权重
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
            
            for i in range(len(weights)):
                if random.random() < self.mutation_rate:
                    # 高斯变异
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
                'controllers': [c.nn.copy() for c in sorted_pop[i]['controllers']],
                'fitness': 0
            }
            # 重新创建控制器
            elite['controllers'] = [
                NeuralNetworkTankController(j + 1, sorted_pop[i]['controllers'][j].nn.copy())
                for j in range(3)
            ]
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
            self.best_individual = best_in_gen
        
        self.fitness_history.append(self.best_fitness)
        
        # Calculate average fitness
        avg_fitness = sum(ind['fitness'] for ind in self.population) / len(self.population)
        self.log_stats(best_in_gen, avg_fitness)
    
    def save(self, filename):
        """保存最佳个体"""
        if self.best_individual:
            data = {
                'generation': self.generation,
                'best_fitness': self.best_fitness,
                'weights': [c.nn.get_weights() for c in self.best_individual['controllers']],
                'fitness_history': self.fitness_history
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
            return True
        return False


# ============================================
# 训练环境
# ============================================

class GATrainingEnvironment:
    """
    遗传算法训练环境
    """
    def __init__(self, render=True, max_game_time=30000, speed=1.0):
        """
        render: 是否渲染游戏画面
        max_game_time: 最大游戏时间（毫秒）
        speed: 游戏速度倍率
        """
        self.render = render
        self.max_game_time = max_game_time
        self.speed = speed
        
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
        # 设置3个玩家
        arena.setup(3, (0, 0, 0))
        
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
                        # Sync kills and suicides
                        controller.kills = p.kills
                        controller.suicides = p.suicides
                        
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
             render=True, save_interval=10, load_checkpoint=True, speed=1.0):
    """
    训练遗传算法
    
    generations: 训练代数
    population_size: 种群大小
    games_per_eval: 每个个体评估时运行的游戏数
    render: 是否渲染
    save_interval: 保存间隔
    load_checkpoint: 是否加载检查点
    speed: 游戏速度倍率
    """
    print("="*50)
    print("遗传算法训练器 - Tank Trouble")
    print("="*50)
    print(f"种群大小: {population_size}")
    print(f"训练代数: {generations}")
    print(f"每个体评估游戏数: {games_per_eval}")
    print(f"游戏速度: {speed}x")
    print("="*50)
    
    # 初始化
    ga = GeneticAlgorithm(population_size=population_size)
    env = GATrainingEnvironment(render=render, speed=speed)
    
    # 尝试加载检查点
    checkpoint_file = 'ga_checkpoint.pkl'
    if load_checkpoint and ga.load(checkpoint_file):
        print(f"从检查点继续训练，当前代数: {ga.generation}")
    
    try:
        for gen in range(ga.generation, generations):
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
            print(f"  平均适应度: {avg_fitness:.2f}, 最佳适应度: {best_fitness:.2f}")
            print(f"  历史最佳适应度: {ga.best_fitness:.2f}")
            
            # 进化
            ga.evolve()
            
            # 定期保存
            if (gen + 1) % save_interval == 0:
                ga.save(checkpoint_file)
        
        # 训练完成，保存最终结果
        ga.save('ga_final.pkl')
        print("\n训练完成！")
        
    except KeyboardInterrupt:
        print("\n训练被中断，保存当前进度...")
        ga.save(checkpoint_file)
    
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
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ga(
            generations=args.generations,
            population_size=args.population,
            games_per_eval=args.games,
            render=not args.no_render,
            load_checkpoint=True,
            speed=args.speed
        )
    elif args.mode == 'demo':
        demo_best(args.checkpoint)
