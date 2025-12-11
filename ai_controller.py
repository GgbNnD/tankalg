import numpy as np
import random
import copy
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        # Simple feed-forward
        x = np.array(x).reshape(1, -1)
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.tanh(self.z1) # Activation function
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = np.tanh(self.z2) # Output activation (-1 to 1 for movement)
        return self.a2[0]

    def get_weights(self):
        return {
            'w1': self.w1, 'b1': self.b1,
            'w2': self.w2, 'b2': self.b2
        }

    def set_weights(self, weights):
        self.w1 = weights['w1']
        self.b1 = weights['b1']
        self.w2 = weights['w2']
        self.b2 = weights['b2']

    def mutate(self, mutation_rate=0.1, mutation_scale=0.2):
        # Mutate weights
        for param in [self.w1, self.b1, self.w2, self.b2]:
            mask = np.random.random(param.shape) < mutation_rate
            mutation = np.random.randn(*param.shape) * mutation_scale
            param[mask] += mutation[mask]

class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size, output_size):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population = [NeuralNetwork(input_size, hidden_size, output_size) for _ in range(population_size)]
        self.generation = 0

    def select_parent(self, fitness_scores):
        # Tournament selection
        tournament_size = 3
        indices = np.random.choice(len(self.population), tournament_size, replace=False)
        best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return self.population[best_idx]

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        w1_1, w1_2 = parent1.get_weights(), parent2.get_weights()
        
        new_weights = {}
        for key in w1_1:
            # Randomly select genes from parents
            mask = np.random.random(w1_1[key].shape) > 0.5
            new_weights[key] = np.where(mask, w1_1[key], w1_2[key])
            
        child.set_weights(new_weights)
        return child

    def evolve(self, fitness_scores):
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        self.population = [self.population[i] for i in sorted_indices]
        
        new_population = []
        
        # Elitism: Keep the best few
        elite_count = max(2, int(self.population_size * 0.1))
        new_population.extend(copy.deepcopy(self.population[:elite_count]))
        
        # Generate rest
        while len(new_population) < self.population_size:
            parent1 = self.select_parent(fitness_scores)
            parent2 = self.select_parent(fitness_scores)
            child = self.crossover(parent1, parent2)
            child.mutate()
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1

    def save_best(self, filename="best_ai.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.population[0].get_weights(), f)

    def load_best(self, filename="best_ai.pkl"):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
            self.population[0].set_weights(weights)
