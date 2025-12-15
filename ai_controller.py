import numpy as np
import random
import copy
import pickle
import torch

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        # Initialize weights and biases
        # Layer 1: Input -> Hidden1
        self.w1 = np.random.randn(input_size, hidden_size1)
        self.b1 = np.zeros((1, hidden_size1))
        # Layer 2: Hidden1 -> Hidden2
        self.w2 = np.random.randn(hidden_size1, hidden_size2)
        self.b2 = np.zeros((1, hidden_size2))
        # Layer 3: Hidden2 -> Output
        self.w3 = np.random.randn(hidden_size2, output_size)
        self.b3 = np.zeros((1, output_size))

    def forward(self, x):
        # Simple feed-forward
        x = np.array(x).reshape(1, -1)
        
        # Layer 1
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.tanh(self.z1) 
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = np.tanh(self.z2)
        
        # Layer 3
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = np.tanh(self.z3) # Output activation (-1 to 1 for movement)
        
        return self.a3[0]

    def get_weights(self):
        return {
            'w1': self.w1, 'b1': self.b1,
            'w2': self.w2, 'b2': self.b2,
            'w3': self.w3, 'b3': self.b3
        }

    def set_weights(self, weights):
        if 'w3' in weights:
            self.w1 = weights['w1']
            self.b1 = weights['b1']
            self.w2 = weights['w2']
            self.b2 = weights['b2']
            self.w3 = weights['w3']
            self.b3 = weights['b3']
        else:
            print("Warning: Loading old model. Adapting weights...")
            self.w1 = weights['w1']
            self.b1 = weights['b1']
            # Map old output weights to new output layer
            # Assuming hidden_size2 == output_size of old w2 (which is hidden_size of old model)
            # Wait, old w2 is (hidden, output). New w3 is (hidden2, output).
            # If hidden == hidden2, shapes match.
            try:
                self.w3 = weights['w2']
                self.b3 = weights['b2']
            except:
                print("Shape mismatch in adaptation, keeping random weights for output layer.")
            # w2/b2 remain random (initialized in __init__)

    def mutate(self, mutation_rate=0.1, mutation_scale=0.2):
        # Mutate weights
        for param in [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]:
            mask = np.random.random(param.shape) < mutation_rate
            mutation = np.random.randn(*param.shape) * mutation_scale
            param[mask] += mutation[mask]

class PopulationNetwork:
    def __init__(self, pop_size, input_size, hidden_size1, hidden_size2, output_size, device='cuda'):
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.device = device
        
        # Initialize weights on GPU
        # Shape: (pop_size, input_size, hidden_size1)
        self.w1 = torch.randn(pop_size, input_size, hidden_size1, device=device)
        self.b1 = torch.zeros(pop_size, 1, hidden_size1, device=device)
        # Shape: (pop_size, hidden_size1, hidden_size2)
        self.w2 = torch.randn(pop_size, hidden_size1, hidden_size2, device=device)
        self.b2 = torch.zeros(pop_size, 1, hidden_size2, device=device)
        # Shape: (pop_size, hidden_size2, output_size)
        self.w3 = torch.randn(pop_size, hidden_size2, output_size, device=device)
        self.b3 = torch.zeros(pop_size, 1, output_size, device=device)
        
    def forward(self, x):
        # x shape: (pop_size, input_size)
        # We need to perform batch matrix multiplication
        # x.unsqueeze(1) -> (pop_size, 1, input_size)
        # w1 -> (pop_size, input_size, hidden_size)
        # bmm -> (pop_size, 1, hidden_size)
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
        x = x.unsqueeze(1)
        z1 = torch.bmm(x, self.w1) + self.b1
        a1 = torch.tanh(z1)
        z2 = torch.bmm(a1, self.w2) + self.b2
        a2 = torch.tanh(z2)
        z3 = torch.bmm(a2, self.w3) + self.b3
        a3 = torch.tanh(z3)
        
        # Return shape: (pop_size, output_size)
        return a3.squeeze(1)

    def forward_subset(self, x, indices):
        # x shape: (batch_size, input_size)
        # indices: list or tensor of indices to use
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
            
        x = x.unsqueeze(1)
        
        # Slice weights
        w1 = self.w1[indices]
        b1 = self.b1[indices]
        w2 = self.w2[indices]
        b2 = self.b2[indices]
        w3 = self.w3[indices]
        b3 = self.b3[indices]
        
        z1 = torch.bmm(x, w1) + b1
        a1 = torch.tanh(z1)
        z2 = torch.bmm(a1, w2) + b2
        a2 = torch.tanh(z2)
        z3 = torch.bmm(a2, w3) + b3
        a3 = torch.tanh(z3)
        
        return a3.squeeze(1)

    def get_best_weights(self, idx):
        # Return weights for a single individual as numpy dict (compatible with NeuralNetwork)
        return {
            'w1': self.w1[idx].cpu().numpy(),
            'b1': self.b1[idx].cpu().numpy(),
            'w2': self.w2[idx].cpu().numpy(),
            'b2': self.b2[idx].cpu().numpy(),
            'w3': self.w3[idx].cpu().numpy(),
            'b3': self.b3[idx].cpu().numpy()
        }
        
    def set_weights_from_numpy(self, idx, weights):
        self.w1[idx] = torch.tensor(weights['w1'], device=self.device)
        self.b1[idx] = torch.tensor(weights['b1'], device=self.device)
        
        if 'w3' in weights:
            self.w2[idx] = torch.tensor(weights['w2'], device=self.device)
            self.b2[idx] = torch.tensor(weights['b2'], device=self.device)
            self.w3[idx] = torch.tensor(weights['w3'], device=self.device)
            self.b3[idx] = torch.tensor(weights['b3'], device=self.device)
        else:
            # Old model
            try:
                self.w3[idx] = torch.tensor(weights['w2'], device=self.device)
                self.b3[idx] = torch.tensor(weights['b2'], device=self.device)
            except:
                pass
            # w2/b2 remain random

class GPUGeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size1, hidden_size2, output_size, device='cuda'):
        self.pop_size = population_size
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.device = device
        self.generation = 0
        
        self.pop = PopulationNetwork(population_size, input_size, hidden_size1, hidden_size2, output_size, device)

    def evolve(self, fitness_scores):
        # fitness_scores: numpy array of shape (pop_size,)
        
        # Sort indices
        sorted_indices = np.argsort(fitness_scores)[::-1].copy()
        
        # Elitism: Keep top 10%
        elite_count = max(2, int(self.pop_size * 0.1))
        elite_indices = sorted_indices[:elite_count]
        
        # Create new population tensors
        new_w1 = torch.empty_like(self.pop.w1)
        new_b1 = torch.empty_like(self.pop.b1)
        new_w2 = torch.empty_like(self.pop.w2)
        new_b2 = torch.empty_like(self.pop.b2)
        new_w3 = torch.empty_like(self.pop.w3)
        new_b3 = torch.empty_like(self.pop.b3)
        
        # Copy elites
        # We can use advanced indexing
        elite_indices_tensor = torch.tensor(elite_indices, device=self.device)
        new_w1[:elite_count] = self.pop.w1[elite_indices_tensor]
        new_b1[:elite_count] = self.pop.b1[elite_indices_tensor]
        new_w2[:elite_count] = self.pop.w2[elite_indices_tensor]
        new_b2[:elite_count] = self.pop.b2[elite_indices_tensor]
        new_w3[:elite_count] = self.pop.w3[elite_indices_tensor]
        new_b3[:elite_count] = self.pop.b3[elite_indices_tensor]
        
        # Generate offspring
        # Tournament selection for the rest
        num_children = self.pop_size - elite_count
        
        # Vectorized tournament selection
        # We need to select 'num_children' pairs of parents
        # Let's do it on CPU for simplicity of logic, then index GPU tensors
        
        parent1_indices = []
        parent2_indices = []
        
        for _ in range(num_children):
            # Tournament 1
            candidates = np.random.choice(self.pop_size, 3, replace=False)
            p1 = candidates[np.argmax(fitness_scores[candidates])]
            parent1_indices.append(p1)
            
            # Tournament 2
            candidates = np.random.choice(self.pop_size, 3, replace=False)
            p2 = candidates[np.argmax(fitness_scores[candidates])]
            parent2_indices.append(p2)
            
        p1_tensor = torch.tensor(parent1_indices, device=self.device)
        p2_tensor = torch.tensor(parent2_indices, device=self.device)
        
        # Crossover (Uniform)
        # Generate masks
        # w1 shape: (num_children, input, hidden)
        w1_p1 = self.pop.w1[p1_tensor]
        w1_p2 = self.pop.w1[p2_tensor]
        mask_w1 = (torch.rand_like(w1_p1) > 0.5)
        child_w1 = torch.where(mask_w1, w1_p1, w1_p2)
        
        b1_p1 = self.pop.b1[p1_tensor]
        b1_p2 = self.pop.b1[p2_tensor]
        mask_b1 = (torch.rand_like(b1_p1) > 0.5)
        child_b1 = torch.where(mask_b1, b1_p1, b1_p2)
        
        w2_p1 = self.pop.w2[p1_tensor]
        w2_p2 = self.pop.w2[p2_tensor]
        mask_w2 = (torch.rand_like(w2_p1) > 0.5)
        child_w2 = torch.where(mask_w2, w2_p1, w2_p2)
        
        b2_p1 = self.pop.b2[p1_tensor]
        b2_p2 = self.pop.b2[p2_tensor]
        mask_b2 = (torch.rand_like(b2_p1) > 0.5)
        child_b2 = torch.where(mask_b2, b2_p1, b2_p2)

        w3_p1 = self.pop.w3[p1_tensor]
        w3_p2 = self.pop.w3[p2_tensor]
        mask_w3 = (torch.rand_like(w3_p1) > 0.5)
        child_w3 = torch.where(mask_w3, w3_p1, w3_p2)
        
        b3_p1 = self.pop.b3[p1_tensor]
        b3_p2 = self.pop.b3[p2_tensor]
        mask_b3 = (torch.rand_like(b3_p1) > 0.5)
        child_b3 = torch.where(mask_b3, b3_p1, b3_p2)
        
        # Mutation
        mutation_rate = 0.1
        mutation_scale = 0.2
        
        def mutate_tensor(t):
            mask = torch.rand_like(t) < mutation_rate
            noise = torch.randn_like(t) * mutation_scale
            t[mask] += noise[mask]
            return t
            
        child_w1 = mutate_tensor(child_w1)
        child_b1 = mutate_tensor(child_b1)
        child_w2 = mutate_tensor(child_w2)
        child_b2 = mutate_tensor(child_b2)
        child_w3 = mutate_tensor(child_w3)
        child_b3 = mutate_tensor(child_b3)
        
        # Fill new population
        new_w1[elite_count:] = child_w1
        new_b1[elite_count:] = child_b1
        new_w2[elite_count:] = child_w2
        new_b2[elite_count:] = child_b2
        new_w3[elite_count:] = child_w3
        new_b3[elite_count:] = child_b3
        
        # Update population
        self.pop.w1 = new_w1
        self.pop.b1 = new_b1
        self.pop.w2 = new_w2
        self.pop.b2 = new_b2
        self.pop.w3 = new_w3
        self.pop.b3 = new_b3
        
        self.generation += 1

    def save_best(self, filename="best_ai.pkl"):
        # Save the 0-th individual (which is the best from previous gen due to sorting? 
        # Wait, we sorted indices but we constructed new population with elites at 0.
        # So yes, index 0 is an elite.
        # But we need to be careful. In evolve, we put elites at 0.
        # So after evolve, 0 is one of the best from PREVIOUS generation.
        # It might not be the absolute best if we didn't sort by fitness before saving.
        # Usually we save BEFORE evolve.
        # But here we assume index 0 is good enough or we pass the best index.
        
        # Actually, let's just save index 0.
        with open(filename, 'wb') as f:
            pickle.dump(self.pop.get_best_weights(0), f)

    def load_best(self, filename="best_ai.pkl"):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
            # Set to index 0
            self.pop.set_weights_from_numpy(0, weights)

class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size1, hidden_size2, output_size):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.population = [NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size) for _ in range(population_size)]
        self.generation = 0

    def select_parent(self, fitness_scores):
        # Tournament selection
        tournament_size = 3
        indices = np.random.choice(len(self.population), tournament_size, replace=False)
        best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return self.population[best_idx]

    def crossover(self, parent1, parent2):
        child = NeuralNetwork(self.input_size, self.hidden_size1, self.hidden_size2, self.output_size)
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
