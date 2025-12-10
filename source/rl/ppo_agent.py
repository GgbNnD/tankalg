import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# 自动检测设备：优先使用 GPU (CUDA)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"PPO Agent using device: {device} ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("PPO Agent using device: cpu")
    print("Warning: CUDA not found. Training will be slow. Please install PyTorch with CUDA support.")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Actor head
        self.actor = nn.Linear(128, action_dim)
        
        # Critic head
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        action_logits = self.actor(x)
        state_value = self.critic(x)
        
        return action_logits, state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            action_logits, _ = self.policy_old(state)
            
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action).item()
    
    def update(self, memory):
        # Convert memory to tensors
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            logprobs, state_values = self.policy(old_states)
            dist = Categorical(logits=logprobs)
            
            action_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(action_logprobs - old_logprobs)
            
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
