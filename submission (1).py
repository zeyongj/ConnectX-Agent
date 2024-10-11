# Import necessary libraries
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from kaggle_environments import make, evaluate, utils
from collections import deque

# Define the neural network model for AlphaZero
class AlphaZeroModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(AlphaZeroModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, output_size)
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value

# Create the ConnectX environment
env = make("connectx", debug=True)
state_size = len(env.reset())
action_size = env.configuration.columns

# Initialize the AlphaZero model
model = AlphaZeroModel(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hyperparameters
GAMMA = 0.99
NUM_EPISODES = 1000
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64

# Replay buffer to store game experiences
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

# Function to perform Monte Carlo Tree Search (MCTS)
def mcts(observation, configuration, model, n_simulations=50):
    valid_actions = [c for c in range(configuration.columns) if observation['board'][c] == 0]
    action_visits = {action: 0 for action in valid_actions}
    action_values = {action: 0 for action in valid_actions}
    
    for _ in range(n_simulations):
        # Randomly simulate a game for each action
        for action in valid_actions:
            simulated_env = make("connectx", debug=True).train([None, "random"])
            simulated_observation = simulated_env.reset()
            _, reward, done, _ = simulated_env.step(action)
            if done:
                action_values[action] += reward
            else:
                board_state = np.array(simulated_observation.board, dtype=np.float32)
                board_tensor = torch.tensor(board_state).unsqueeze(0)
                _, value = model(board_tensor)
                action_values[action] += value.item()
            action_visits[action] += 1
    
    # Compute average value for each action
    best_action = max(valid_actions, key=lambda a: action_values[a] / (action_visits[a] + 1e-5))
    return best_action

# Train the AlphaZero model
for episode in range(NUM_EPISODES):
    trainer = env.train([None, "random"])
    observation = trainer.reset()
    done = False
    game_data = []
    
    while not done:
        # Use MCTS to decide on an action
        action = mcts(observation, env.configuration, model)
        next_observation, reward, done, _ = trainer.step(action)
        game_data.append((observation, action, reward, next_observation, done))
        observation = next_observation
    
    # Store game data in replay buffer
    for state, action, reward, next_state, done in game_data:
        replay_buffer.append((state, action, reward, next_state, done))
    
    # Train the model using replay buffer
    if len(replay_buffer) >= BATCH_SIZE:
        batch = random.sample(replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        policy, values = model(states)
        next_policy, next_values = model(next_states)

        target_values = rewards + GAMMA * next_values.squeeze() * (1 - dones)
        value_loss = nn.MSELoss()(values.squeeze(), target_values.detach())

        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

# Create an AlphaZero-based agent
def my_agent(observation, configuration):
    board_state = np.array(observation, dtype=np.float32)
    board_tensor = torch.tensor(board_state).unsqueeze(0)
    with torch.no_grad():
        policy, _ = model(board_tensor)
    valid_actions = [c for c in range(configuration.columns) if observation[c] == 0]
    policy_values = [(policy[0, c].item(), c) for c in valid_actions]
    return max(policy_values, key=lambda x: x[0])[1]

# Test the trained agent
trainer = env.train([None, "random"])
observation = trainer.reset()
done = False

while not done:
    action = my_agent(observation, env.configuration)
    observation, reward, done, _ = trainer.step(action)

# Render the game to see the performance
env.render(mode="ipython")
