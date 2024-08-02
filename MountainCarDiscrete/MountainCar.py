import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)  # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions)  # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply rectified linear unit (ReLU) activation
        x = self.out(x)  # Calculate output
        return x


class ReplayBuffer:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


# MountainCar Deep Q-Learning
class MountainCarDQL:
    # Hyperparameters (adjustable)
    learning_rate_a = 0.01  # learning rate (alpha)
    discount_factor_g = 0.9  # discount rate (gamma)
    network_sync_rate = 50000  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 100000  # size of replay memory
    mini_batch_size = 32  # size of the training data set sampled from the replay memory
    position_space = None
    velocity_space = None
    num_divisions = 20

    # Neural Network
    loss_fn = nn.MSELoss()  # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None  # NN Optimizer. Initialize later.

    # Train the environment
    def train(self, episodes, render=False):
        # Create FrozenLake instance
        env = gym.make('MountainCar-v0', max_episode_steps=1000, render_mode='human' if render else None)
        num_states = env.observation_space.shape[0]  # expecting 2: position & velocity
        num_actions = env.action_space.n

        # Divide position and velocity into segments
        self.position_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)  # Between -1.2 and 0.6
        self.velocity_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)  # Between -0.07 and 0.07

        epsilon = 1  # 1 = 100% random actions
        memory = ReplayBuffer(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = []

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0
        best_rewards = -1000

        for i in range(episodes):
            state = env.reset()  # Initialize to state 0
            done = False
            rewards = 0

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while not done and rewards > -1000:
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()  # actions: 0=left,1=idle,2=right
                else:
                    # select best action
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                new_state, reward, done, _ = env.step(action)

                # Accumulate reward
                rewards += reward

                # Save experience into memory
                memory.append((state, action, new_state, reward, done))

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(rewards)

            # Graph training progress
            if i != 0 and i % 1000 == 0:
                print(f'Episode {i} Epsilon {epsilon}')
                print(f'Best rewards so far: {best_rewards}')
                self.plot_progress(rewards_per_episode, epsilon_history)

            if rewards > best_rewards:
                best_rewards = rewards
                # Save policy
                torch.save(policy_dqn.state_dict(), f"resources\\mountaincar_dql_{i}.pt")

            # Check if enough experience has been collected
            if len(memory) > self.mini_batch_size and best_rewards > -1000:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        # Close environment
        env.close()

    def plot_progress(self, rewards_per_episode, epsilon_history):
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        plt.clf()
        plt.xlabel("Episodes")
        plt.ylabel("Learning rate")
        plt.plot(rewards_per_episode, color="orange", label="learning rate")
        plt.legend()
        plt.savefig("resources\\learning_curve.png")

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.clf()
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon Value")
        plt.plot(epsilon_history, color="orange", label="epsilon decay")
        plt.legend()
        plt.savefig("resources\\epsilon_decay.png")

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                # Agent receive reward of 0 for reaching goal.
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state))
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def state_to_dqn_input(self, state) -> torch.Tensor:
        state_p = np.digitize(state[0], self.position_space)
        state_v = np.digitize(state[1], self.velocity_space)

        return torch.FloatTensor([state_p, state_v])

    # Run the environment with the learned policy
    def test(self, episodes, model_filepath):
        # Create FrozenLake instance
        env = gym.make('MountainCar-v0', render_mode='human')
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        self.position_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)  # Between -1.2 and 0.6
        self.velocity_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)  # Between -0.07 and 0.07

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=10, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load(model_filepath))
        policy_dqn.eval()  # switch model to evaluation mode

        for i in range(episodes):
            state = env.reset()  # Initialize to state 0
            done = False

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while not done:
                # Select best action
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                state, reward, done, _ = env.step(action)

        env.close()


mountain_car = MountainCarDQL()
# mountain_car.train(25_000)
mountain_car.test(10, "resources\\mountaincar_dql_20708.pt")
