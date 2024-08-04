import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random


class TaxiQLearning:
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1

    def train(self, episodes):
        # initialize the environment
        env = gym.make('Taxi-v3')
        # initialize the q-table
        q = np.zeros((env.observation_space.n, env.action_space.n))  # 500 possible states, 6 possible actions => 500 x 6
        # init arrays used for plotting
        mean_rewards = []
        epsilon_history = []
        rewards_per_episode = []

        for i in range(episodes):
            state = env.reset()
            done = False
            rewards = 0

            while not done:
                if random.random() < self.epsilon:
                    # pick a random action
                    action = env.action_space.sample()
                else:
                    # pick the most optimal action
                    action = np.argmax(q[state, :])

                # take a step
                new_state, reward, done, _ = env.step(action)

                # accumulate reward
                rewards += reward

                # new values in the q-table based on reward that we got
                q[state, action] = q[state, action] + self.learning_rate * (reward + self.discount_factor * np.max(q[new_state, :]) - q[state, action])

                # move to the next state
                state = new_state

            # decay epsilon
            self.epsilon = self.epsilon - 1 / episodes
            self.epsilon = max(self.epsilon, 0)

            # this is used for plotting
            rewards_per_episode.append(rewards)
            if i % 50 == 0 and i > 0:
                epsilon_history.append(self.epsilon)
                mean_rewards.append(np.mean(rewards_per_episode[-49]))

            if self.epsilon == 0:
                self.learning_rate = 0.0001

        env.close()

        self.save_q_table(q)

        self.plot_results(mean_rewards, epsilon_history)

    def test(self, episodes):
        env = gym.make('Taxi-v3', render_mode='human')

        q = self.load_q_table()
        for i in range(episodes):
            state = env.reset()  # states: 0 to 63, 0=top left corner,63=bottom right corner
            done = False
            while not done:
                # take only the most optimal actions
                action = np.argmax(q[state, :])
                state, reward, done, _ = env.step(action)
        env.close()

    def plot_results(self, rewards, epsilon_history):
        plt.clf()
        plt.xlabel("Episodes")
        plt.ylabel("Mean Rewards")
        plt.plot(rewards, color="orange", label="learning rate")
        plt.legend()
        plt.savefig("resources\\learning_curve.png")

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.clf()
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon Value")
        plt.plot(epsilon_history, color="orange", label="epsilon decay")
        plt.legend()
        plt.savefig("resources\\epsilon_decay.png")

    def save_q_table(self, q_table):
        f = open("resources\\taxi.pkl", "wb")
        pickle.dump(q_table, f)
        f.close()

    def load_q_table(self):
        f = open("resources\\taxi.pkl", "rb")
        q = pickle.load(f)
        f.close()
        return q


taxi = TaxiQLearning()
# taxi.train(10000)
taxi.test(200)
