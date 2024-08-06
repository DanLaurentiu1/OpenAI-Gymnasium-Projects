# for taking screenshots
from mss import mss
# for making the agent input actual commands => duck, space
import pydirectinput
# image down-scaling and grey-scaling
import cv2
# numpy
import numpy as np
# wait 1 second before resetting the env
import time
# for the environment template
from gym import Env
from gym.spaces import Box, Discrete
# file path management
import os
# saving models
from stable_baselines3.common.callbacks import BaseCallback
# rl algorithm
from stable_baselines3 import DQN

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'


class ChromeDinoGame(Env):
    def __init__(self):
        super().__init__()
        # observation and action space
        self.observation_space = Box(low=0, high=255, shape=(1, 80, 100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # screen-capture
        self.capture = mss()
        # where the screen-capture will be for the game
        self.game_location = {'top': 300, 'left': 0, 'width': 600, 'height': 500}
        # where the screen-capture will be for the done message when the game ends
        self.done_location = {'top': 451, 'left': 573, 'width': 20, 'height': 5}
        # action map
        self.action_map = {0: 'space', 1: 'down', 2: 'idle'}

    def step(self, action):
        reward = 0

        if action != 2:
            pydirectinput.press(self.action_map[action])

        # idle => reward + 2, jump => reward + 1, duck => reward + 1
        # this will help in making the agent's movements more deliberate
        if action == 2:
            reward = 2
        elif action == 1:
            reward = 1
        elif action == 0:
            reward = 1

        done = self.get_done()
        observation = self.get_observation()
        return observation, reward, done, {}

    def reset(self):
        # after the game is over, wait 1 second
        time.sleep(1)
        # left-click
        pydirectinput.click(x=150, y=150)
        # press space
        pydirectinput.press('space')
        # return the first state
        return self.get_observation()

    def render(self):
        cv2.imshow('Game', np.array(self.capture.grab(self.game_location))[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self):
        # take screen-capture and return 3 channels instead of 4
        raw = np.array(self.capture.grab(self.game_location))[:, :, :3].astype(np.uint8)
        # gray-scale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # make the capture's shape be the same as the expected shape (down-scaling)
        resized = cv2.resize(gray, (100, 80))
        channel = np.reshape(resized, (1, 80, 100))
        return channel

    def get_done(self):
        # take screen-capture
        done_capture = np.array(self.capture.grab(self.done_location))
        # sum the pixel values
        sum_of_pixel_values = np.sum(done_capture)
        done = False
        # if the sum is a specific number, the game is over
        if sum_of_pixel_values == 77100:
            done = True
        return done


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


class TrainingTestingClass:
    # environment
    env = ChromeDinoGame()
    # callback
    callback = TrainAndLoggingCallback(check_freq=500, save_path=CHECKPOINT_DIR)
    # model
    model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=300_000, learning_starts=1_000)

    def train(self, steps):
        self.model.learn(total_timesteps=steps, callback=self.callback)

    def test(self, steps):
        self.load("C:\\Users\\Lau\\PycharmProjects\\ReinforcementLearningProjects\\ChromeDinoGame\\train\\best_model_109500.zip")
        for i in range(steps):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.model.predict(state)
                state, reward, done, _ = self.env.step(int(action))
                total_reward += reward
            print(f"Episode: {i}, reward: {total_reward}")

    def load(self, path):
        self.model = DQN.load(path)


if __name__ == '__main__':
    dino_game = TrainingTestingClass()
    # dino_game.train(110_000)
    dino_game.test(25)
