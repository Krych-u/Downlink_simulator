import network
import user
import simulator
import block_allocation

import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, Activation, Flatten
from keras.callbacks import TensorBoard
from collections import deque
import time
from datetime import datetime

from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from tqdm import tqdm
import os

from tensorflow.tools.docs.doc_controls import T

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'DQL_Block_Allocation_4096x1024'
MIN_REWARD = 15_000  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20_000

# Exploration settings
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

ep_rewards = [0]

LOAD_MODEL = False
STARTING_TIME = datetime.now().strftime("%H_%M_%S")
LEARNING = True

if LEARNING:
    epsilon = 0.8  # Exploring state
else:
    epsilon = 0

class NetworkEnv:
    WRONG_ALLOCATION_PENALTY = 200
    CORRECT_ALLOCATION_REWARD = 200
    EMPTY_ALLOCATION_REWARD = -20
    SPACE_SIZE = 51
    OBSERVATION_SPACE = (2, 50)

    def reset(self, network):

        self.network = network
        self.users_list = network.users_list
        self.episode_step = 0

        observation = np.array([network.rb_throughputs, network.rb_throughputs])
        return observation

    def set_observation(self, user):
        return np.array([self.network.rb_throughputs, user.th_list])

    def step(self, user, action):
        self.episode_step += 1

        reward = 0
        if action in user.allocated_rb_list:
            reward = -self.WRONG_ALLOCATION_PENALTY
        old_throughput = self.network.return_sys_th()

        # If action == simulator.Simulator.RB_NUMBER means that we won't allocate rb for this user
        if action < simulator.Simulator.RB_NUMBER:
            reward += self.network.update_rb(user, action)

        new_observation = np.array([self.network.rb_throughputs, user.th_list])
        new_throughput = self.network.return_sys_th()

        if reward < 0:
            return new_observation, reward
        else:
            # Comparing allocation results to determine reward
            if new_throughput > old_throughput:
                reward = self.CORRECT_ALLOCATION_REWARD
            elif new_throughput == old_throughput:
                reward = self.EMPTY_ALLOCATION_REWARD
            else:
                reward = -self.WRONG_ALLOCATION_PENALTY

        return new_observation, reward


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'dir', 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    # Overrode, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrode
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


class DQNAAgent:
    def __init__(self, env):

        # main model -> gets trained every step
        self.env = env
        self.model = self.create_model()

        # target model this is what we .predict in every single step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque()
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):

        if LOAD_MODEL:

            model = load_model(LOAD_MODEL)
            print(f"Model loaded")

        else:
            model = Sequential()

            model.add(Conv1D(4096, 2,
                             input_shape=self.env.OBSERVATION_SPACE))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))

            model.add(Conv1D(1024, 1))
            model.add(Activation('relu'))
            model.add(Dropout(0.2))

            model.add(Flatten())
            model.add(Dense(512))

            model.add(Dense(self.env.SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (51)
            model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
         return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_states) in enumerate(minibatch):
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + DISCOUNT * max_future_q

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
