import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from modifiedtfboard import ModifiedTensorBoard
from collections import deque
import time

REPALY_MEMORY_SIZE = 50000
MODEL_NAME = "JIMBOT"


class Agent:
    def __init__(self):
        # main model
        # train every step
        self.model = self.create_model()

        # target model
        # this is what we predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # part of the batching process
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential
        model.add(Conv2D(64, (3,3), input_shape=OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3,3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(env.ACTION_SPACE_SIZE), activation="linear")
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    def get_qs(self, state, step):
        # this is used for rgb image data, perhaps remove the /255
        return self.model_predict(np.array(state).reshape(-1, *state.shape)/255)[0]
