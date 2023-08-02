import tkinter as tk
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque


CELL_SIZE = 10
GRID_WIDTH = 150
GRID_HEIGHT = 150

PLAYER_COLOR = ["blue", "red"]
TAIL_COLOR = ["#ADD8E6", "#FFA07A"]
TERRITORY_COLOR = ["#87CEFA", "#FFC0CB"]


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class Game:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(
            root, width=GRID_WIDTH * CELL_SIZE, height=GRID_HEIGHT * CELL_SIZE, bg="white")
        self.canvas.pack()
        self.canvas.focus_set()

        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=float)
        self.players = [Player(int(GRID_WIDTH*0.5), int(GRID_HEIGHT*0.25), id=1), Player(
            int(GRID_WIDTH*0.5), int(GRID_HEIGHT*0.75), id=2)]

        self.dqn = DQN(state_size=GRID_HEIGHT * GRID_WIDTH, action_size=4)
        self.batch_size = 32

    def update(self):
        state = self.grid.copy()
        for player in self.players:
            if not player.is_human:
                action = self.dqn.act(player.pov(state))
                if action == 0:
                    player.direction = (-1, 0)
                elif action == 1:
                    player.direction = (1, 0)
                elif action == 2:
                    player.direction = (0, -1)
                elif action == 3:
                    player.direction = (0, 1)
            self.grid, reward = player.move(state)
            self.dqn.remember(state, action, reward, )
            if player.dead:
                self.players = [Player(int(GRID_WIDTH*0.5), int(GRID_HEIGHT*0.25), id=1), Player(
                    int(GRID_WIDTH*0.5), int(GRID_HEIGHT*0.75), id=2)]
        self.draw()
        self.root.after(50, self.update)

    def refresh(self):
        for player in self.players:
            for x, y in player.territory:
                self.grid[y][x] = .3 + player.id
            self.grid[player.y][player.x] = .1 + player.id

    def draw(self):
        self.canvas.delete("all")
        for player in self.players:
            for y, x in np.argwhere(self.grid == player.id + .1):
                self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                             fill=PLAYER_COLOR[player.id - 1], outline="black")

            for y, x in np.argwhere(self.grid == player.id + .2):
                self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                             fill=TAIL_COLOR[player.id - 1], outline="")

            for y, x in np.argwhere(self.grid == player.id + .3):
                self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                             fill=TERRITORY_COLOR[player.id - 1], outline="")


class Player:
    def __init__(self, player_x, player_y, id=1, is_human=False) -> None:
        self.id = id
        self.x, self.y = player_x, player_y
        self.start_coords = (self.x, self.y)
        self.tail = [(self.x, self.y)]
        self.direction = (0, 1)  # y, x
        self.territory = self.initial_territory()
        self.dead = False
        self.is_human = is_human

    def initial_territory(self):
        territory = []
        for y in range(self.y - 2, self.y + 3):
            for x in range(self.x - 2, self.x + 3):
                territory.append((x, y))
        return territory

    def move(self, state):
        reward = 0
        if (self.id + .1) not in state.flatten():
            self.dead = True
            reward = -1000
            return state, reward

        new_x = self.x + self.direction[1]
        new_y = self.y + self.direction[0]

        if 0 <= new_x < state.shape[0] and 0 <= new_y < state.shape[1] and state[new_y][new_x] != self.id + .2:
            if state[new_y][new_x] == 0:
                if (self.x, self.y) not in self.territory:
                    self.tail.append((self.x, self.y))
                    state[self.y][self.x] = self.id + .2
                else:
                    state[self.y][self.x] = self.id + .3
            elif state[new_y][new_x] == self.id + .3:
                if self.tail:
                    state, reward = self.update_territory(state)
                state[self.y][self.x] = self.id + .3
                self.tail = []
            elif int(state[new_y][new_x]) != self.id and int(str(state[new_y][new_x]).split('.')[1]) == 2 or int(str(state[new_y][new_x]).split('.')[1]) == 1:
                state[np.where(state.astype('int') ==
                               int(state[new_y][new_x]))] = 0.
                state[self.y][self.x] = self.id + .2
                for x, y in self.territory:
                    state[y][x] = self.id + .3
                for x, y in self.tail:
                    state[y][x] = self.id + .2
            self.x = new_x
            self.y = new_y
            state[new_y][new_x] = self.id + .1
        else:
            self.dead = True
            reward = -1000
            return state, reward

        return state, reward

    def update_territory(self, state):
        for t in self.tail:
            self.territory.append(t)
        state[np.where(state == self.id + .2)] = self.id + .3
        self.tail = []

        return state, len(self.territory)

    def pov(self, state):
        state[np.where(state.astype('int') != self.id)] *= -1
        state[np.where(state < 0)] /= -state[np.where(state < 0)].astype('int')
        return state.astype(float)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("paper.io")
    game = Game(root)

    game.update()
    root.mainloop()
