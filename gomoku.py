import matplotlib.pyplot as plt
import numpy as np

class Occupied(Exception):
    def __init__(self, x, y):
        print('Error: Board location (',x,',',y,') is already taken')
        exit(0)

class Gomoku:
    SIZE = 10
    OBSERVATION_SPACE = (SIZE, SIZE, 1)
    ACTION_SPACE = SIZE * SIZE
    MOVE_PENALTY = -1
    LOSS_PENALTY = -500
    WIN_REWARD = 1000
    INVALID_MOVE_PENALTY = -5000
    PLAYER0 = 0
    PLAYER1 = 1
    EMPTY_SPACE = -1

    def __init__(self):
        self.board = np.full((self.SIZE, self.SIZE), self.EMPTY_SPACE, dtype=int)
        self.winner = -69
        self.episode_step = 0
    
    def action_to_coord(self, action):
        y = action % self.SIZE
        x = int((action / self.SIZE) % self.SIZE)
        return x, y

    def step(self, player, action):
        self.episode_step += 1
        observation = self.get_board()

        x, y = self.action_to_coord(action)
        print('ACTION:', x,y)
        reward = self.place(player, x, y)

        if self.winner == player:
            done = True
            reward = WIN_REWARD
            return obervation, reward, done
        else:
            done = False
            return observation, reward, done
    
    def get_board(self):
        return self.board

    def show(self):
        plt.imshow(self.board)
        plt.show()

    def place(self, player, x, y):
        if self.board[y][x] == self.EMPTY_SPACE:
            self.board[y][x] = player
            return self.MOVE_PENALTY
        else:
            return self.INVALID_MOVE_PENALTY
            #raise Occupied(x, y)
    
    def reset(self):
        self.board = np.full((self.SIZE, self.SIZE), self.EMPTY_SPACE, dtype=int)
        self.winner = -69
        self.episode_step = 0

    def check_state(self):
        # check vertical
        for x in range(0, SIZE):
            for y in range(0, SIZE-4):
                if self.board[y][x] != EMPTY_SPACE:
                    curplayer = self.board[y][x]
                    score = 1
                    for j in range(4):
                        if self.board[y+j][x] == curplayer:
                            score+=1
                        else:
                            break

                        if score == 5:
                            print(curplayer, ' wins!')
                            return curplayer

        # check horizontal
        for x in range(0, SIZE-4):
            for y in range(0, SIZE-4):
                if self.board[y][x] != 0:
                    curplayer = self.board[y][x]
                    score = 1
                    for j in range(4):
                        if self.board[y][x+j] == curplayer:
                            score+=1
                        else:
                            break

                        if score == 5:
                            print(curplayer, ' wins!')
                            return curplayer
        
        # check diagonal \
        for x in range(0, SIZE-4):
            for y in range(0, SIZE-4):
                if self.board[y][x] != 0:
                    curplayer = self.board[y][x]
                    score = 1
                    for j in range(4):
                        if self.board[y+j][x+j] == curplayer:
                            score+=1
                        else:
                            break

                        if score == 5:
                            print(curplayer, ' wins!')
                            return curplayer
        
        # check diagonal /
        for x in range(0, SIZE-4):
            for y in reversed(range(4, SIZE)):
                if self.board[y][x] != 0:
                    curplayer = self.board[y][x]
                    score = 1
                    for j in range(4):
                        if self.board[y-j][x+j] == curplayer:
                            score+=1
                        else:
                            break

                        if score == 5:
                            print(curplayer, ' wins!')
                            return curplayer

        return False

    def print_state(self):
        print(self.board)