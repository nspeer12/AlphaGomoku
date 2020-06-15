import numpy as np
import pickle
from tqdm import tqdm

N = 3
IN_A_ROW = 3
EMPTY_SPACE = 0

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((N, N))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        self.winner = None

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(N * N))
        return self.boardHash
    
    def check_state(self, print_game=False):
        
        # check vertical
        for j in range(0, N):
            for i in range(0, N - IN_A_ROW):
                if self.board[i][j] != EMPTY_SPACE:
                    curplayer = self.board[i][j]
                    score = 1
                    for k in range(i+1, i+IN_A_ROW):
                        if self.board[k][j] == curplayer:
                            score+=1
                        else:
                            break

                        if score == IN_A_ROW:
                            if print_game:
                                print(curplayer, ' wins!')
                            return curplayer

        # check horizontal
        for i in range(0, N):
            for j in range(0, N - IN_A_ROW):
                if self.board[i][j] != EMPTY_SPACE:
                    curplayer = self.board[i][j]
                    score = 1
                    for k in range(j+1, j+IN_A_ROW):
                        if self.board[i][k] == curplayer:
                            score+=1
                        else:
                            break

                        if score == IN_A_ROW:
                            if print_game:
                                print(curplayer, ' wins!')
                            return curplayer

        # check diagonal \
        for i in range(0, N - IN_A_ROW):
            for j in range(0, N - IN_A_ROW):
                if self.board[i][j] != EMPTY_SPACE:
                    curplayer = self.board[i][j]
                    score = 1
                    for k in range(1, IN_A_ROW):
                        if self.board[j+k][i+k] == curplayer:
                            score+=1
                        else:
                            break

                        if score == IN_A_ROW:
                            if print_game:
                                print(curplayer, ' wins!')
                            return curplayer

        # check diagonal /
        for j in range(0, N - IN_A_ROW):
            for i in reversed(range(IN_A_ROW, N)):
                if self.board[i][j] != EMPTY_SPACE:
                    curplayer = self.board[i][j]
                    score = 1
                    for k in range(1, IN_A_ROW):
                        if self.board[i-k][j+k] == curplayer:
                            score+=1
                        else:
                            break

                        if score == IN_A_ROW:
                            if print_game:
                                print(curplayer, ' wins!')
                            return curplayer
            
        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(N):
            for j in range(N):
                if self.board[i, j] == EMPTY_SPACE:
                    positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.check_state()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        self.board = np.zeros((N, N))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        for i in tqdm(range(rounds)):
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.check_state()
                if win is not None:
                    # self.showBoard()
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.check_state()
                    if win is not None:
                        # self.showBoard()
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    # play with human
    def play2(self):
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateState(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.check_state()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions)

                self.updateState(p2_action)
                self.showBoard()
                win = self.check_state()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self):
        # p1: x  p2: o
        for k in range(N):
            if k == 0:
                print('   ', end='')
            print(' ', k, '', end='' if k != N-1 else '\n')
            
        for k in range(N):
            if k == 0:
                print('   ', end='')
            print('----', end='' if k != N-1 else '-\n')
        
        for i in range(0, N):
            print(i, ' ', end='')
            out = '| '
            for j in range(0, N):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
            for k in range(N):
                if k == 0:
                    print('   ', end='')
                print('----', end='' if k != N-1 else '-\n')
