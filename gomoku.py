import matplotlib.pyplot as plt
import numpy as np


class Occupied(Exception):
    def __init__(self, x, y):
        print('Error: Board location (',x,',',y,') is already taken')
        exit(0)

class Board:
    def __init__(self, n):
        self.n = n
        self.board = np.zeros((n, n), dtype=int)
        self.winner = None
    
    def place(self, player, x, y):
        if self.board[y][x] == 0:
            self.board[y][x] = player
        else:
            raise Occupied(x, y)
    
    def check_state(self):
        # check vertical
        for x in range(0, self.n-4):
            for y in range(0, self.n-4):
                if self.board[y][x] != 0:
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
        for x in range(0, self.n-4):
            for y in range(0, self.n-4):
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
        for x in range(0, self.n-4):
            for y in range(0, self.n-4):
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
        for x in range(0, self.n-4):
            for y in reversed(range(4, self.n)):
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

def main():
    n = 10
    game = Board(n)
    for i in range(10):
        game.place(1,9-i,i)

    print(game.check_state())
    game.print_state()


if __name__ == '__main__':
    main()