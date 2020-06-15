from agent import *
from gomoku import *


def play():
    # play with human
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("models/3_dim_10000_rounds")

    p2 = HumanPlayer("human")

    st = State(p1, p2)
    st.play2()



if __name__=='__main__':
    play()
