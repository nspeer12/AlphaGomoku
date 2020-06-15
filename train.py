import sys
from agent import *
from gomoku import *

def train(rounds):
    # training
    p1 = Player("p1")
    p2 = Player("p2")

    st = State(p1, p2)
    print("training...")
    st.play(int(rounds))
    p1.savePolicy(rounds)
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train(sys.argv[1])
    else:
        print('Usage: python train.py < # rounds >')


