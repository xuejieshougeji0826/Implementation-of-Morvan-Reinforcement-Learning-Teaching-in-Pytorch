
"""
This is application of Reinforcement learning algorithm named Q_Learning
which is a maze in one dimensional space


"""


import numpy as np
import pandas as pd
import time

np.random.seed(2)

Numbers_States=6
Actions=['left','right']
Epsilon=0.9
Alpha=0.1
Lambda=0.9
Max_Episodes=20
fps=0.3



def build_Q_table(numbers_of_states,numbers_of_acitons):
    table=np.zeros((numbers_of_states,numbers_of_acitons),dtype=np.float)
    return table


def choose_action(state,table):
    actions=table[:state]
    randon_number=np.random.uniform()
    # print("all:",actions.all())
    # print(actions.all()==0.)
    # # randon_number =0.91
    # # print(randon_number)
    # print("actions",actions)
    if randon_number<Epsilon or actions.all():
        # print("random choice",np.random.choice(Actions))
        action_index=Actions.index(np.random.choice(Actions))
    else:
        action_index=np.argmax(actions)
    print(action_index)
    return action_index

def Get_Reward(S,A):
    A=Actions[A]
    if A=='right':
        if S==Numbers_States-2:
            S_='termianl'
            R=1
        else:
            S_=S+1
            R=0
    else:
        R=0
        if S==0:
            S_=S
        else:
            S_=S-1
    return S_,R

def Update_Env(S,episode,step_counter):

    env_list = ['-']*(Numbers_States-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(fps)

if __name__=='__main__':
    Q_table=build_Q_table(Numbers_States,len(Actions))

    Q_table[0][1]=0.2
    print(Q_table)
    choose_action(0,Q_table)
