
"""
This is application of Reinforcement learning algorithm named Q_Learning
which is a maze in one dimensional space


"""


import numpy as np
import pandas as pd
import time

np.random.seed(2)

Numbers_States=16
Actions=['left','right']
Epsilon=0.9
Alpha=0.1
Lambda=0.9
Max_Episodes=10
fps=0.05


def build_Q_table(numbers_of_states,numbers_of_acitons):
    table=np.zeros((numbers_of_states,numbers_of_acitons),dtype=np.float)
    return table


def choose_action(state,table):
    actions=table[state]
    # print(actions)
    randon_number=np.random.uniform()
    # print("all:",actions.all())
    # print(actions.all()==0.)
    # # randon_number =0.91
    # # print(randon_number)
    # print("actions",actions)
    # print(randon_number)
    if randon_number>Epsilon or actions.all()==0:
        # print("random choice",np.random.choice(Actions))
        action_index=Actions.index(np.random.choice(Actions))
        # print("actions:",action_index)
    else:
        action_index=np.argmax(actions)
        # print(actions)
    # print(action_index)

    # print(Actions[action_index])
    return action_index

def Get_Reward(S,A):
    A=Actions[A]
    if A=='right':
        if S==Numbers_States-2:
            S_='terminal'
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

    env_list = ['-']*(Numbers_States-1) + ['T']
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
    #
    # Q_table[0][1]=0.2
    # print(Q_table)
    for epsilon in range(Max_Episodes):
        step_counter=0
        S=0
        is_done=False
        Update_Env(S,epsilon,step_counter)
        while not is_done:
            # print("s:",S)
            A=choose_action(S,Q_table)
            # print("A:",A)
            S_,R=Get_Reward(S,A)
            predict=Q_table[S,A]
            if S_ != 'terminal':

                a=Lambda * Q_table[int(S_), :].max()
                # print(a)
                q_target = R + a
            else:
                q_target = R
                is_done = True
            Q_table[S][A]+=Alpha*(q_target-predict)
            # print('\r',Q_table,end='',flush=True)
            S=S_
            Update_Env(S,epsilon,step_counter+1)

    # S_