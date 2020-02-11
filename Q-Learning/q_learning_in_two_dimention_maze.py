"""

This .py file make a complete algorithm(Q-Learning) to play two-dimension maze

"""
import tkinter as tk
import numpy as np
import time

np.set_printoptions(suppress = True)

# np.random.seed(1)
unit = 40
height1 = 6
weight1 = 6

original_position = np.array([20, 20])


# Environment (two-dimension maze)

class Two_Dimension_Maze(tk.Tk):
    def __init__(self):
        super(Two_Dimension_Maze, self).__init__()
        self.actions = ['up', 'down', 'left', 'right']
        self.num_actions = len(self.actions)
        self.title('Two_Dimension_Maze')
        self.geometry('%dx%d' % (unit * height1, unit * weight1))
        self.build_maze()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.background.delete(self.agent_rectangle)

        self.agent_rectangle = self.background.create_rectangle(
            self.agent_position[0] - 10, self.agent_position[1] - 10,
            self.agent_position[0] + 10, self.agent_position[1] + 10, fill='Yellow')
        # return observation
        return self.background.coords(self.agent_rectangle)

    def step(self, action):
        s = self.background.coords(self.agent_rectangle)
        # print(s)
        base_move = np.array([0, 0])
        if action == 'up':
            if s[1] > unit:
                base_move[1] -= unit

        elif action == 'down':
            if s[1] < unit * (height1 - 1):
                base_move[1] += unit
        elif action == 'left':
            if s[0] > unit:
                base_move[0] -= unit
        elif action=='right':
            if s[0] < unit * (weight1 - 1):
                base_move[0] += unit
        # print(base_move)
        self.background.move(self.agent_rectangle, base_move[0], base_move[1])
        # print(self.agent_rectangle)
        s_next = self.background.coords(self.agent_rectangle)
        # print("s_begin",s)
        # print("s_next:",s_next)
        if s_next == self.background.coords(self.target):
            reward = 50
            done = True
        elif s_next in [self.background.coords(self.hell1), self.background.coords(self.hell2),]:
                        # self.background.coords(self.hell3), self.background.coords(self.hell4),
                        # self.background.coords(self.hell5), self.background.coords(self.hell6),
                        # self.background.coords(self.hell7) or self.background.coords(self.hell8),
                        # self.background.coords(self.hell9)
            reward = -50
            done = True
        else:
            reward = 0
            done = False
        # print("s_next:",s_next,"reward:",reward)
        # print("s_next", s_next)
        return done, s_next, reward

    def render(self):
        self.update()

    def build_maze(self):
        self.background = tk.Canvas(self, bg='white', height=unit * height1, width=unit * weight1)

        for i in range(0, unit * height1, unit):
            x1, y1, x2, y2 = i, 0, i, unit * height1
            self.background.create_line(x1, y1, x2, y2)
        for i in range(0, unit * weight1, unit):
            x1, y1, x2, y2 = 0, i, unit * height1, i
            self.background.create_line(x1, y1, x2, y2)

        hell1_center = original_position + np.array([unit * 2, unit])
        hell2_center = original_position + np.array([unit * 2, unit * 3])
        hell3_center = original_position + np.array([unit * 5, unit * 3])
        hell4_center = original_position + np.array([unit * 4, unit * 4])
        hell5_center = original_position + np.array([unit * 6, unit * 1])
        hell6_center = original_position + np.array([unit * 1, unit * 5])
        hell7_center = original_position + np.array([unit * 6, unit * 5])
        hell8_center = original_position + np.array([unit * 6, unit * 7])
        hell9_center = original_position + np.array([unit * 1, unit * 7])

        target_center = original_position + np.array([unit * weight1 - unit, unit * height1 - unit])

        self.agent_position = original_position

        self.agent_rectangle = self.background.create_rectangle(
            self.agent_position[0] - 10, self.agent_position[1] - 10,
            self.agent_position[0] + 10, self.agent_position[1] + 10, fill='Yellow')

        self.hell1 = self.background.create_rectangle(
            hell1_center[0] - 10, hell1_center[1] - 10,
            hell1_center[0] + 10, hell1_center[1] + 10,
            fill='DarkOliveGreen')
        self.hell2 = self.background.create_rectangle(
            hell2_center[0] - 10, hell2_center[1] - 10,
            hell2_center[0] + 10, hell2_center[1] + 10,
            fill='DarkOliveGreen')
        # self.hell3 = self.background.create_rectangle(
        #     hell3_center[0] - 10, hell3_center[1] - 10,
        #     hell3_center[0] + 10, hell3_center[1] + 10,
        #     fill='DarkOliveGreen')
        # self.hell4 = self.background.create_rectangle(
        #     hell4_center[0] - 10, hell4_center[1] - 10,
        #     hell4_center[0] + 10, hell4_center[1] + 10,
        #     fill='DarkOliveGreen')
        # self.hell5 = self.background.create_rectangle(
        #     hell5_center[0] - 10, hell5_center[1] - 10,
        #     hell5_center[0] + 10, hell5_center[1] + 10,
        #     fill='DarkOliveGreen')
        # self.hell6 = self.background.create_rectangle(
        #     hell6_center[0] - 10, hell6_center[1] - 10,
        #     hell6_center[0] + 10, hell6_center[1] + 10,
        #     fill='DarkOliveGreen')
        # self.hell7 = self.background.create_rectangle(
        #     hell7_center[0] - 10, hell7_center[1] - 10,
        #     hell7_center[0] + 10, hell7_center[1] + 10,
        #     fill='DarkOliveGreen')
        # self.hell8 = self.background.create_rectangle(
        #     hell8_center[0] - 10, hell8_center[1] - 10,
        #     hell8_center[0] + 10, hell8_center[1] + 10,
        #     fill='DarkOliveGreen')
        # self.hell9 = self.background.create_rectangle(
        #     hell9_center[0] - 10, hell9_center[1] - 10,
        #     hell9_center[0] + 10, hell9_center[1] + 10,
        #     fill='DarkOliveGreen')

        self.target = self.background.create_rectangle(
            target_center[0] - 10, target_center[1] - 10,
            target_center[0] + 10, target_center[1] + 10,
            fill='DarkRed')

        self.background.pack()

    def test(self):

        for i in range(len(self.actions) - 1, -1, -1):
            env.render()
            # print(i)
            s_, r, done = self.step(self.actions[i])
            time.sleep(1)
        self.reset()
        for i in range(len(self.actions) - 1, -1, -1):
            env.render()
            # print(i)
            s_, r, done = self.step(self.actions[i])
            time.sleep(1)


# Algorithm (Q-L)

class Q_Learning:
    def __init__(self, actions, lr=0.01, Gmma=0.9, Epsilon_greddy=0.8):
        self.actions = actions
        self.lr = lr
        self.Gmma = Gmma
        self.Epsilon_greddy = Epsilon_greddy
        self.Q_table = np.zeros((1, 2, self.actions), np.float)


    def check_if_state_exist(self, state):

        pool=self.Q_table[:,0]

        index=0
        for i in range(len(pool)):

            if( state != pool[i]).any():
                pass
            else:
                index=i
                break

        if index==0:

            self.Q_table = np.insert(self.Q_table, len(self.Q_table), values=np.zeros((1, 2, self.actions), np.float), axis=0)

            self.Q_table[-1][0]=state
            flag=len(self.Q_table)-1
        else:flag=index

        # print(flag)
        return flag



    def choose_action(self, state):
        state_flag=self.check_if_state_exist(state)
        random_number=np.random.uniform()
        # print(self.Q_table[state_flag-1][0])
        if  random_number> self.Epsilon_greddy or self.Q_table[state_flag][1].all() == 0:
            action = np.random.choice(4, 1)[0]
            # print("action__________",action)

        else:
            action = np.argmax(self.Q_table[state_flag][1])
        # print(self.Q_table[state_flag])
        # print(self.Q_table[state_flag][1])
        # print("action__________", action)
        # print(state ,"is action:",action)
        return action

    def learn(self, state, action, reward, state_next,done):
        # print("learnig state:",state_next)
        state_flag=self.check_if_state_exist(state)
        state_next_flag = self.check_if_state_exist(state_next)
        # print("state_next_table:",self.Q_table[state_next_flag])
        # action_flag=self.choose_action(state)
        # print("state_falg",state_flag,"state_next_falg",state_next_flag,len(self.Q_table))
        q_value=self.Q_table[state_flag][1][action]
        # print("q_value",q_value)
        # print("s:",self.Q_table[state_flag][1],"s_:",self.Q_table[state_next_flag],"max()",self.Q_table[state_next_flag][1].max())
        # print("q_value:",q_value)

        # print(self.Q_table[state_flag][1].max())
        # update parameters

        if done==True:
            q_pre=reward
        else:
            q_pre=reward+self.Gmma*self.Q_table[state_next_flag][1].max()


        self.Q_table[state_flag][1][action]+=self.lr*(q_pre-q_value)



    #
    # def see_data(self):
    #     print(self.Q_table)
    #
    #     self.Q_table = np.insert(self.Q_table, 0, values=np.zeros((1, self.actions), np.float), axis=0)
    #     print(self.Q_table)
    #     print(self.Q_table.shape)
    # # def choose_aciton(self,observation):


if __name__ == '__main__':

    env = Two_Dimension_Maze()
    ql = Q_Learning(4)
    def train():
        for j in range(10000):
            s=env.reset()
            while(1):
                env.render()
                action=ql.choose_action(s)
                # print("action_content;",action)
                done,s_,reward=env.step(env.actions[action])
                state_flag = ql.check_if_state_exist(s_)
                # print("s____",ql.Q_table[state_flag])
                ql.learn(s,action,reward,s_,done)
                s=s_
                # print("q_____",ql.Q_table)
                if done:
                    break

            if j%100==0:
                print("q_____",ql.Q_table,len(ql.Q_table))

    # # def train():
    #
    # env.after(0, env.test)
    env.after(0,train())
    env.mainloop()

