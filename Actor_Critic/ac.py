import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

LR = 0.01
gamma = 0.9
total_episodes = 50000
render = False

env = gym.make('CartPole-v0')
env = env.unwrapped

# env.seed(1)
# torch.manual_seed(1)
eps = np.finfo(np.float32).eps.item()
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
# print(num_state)





class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.output = nn.Linear(128, num_action)
        self.gamma = gamma

        self.memory = []
        self.reward=[]
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.output(x))

        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(num_state,128)
        self.output=nn.Linear(128,1)
        self.gamma=gamma

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x = F.relu(self.output(x))
        return x


actor=Actor()
critic=Critic()

a_optimezer=torch.optim.Adam(actor.parameters(),lr=LR)
c_optimezer=torch.optim.Adam(critic.parameters(),lr=LR)


def choose_action(state):
    state=torch.from_numpy(state).float()
    value=critic(state)
    prob=actor(state)
    # print(prob)
    c = torch.distributions.Categorical(prob)

    action = c.sample()

    log_prob = c.log_prob(action)
    actor.memory.append([log_prob,value])
    # print("c:", c)
    # print("action:", action)
    # print("c:", log_prob)



    return action.item()

def learn():
    rewards=[]
    Mem=actor.memory
    critic_loss=[]
    actor_loss=[]
    R=0

    for r in actor.reward[::-1]:
        # print('reward_total:',actor.reward)
        # print('reward',actor.reward[::-1])
        # print('r',r)
        # print(r)
        R=r+actor.gamma*R
        rewards.insert(0,R)
    rewards=torch.tensor(rewards)
    rewards=(rewards-rewards.mean())/(rewards.std()+eps)
    # print('1',rewards)
    for (log_prob,value),r in zip(Mem,rewards):

        reward=r-value.item()
        actor_loss.append(-log_prob*reward)
        critic_loss.append(F.smooth_l1_loss(value,torch.tensor([r])))
        print('a',actor_loss)
        print('c',critic_loss)
    a_optimezer.zero_grad()
    c_optimezer.zero_grad()
    a_l=torch.stack(actor_loss).sum()
    c_l=torch.stack(critic_loss).sum()

    a_l.backward()
    c_l.backward()
    c_optimezer.step()
    a_optimezer.step()

    loss_total=c_l+a_l
    return loss_total

if __name__=='__main__':
    actor = Actor()
    critic = Critic()

    a_optimezer = torch.optim.Adam(actor.parameters(), lr=LR)
    c_optimezer = torch.optim.Adam(critic.parameters(), lr=LR)

    # choose_action(np.ones((1,2)))
    run_steps=[]

    for i in range(total_episodes):
        state=env.reset()
        if render :env.render()
        steps=0
        while 1:

            # print(state)
            action=choose_action(state)
            state,re,done,info=env.step(action)
            # print(state)
            re=state[0]+re
            if render: env.render()
            actor.reward.append(re)
            steps+=1
            if done:
                run_steps.append(steps)
                # print("Epiosde {} , run step is {} ".format(i + 1, steps + 1))
                # print(reward)
                break

        l=learn()

        print('i:',i,'loss:',steps)
        actor.memory = []
        actor.reward = []

