import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# np.random.seed(1)
# torch.manual_seed(1)


class Net(nn.Module):
    def __init__(self, number_states, number_actions):
        super(Net, self).__init__()
        self.number_states = number_states
        self.number_actions = number_actions
        print(number_actions)



        self.layer_dense = nn.Linear(self.number_states, 10, bias=True)

        self.output = nn.Linear(10, self.number_actions, bias=True)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.xavier_normal_(self.layer_dense.weight)
    def forward(self, x):
        x = F.leaky_relu(self.layer_dense(x))
        x = F.leaky_relu(self.output(x))
        # print("b", x)
        x = F.softmax(x)
        return x


class PolicyGradient(Net):
    def __init__(self, number_states, number_actions, lr=0.01, batch_size=512,
                 gamma=0.95):
        super(Net, self).__init__()

        self.pl = Net(number_states, number_actions)
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = torch.optim.RMSprop(self.pl.parameters(), lr=self.lr)

        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []


    def choose_action(self, state):
        state = torch.FloatTensor(state)
        probability_actions = F.softmax(self.pl(state))
        # print('pppppp',probability_actions)
        action = torch.multinomial(probability_actions, 1)
        # print("p",probability_actions)
        # print("action",action)
        action = action.numpy()[0]
        # print("action_after",action)
        return action

    def store_data(self,state,action,reward):
        self.ep_states.append(state)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)


    def Normlize_reward(self,rewards):

        discounted_ep_rewards=np.zeros_like(rewards)
        running_add=0
        for i in range(len(self.ep_rewards)):
            running_add=running_add * self.gamma+self.ep_rewards[i]
            discounted_ep_rewards[i]=running_add

        discounted_ep_rewards-=np.mean(discounted_ep_rewards)
        discounted_ep_rewards/=np.std(discounted_ep_rewards)
        return discounted_ep_rewards

    def learn(self):
        discounted_ep_rewards_normal=self.Normlize_reward(self.ep_rewards)
        # print(discounted_ep_rewards_normal)
        states=torch.FloatTensor(self.ep_states)
        actions = torch.LongTensor(self.ep_actions)
        # print("actions",actions)
        rewards = torch.FloatTensor(self.ep_rewards)
        pro_actions=self.pl(states)
        # print(pro_actions)
        # print(actions)
        neg_log_prob=F.cross_entropy(pro_actions,actions)

        # print("neg_log_prob",neg_log_prob)
        # print(rewards)
        loss=-torch.mean(neg_log_prob*rewards)
        # print(loss)

        # m = torch.bernoulli(pro_actions)
        # print(m)
        # loss = -m.log_prob(actions) * rewards


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_states,self.ep_actions,self.ep_rewards=[],[],[]
        return sum(discounted_ep_rewards_normal),loss.detach().numpy()






if __name__ == '__main__':
    pl = PolicyGradient(4, 2)
    a = np.ones((1, 4))
    # a=torch.FloatTensor(a)
    # print(pl.forward(a))
    # print(pl.choose_action(a))
