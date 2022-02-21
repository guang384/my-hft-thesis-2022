import argparse, math, os, sys
import timeit

import numpy as np
import gym
from gym import wrappers, register, envs
import matplotlib.pyplot as plt
import random

import torch
from gym.spaces import Discrete
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.utils as utils

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tiny_market import profits_or_loss_reward, linear_fine

plt.ion()  # 交互模式

""" 参数"""
if_display = False
if_plot = True
if_gpu = torch.cuda.is_available()

params_hidden_size = 128
params_num_episodes = 2000
params_max_steps = 10000
params_checkpoint_frequency = 100
params_gamma = 0.99
params_seed = random.randint(0, 2 ** 32 - 1)

""" 初始化环境 """

ENV_NAME = 'TinyMarketGymEnvRandomInd-v0'

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

if ENV_NAME not in env_ids:
    register(
        id=ENV_NAME,
        entry_point='step14_train:GymEnvRandomIndData',
        max_episode_steps=3600,  # 一个episode最大步数
        reward_threshold=10000.0,
    )


# gym.logger.set_level(40)  # Block warning


def make_env_func(**kwargs):
    env_inner = gym.make(ENV_NAME,
                         capital=20000,
                         file_path="data/dominant_reprocessed_data_202111_ind.h5",
                         date_start="20211115", date_end="20211121",
                         reward_func=profits_or_loss_reward,
                         fine_func=linear_fine(1)
                         )
    return env_inner


# 创建环境
env = make_env_func()  # 创建环境

if if_display:
    env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(ENV_NAME), force=True)

# 为随机数设置随机种子
# Gym、numpy、Pytorch都要设置随机数种子
env.seed(params_seed)
torch.manual_seed(params_seed)
np.random.seed(params_seed)

""" 神经网络定义的策略 """


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space  # 动作空间
        if isinstance(action_space, Discrete):
            num_outputs = action_space.n  # 离散动作空间 （最后用softmax）
        else:
            num_outputs = action_space.shape[0]  # 动作空间的维度

        self.lstm = nn.LSTM(num_inputs, hidden_size, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, hidden_size)  # 隐层神经元数量
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.linear2_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)  # 为了输出连续域动作，实际上policy net定义了
        sigma_sq = self.linear2_(x)  # 一个多维高斯分布，维度=动作空间的维度

        return mu, sigma_sq, hidden


class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)  # 创建策略网络
        if if_gpu:
            self.model = self.model.cuda()  # GPU版本
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)  # 优化器
        self.model.train()  # Sets the module in training mode.
        self.pi = Variable(torch.FloatTensor([math.pi]))  # .cuda()    # 圆周率
        if if_gpu:
            self.pi = self.pi.cuda()

    def normal(self, x, mu, sigma_sq):  # 计算动作x在policy net定义的高斯分布中的概率值
        a = (-1 * (Variable(x) - mu).pow(2) / (2 * sigma_sq)).exp()
        b = 1 / (2 * sigma_sq * self.pi.expand_as(sigma_sq)).sqrt()
        return a * b  # pi.expand_as(sigma_sq)的意义是将标量π扩展为与sigma_sq同样的维度

    def select_action(self, state, hx, cx):
        if if_gpu:
            mu, sigma_sq, (hx, cx) = self.model(Variable(state).cuda(), (hx, cx))
            sigma_sq = F.softplus(sigma_sq).cuda()
        else:
            mu, sigma_sq, (hx, cx) = self.model(Variable(state), (hx, cx))
            sigma_sq = F.softplus(sigma_sq)

        eps = torch.randn(mu.size())  # 产生一个与动作向量维度相同的标准正态分布随机向量
        if if_gpu:
            action = (mu + sigma_sq.sqrt() * Variable(eps).cuda()).data
        else:
            action = (mu + sigma_sq.sqrt() * Variable(eps)).data  # 相当于从N(μ,σ²)中采样一个动作
        prob = self.normal(action, mu, sigma_sq)  # 计算动作概率
        # 高斯分布的信息熵，参考https://blog.csdn.net/raby_gyl/article/details/73477043
        entropy = -0.5 * ((sigma_sq + 2 * self.pi.expand_as(sigma_sq)).log() + 1)

        log_prob = prob.log()  # 对数概率
        return action, log_prob, entropy, hx, cx

    def update_parameters(self, rewards, log_probs, entropies, gamma):  # 更新参数
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]  # 倒序计算累计期望
            if if_gpu:
                loss = loss - (log_probs[i] * (Variable(R).expand_as(log_probs[i])).cuda()).sum() \
                       - (0.0001 * entropies[i].cuda()).sum()
            else:
                loss = loss - (log_probs[i] * (Variable(R).expand_as(log_probs[i]))).sum() \
                       - (0.0001 * entropies[i]).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)  # 梯度裁剪，梯度的最大L2范数=40
        self.optimizer.step()


def train():
    agent = REINFORCE(params_hidden_size, env.observation_space.shape[0], env.action_space)

    if_discrete = isinstance(env.action_space, Discrete)

    working_dir = 'checkpoint_' + ENV_NAME
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    log_reward = []
    log_smooth = []
    log_step_count = []

    if if_plot:
        # 双Y轴展示
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel(r"Reward")
        ax.set_xlabel(r"Episode")
        ax2 = ax.twinx()
        ax2.set_ylabel(r"Use Steps")

    for i_episode in range(params_num_episodes):
        state = torch.Tensor(np.array([env.reset()]))
        entropies = []
        log_probs = []
        rewards = []
        hx = torch.zeros(params_hidden_size).unsqueeze(0).unsqueeze(0)  # 初始化隐状态
        cx = torch.zeros(params_hidden_size).unsqueeze(0).unsqueeze(0)
        if if_gpu:
            hx = hx.cuda()
            cx = cx.cuda()
        # print(hx.shape)
        for t in range(params_max_steps):  # 1个episode最长num_steps
            # print(state.shape)
            action, log_prob, entropy, hx, cx = agent.select_action(state.unsqueeze(0), hx, cx)
            action = action.cpu()
            if if_discrete:  # 离散化
                action = action.squeeze().softmax(dim=0).argmax()
                next_state, reward, done, info = env.step(action)
            else:
                next_state, reward, done, info = env.step(action.numpy()[0, 0])

            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = torch.Tensor(np.array([next_state]))

            if done:
                break
        # episode结束，开始训练
        agent.update_parameters(rewards, log_probs, entropies, params_gamma)

        if i_episode % params_checkpoint_frequency == 0:
            torch.save(agent.model.state_dict(), os.path.join(working_dir, 'reinforce-' + str(i_episode) + '.pkl'))

        print("Episode: {}, reward: {}, use {} ticks, amount {}".format(
            i_episode, np.sum(rewards), t, info['amount'] - 20000))

        log_reward.append(np.sum(rewards))
        log_step_count.append(t)
        if i_episode == 0:
            log_smooth.append(log_reward[-1])
        else:
            log_smooth.append(log_smooth[-1] * 0.99 + 0.01 * np.sum(rewards))

        if if_plot:
            # 交互绘图
            ax.plot(log_reward, '-b', label='Reward')
            ax.plot(log_smooth, '-y')
            ax2.plot(log_step_count, '-r', label='steps')
            plt.pause(1e-5)

    env.close()


if __name__ == '__main__':
    train()
