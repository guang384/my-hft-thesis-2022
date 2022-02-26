import os
import random
import timeit

import gym
import numpy as np
import torch
from gym import envs, register
from matplotlib import pyplot as plt, gridspec

from tiny_market import GymEnvLongAndShortImbalanceWillResultInFines, \
    GymEnvUseSharpRatioAsReward, GymEnvRandom, GymEnvWaitAndSeeWillResultInFines, \
    GymEnvDoneIfUndermargined, GymEnvDoneIfOrderTimeout, GymEnvFeatureScaling

from step17_try_lstm_train import REINFORCE

''' 参数 '''
if_plot_ion = True  # 交互模式
if if_plot_ion:
    plt.ion()  # 交互模式

if_gpu = torch.cuda.is_available()

params_hidden_size = 256
params_num_episodes = 10000
params_max_steps = 1000000
params_checkpoint_frequency = 500
params_gamma = 0.99
random.seed(timeit.default_timer())
params_seed = random.randint(0, 2 ** 32 - 1)
params_fine_pre_seconds = 0.5

''' 定义环境类 '''


# 依赖关系是从左往右（先调用自己内部的然后遇到super从左往右，直到没有super为止
class GymEnvRandomWithSharpeReward(GymEnvDoneIfUndermargined,
                                   GymEnvDoneIfOrderTimeout,
                                   GymEnvLongAndShortImbalanceWillResultInFines,
                                   GymEnvWaitAndSeeWillResultInFines,
                                   # GymEnvUseSharpRatioAsReward,
                                   GymEnvFeatureScaling,
                                   GymEnvRandom):
    pass


''' 初始化环境 '''
ENV_NAME = 'TinyMarketGymEnvRandomWithSharpeReward-v0'

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

if ENV_NAME not in env_ids:
    register(
        id=ENV_NAME,
        entry_point='step19_tran_with_sharpe_ratio:GymEnvRandomWithSharpeReward',
        max_episode_steps=1800,  # 一个episode最大步数
        reward_threshold=10000.0,
    )


def train(env_name=ENV_NAME, seed=params_seed, dir_suffix=None):
    # 创建工作目录
    if dir_suffix is None:
        working_dir = 'checkpoint_' + ENV_NAME
    else:
        working_dir = 'checkpoint_' + ENV_NAME + '_' + dir_suffix

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    # 创建环境
    env = gym.make(env_name,
                   capital=20000,
                   fine_pre_seconds=params_fine_pre_seconds,
                   file_path="data/dominant_reprocessed_data_202111_ind.h5",
                   date_start="20211101", date_end="20211121",
                   )
    # 为随机数设置随机种子 (Gym、numpy、Pytorch都要设置随机数种子)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 创建Agent
    agent = REINFORCE(params_hidden_size, env.observation_space.shape[0], env.action_space)

    # 为训练做准备
    log_reward = []
    log_smooth = []
    log_step_count = []
    log_loss = []

    # 双Y轴展示
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    plt.subplots_adjust(left=0.13, right=0.87, top=0.9, bottom=0.1)
    ax1 = plt.subplot(gs[0])
    ax2 = ax1.twinx()
    ax3 = plt.subplot(gs[1])
    ax1.set_ylabel(r"Ticks")
    ax2.set_ylabel(r"The profit and loss")
    ax3.set_ylabel(r"Model loss")
    plt.grid(ls='--')

    best_loss = -100000

    # 开始训练
    for i_episode in range(1, params_num_episodes):
        plt.suptitle(" Progress : %2.f%% " % ((i_episode / params_num_episodes) * 100))
        # 重置环境
        state = torch.Tensor(np.array([env.reset()]))
        # 重置参数
        entropies = []
        log_probs = []
        rewards = []
        actions = []
        positions = []
        hx = torch.zeros(params_hidden_size).unsqueeze(0).unsqueeze(0)  # 初始化隐状态
        cx = torch.zeros(params_hidden_size).unsqueeze(0).unsqueeze(0)
        if if_gpu:
            hx = hx.cuda()
            cx = cx.cuda()
        action_count = [0, 0, 0]
        # 进入训练循环
        for t in range(params_max_steps):  # 1个episode最长num_steps
            # 获取Action
            action, log_prob, entropy, hx, cx = agent.select_action(state.unsqueeze(0), hx, cx)
            action = action.cpu()
            # 离散化
            action = action.squeeze().softmax(dim=0).argmax()
            # 行动
            next_state, reward, done, info = env.step(action)
            # 记录
            action_count[action] += 1
            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)
            actions.append(action)
            positions.append(info['position'])
            state = torch.Tensor(np.array([next_state]))
            # 是否完成
            if done:
                break

        # episode结束，开始训练
        new_loss = agent.update_parameters(rewards, log_probs, entropies, params_gamma)

        if new_loss > best_loss:
            best_loss = new_loss
            torch.save(agent.model.state_dict(),
                       os.path.join(working_dir,
                                    'finePS_' + str(params_fine_pre_seconds)
                                    + '_episode_' + str(i_episode)
                                    + '_best_loss_' + str(best_loss) + '.pkl'))

        # 到达检测点 记录模型
        if i_episode % params_checkpoint_frequency == 0:
            # 保存模型
            torch.save(agent.model.state_dict(),
                       os.path.join(working_dir,
                                    'ckp-' + str(i_episode) + 'finePS_' + str(params_fine_pre_seconds)+'.pkl'))
            # 保存图片
            fig.savefig(os.path.join(working_dir,
                                     'ckp-' + str(i_episode) + 'finePS_' + str(params_fine_pre_seconds)+'.jpg'))
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax1.set_ylabel(r"Ticks")
            ax2.set_ylabel(r"The profit and loss")
            ax3.set_ylabel(r"Model loss")
            # 重置参数
            log_reward = []
            log_smooth = []
            log_step_count = []
            log_loss = []

        print("Episode: {:<5.0f}, reward: {:<10.1f}, use {:5.0f} ticks, amount {:<10.1f} , actions {}".format(
            i_episode, np.sum(rewards), t + 1, info['amount'] - 20000, str(action_count)))

        # log_reward.append(np.sum(rewards))
        log_reward.append(info['amount'] - 20000)

        log_step_count.append(t)

        log_loss.append(new_loss)

        if len(log_smooth) == 0:
            log_smooth.append(log_reward[-1])
        else:
            # log_smooth.append(log_smooth[-1] * 0.99 + 0.01 * np.sum(rewards))
            log_smooth.append(log_smooth[-1] * 0.99 + 0.01 * (info['amount'] - 20000))

        if if_plot_ion:
            # 交互绘图
            ax1.plot(log_step_count, '--', color='aquamarine', label='steps')
            ax2.plot(log_reward, ':b', label='Reward')
            ax2.plot(log_smooth, '-y')
            ax3.plot(log_loss, '-r')
            plt.pause(1e-5)

    env.close()


if __name__ == '__main__':
    ''' 执行训练 '''
    train()
    print('Done.')
