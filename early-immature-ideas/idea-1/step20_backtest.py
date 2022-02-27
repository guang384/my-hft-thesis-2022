import os
import re
import sys
import time
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from gym import envs, register

from step19_tran import REINFORCE, params_hidden_size, if_gpu, params_fine_pre_seconds
from tiny_market import GymEnvLongAndShortImbalanceWillResultInFines, \
    GymEnvUseSharpRatioAsReward, GymEnvDaily, GymEnvWaitAndSeeWillResultInFines, \
    GymEnvDoneIfUndermargined, GymEnvDoneIfOrderTimeout, GymEnvFeatureScaling


''' 定义环境类 '''


# 依赖关系是从左往右（先调用自己内部的然后遇到super从左往右，直到没有super为止
class GymEnvDailyBacktestForStep19(GymEnvDoneIfUndermargined,
                                   GymEnvDoneIfOrderTimeout,
                                   GymEnvLongAndShortImbalanceWillResultInFines,
                                   GymEnvWaitAndSeeWillResultInFines,
                                   # GymEnvUseSharpRatioAsReward,
                                   GymEnvFeatureScaling,
                                   GymEnvDaily):
    pass


''' 初始化环境 '''
ENV_NAME = 'TinyMarketGymEnvDailyBacktestForStep19-v0'

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

if ENV_NAME not in env_ids:
    register(
        id=ENV_NAME,
        entry_point='step20_backtest:GymEnvDailyBacktestForStep19',
        max_episode_steps=50000,  # 一个episode最大步数
        reward_threshold=10000.0,
    )


def backtest(model_file):
    env = gym.make(ENV_NAME,
                   capital=20000,
                   fine_pre_seconds=params_fine_pre_seconds,
                   file_path="data/dominant_reprocessed_data_202111_ind.h5",
                   date_start="20211122", date_end="20211125")
    """ 加载模型 """
    agent = REINFORCE(params_hidden_size, env.observation_space.shape[0], env.action_space)

    if os.path.exists(model_file):
        if if_gpu:
            model_static_dict = torch.load(model_file, map_location=lambda storage, loc: storage.cuda())
        else:
            model_static_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        agent.model.load_state_dict(model_static_dict)
        agent.model.eval()  # Sets the module in evaluation mode.

        print('Model loaded : ', model_file)
    else:
        raise RuntimeError(model_file, ' not found!')

    """ 开始测试 """
    log_price = []
    log_position = []
    log_amount = []

    while not env.if_all_days_done():

        # 获取第一个状态
        state = torch.Tensor(np.array([env.reset()]))
        # 初始化隐状态
        hx = torch.zeros(params_hidden_size).unsqueeze(0).unsqueeze(0)
        cx = torch.zeros(params_hidden_size).unsqueeze(0).unsqueeze(0)
        if if_gpu:
            hx = hx.cuda()
            cx = cx.cuda()
        # 开始回测
        done = False
        episode_return = 0

        while not done:
            action, log_prob, entropy, hx, cx = agent.select_action(state.unsqueeze(0), hx, cx)
            action = action.cpu()
            # 离散化
            action = action.squeeze().softmax(dim=0).argmax()
            # 行动
            next_state, reward, done, info = env.step(action)

            # 记录
            episode_return += reward
            log_price.append(int(state[0][0].item() * 1000))  # 注意要缩放回去
            log_position.append(info['position'])
            log_amount.append(info['amount'])

            # 下一个
            state = torch.Tensor(np.array([next_state]))
        print("The final total account balance is %.2f RMB, Date: %s\n %s"
              % (info['amount'], env.current_day(), str(info)))
        # 设置金额
        env.set_capital(info['amount'])

    """ 准备画板 """
    plt.figure()
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    plt.suptitle("The final balance is " + str(info['amount']))
    ax1.plot(log_price, '-b', label='Price')
    ax2.plot(log_position, '-y', label='Position')
    ax3.plot(log_amount, '-r', label='steps')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        argv_model_file = sys.argv[1]
        backtest(argv_model_file)
    else:
        backtest('checkpoint_TinyMarketGymEnvRandomWithSharpeReward-v0/'
                 'finePS_0.3_episode_322_best_loss_31.630949020385742.pkl')
    print('Done.')
