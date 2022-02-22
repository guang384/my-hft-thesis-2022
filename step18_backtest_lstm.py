import os
import re
import sys
import time
import gym
import numpy as np
import torch

import step15_backtest
import matplotlib.pyplot as plt
from step17_try_lstm_train import REINFORCE, params_hidden_size, if_gpu, ENV_NAME

plt.ion()  # 交互模式


def load_last_backtest(agent, dir):
    # 加载尚未测试
    file_names = os.listdir(dir)
    for name in file_names:
        match = re.search(r'reinforce-(\d+)\.pkl', name)
        if match is None:
            continue
        index = int(match.group(1))
        if not os.path.exists(os.path.join(dir, 'backtest-' + str(index) + '.jpg')):
            # 不存在 需要测试
            model_path = os.path.join(dir, name)
            if not os.path.exists(model_path):
                return None
            if if_gpu:
                model_static_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
            else:
                model_static_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            agent.model.load_state_dict(model_static_dict)
            print('Model loaded : ', model_path)
            return index
    return None


def backtest(dir_suffix=None):

    if dir_suffix is None:
        working_dir = 'checkpoint_' + ENV_NAME
    else:
        working_dir = 'checkpoint_' + ENV_NAME + '_' + dir_suffix

    env = gym.make(step15_backtest.ENV_NAME,
                   capital=20000,
                   file_path="data/dominant_reprocessed_data_202111_ind.h5",
                   date_start="20211122", date_end="20211130")
    """ 加载模型 """
    agent = REINFORCE(params_hidden_size, env.observation_space.shape[0], env.action_space)

    testing_index = load_last_backtest(agent, working_dir)
    if testing_index is None:
        return

    """ 准备画板 """
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    """ 开始测试 """
    counter = 0
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

            # 交互展示
            counter += 1
            if counter % 300 == 0:
                ax1.plot(log_price, '-b', label='Price')
                ax2.plot(log_position, '-y', label='Position')
                ax3.plot(log_amount, '-r', label='steps')
                plt.pause(1e-6)

            # 下一个
            state = torch.Tensor(np.array([next_state]))
        print("The final gain is %.2f. The final total account balance is %.2f RMB, Date: %s\n %s"
              % (episode_return, info['amount'], env.current_day(), str(info)))
        # 下一天
        env.set_capital(info['amount'])
        env.reset()
    plt.suptitle("The final balance is " + str(info['amount']))
    plt.savefig(os.path.join(working_dir, 'backtest-' + testing_index + '.jpg'))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        argv_dir_suffix = sys.argv[1]
        while True:
            print('try testing ...')
            backtest(argv_dir_suffix)
            print('sleeping ...')
            time.sleep(60)
    else:
        while True:
            print('try testing ...')
            backtest()
            print('sleeping ...')
            time.sleep(60)
    print('Done.')
