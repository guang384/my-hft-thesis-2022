"""
尝试把训练好的模型用作预测查看效果
"""

import sys

import numpy as np
import torch
from elegantrl import evaluator, config
from elegantrl.agent import AgentModSAC
from elegantrl.config import get_gym_env_args, Arguments

import gym

gym.logger.set_level(40)  # Block warning


def test_model(actor_path="data_sample/actor_00198252_00219.352.pth"):
    get_gym_env_args(gym.make("LunarLanderContinuous-v2"), if_print=True)

    env_func = gym.make
    env_args = {
        "env_num": 1,
        "env_name": "LunarLanderContinuous-v2",
        "max_step": 1000,
        "state_dim": 8,
        "action_dim": 2,
        "if_discrete": False,
        "target_return": 200,
        "id": "LunarLanderContinuous-v2",
    }

    agent = AgentModSAC

    args = Arguments(agent, env_func=env_func, env_args=env_args)
    gpu_id = args.learner_gpus

    env = config.build_env(env_func=env_func, env_args=env_args)

    act = agent(args.net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id, args=args).act
    act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    torch.no_grad()
    r_s_ary = [evaluator.get_episode_return_and_step(env, act) for _ in range(args.eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    print("run %d rounds. r_avg = %f  s_avg = %f" % (args.eval_times, r_avg, s_avg))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        argv_actor_path = sys.argv[1]
        test_model(argv_actor_path)
    else:
        test_model()
