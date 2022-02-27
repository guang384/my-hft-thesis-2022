"""
统计多线程情况下训练过程每个阶段执行时间
"""

import multiprocessing as mp
import os
import sys
import time
import timeit

import torch
from elegantrl.config import build_env
from elegantrl.run import PipeEvaluator, PipeLearner, process_safely_terminate, init_agent, PipeWorker, init_buffer, \
    init_evaluator
from step08_train_model import try_train


def train_and_evaluate_mp_async_timed(args):
    args.init_before_training()

    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    evaluator_pipe = MultiRemindersPipeEvaluator()
    process.append(mp.Process(target=evaluator_pipe.run, args=(args,)))

    worker_pipe = PipeWorkerAsync(args.worker_num)
    process.extend([mp.Process(target=worker_pipe.run, args=(args, worker_id))
                    for worker_id in range(args.worker_num)])

    learner_pipe = TimeitPipeLearner()
    process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe)))

    [p.start() for p in process]
    process[-1].join()  # waiting for learner
    process_safely_terminate(process)


""" 针对训练完成进行多次提醒 """
""" Multiple reminders for the completion of the training """


class MultiRemindersPipeEvaluator(PipeEvaluator):

    def run(self, args):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        agent = init_agent(args, gpu_id)
        evaluator = init_evaluator(args, gpu_id)

        '''loop'''
        cwd = args.cwd
        act = agent.act
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        del args

        if_save = False
        if_train = True
        if_reach_goal = False
        temp = 0  # todo
        while if_train:
            act_dict, steps, r_exp, logging_tuple = self.pipe0.recv()
            if act_dict:
                act.load_state_dict(act_dict)
                if_reach_goal, if_save = evaluator.evaluate_save_and_plot(act, steps, r_exp, logging_tuple)

                temp += 1
                if temp == 4:  # todo
                    temp = 0
                    torch.save(act.state_dict(), f"{cwd}/actor_{evaluator.total_step:09}.pth")  # todo
            else:
                evaluator.total_step += steps

            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
            self.pipe0.send((if_train, if_save))
        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

        while True:  # wait for the forced stop from main process
            self.pipe0.recv()  # 防止send阻塞后卡死
            self.pipe0.send((False, False))


class TimeitPipeLearner(PipeLearner):

    @staticmethod
    def run(args, comm_eva, comm_exp):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        agent = init_agent(args, gpu_id)
        buffer = init_buffer(args, gpu_id)

        '''loop'''
        loop_counter = 0
        if_train = True
        while if_train:
            start = timeit.default_timer()
            traj_list = comm_exp.explore(agent)
            explored = timeit.default_timer()
            steps, r_exp = buffer.update_buffer(traj_list)
            buffer_updated = timeit.default_timer()

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
            net_updated = timeit.default_timer()

            if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)
            evaluated = timeit.default_timer()

            explore_time = explored - start
            update_buffer_time = buffer_updated - explored
            update_net_time = net_updated - buffer_updated
            evaluate_time = evaluated - net_updated
            total_time = evaluated - start
            loop_counter += 1
            if loop_counter % 10 == 0:
                try:
                    sample_rounds = int(1 + buffer.now_len * args.repeat_times / args.batch_size)
                    print("Train loop %d, New steps %d (%.2f%%), "
                          "Buffed %d (%.2f%%), UpdNet sampled %d rounds. "
                          "Time %.4fs ( Exp %.4fs, UpdBuf %.4fs, UpdNet %.4fs (%.6fs/r), Eva %.4fs ) if_train=%s"
                          % (loop_counter,
                             steps,
                             (steps / buffer.now_len) * 100,
                             buffer.now_len,
                             (buffer.now_len / args.max_memo) * 100,
                             sample_rounds,
                             total_time,
                             explore_time,
                             update_buffer_time,
                             update_net_time,
                             update_net_time / sample_rounds,
                             evaluate_time, str(if_train)))
                except AttributeError:
                    print("Train loop %d, New steps %d , "
                          "Time %.4fs ( Exp %.4fs, UpdBuf %.4fs, UpdNet %.4fs, Eva %.4fs ) if_train=%s"
                          % (loop_counter,
                             steps,
                             total_time,
                             explore_time,
                             update_buffer_time,
                             update_net_time,
                             evaluate_time, str(if_train)))
        agent.save_or_load_agent(args.cwd, if_save=True)
        print(f'| Learner: Save in {args.cwd}')

        if hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {args.cwd}")
            buffer.save_or_load_history(args.cwd, if_save=True)


class PipeWorkerAsync(PipeWorker):
    def __init__(self, worker_num):
        super().__init__(worker_num)
        self.exploring_act_dict = None
        self._closed = False

    def explore(self, agent):
        act_dict = agent.act.state_dict()

        if self.exploring_act_dict is None:  # 第一次进来的时候没有模型不能发起探索（用异步探索模型更新会延后一个节拍
            for worker_id in range(self.worker_num):
                self.pipe1s[worker_id].send(act_dict)
            self.exploring_act_dict = act_dict
        # 异步接受探索
        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        # 提前发起一次探索
        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)
        self.exploring_act_dict = act_dict

        return traj_lists

    def run(self, args, worker_id):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        env = build_env(args.env, args.env_func, args.env_args)
        agent = init_agent(args, gpu_id, env)

        '''loop'''
        target_step = args.target_step
        if args.if_off_policy:
            trajectory = agent.explore_env(env, args.target_step)
            self.pipes[worker_id][0].send(trajectory)
        del args

        while True:
            act_dict = self.pipes[worker_id][0].recv()
            agent.act.load_state_dict(act_dict)
            trajectory = agent.explore_env(env, target_step)
            self.pipes[worker_id][0].send(trajectory)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        argv_file_path = sys.argv[1]
        try_train(file_path=argv_file_path,
                  train_and_evaluate_func=train_and_evaluate_mp_async_timed)
    elif len(sys.argv) == 3:
        argv_file_path = sys.argv[1]
        argv_cwd_suffix = sys.argv[2]
        try_train(file_path=argv_file_path, cwd_suffix=argv_cwd_suffix,
                  train_and_evaluate_func=train_and_evaluate_mp_async_timed)
    elif len(sys.argv) == 4:
        argv_agent_name = sys.argv[1]
        argv_file_path = sys.argv[2]
        argv_cwd_suffix = sys.argv[3]

        try_train(agent_name=argv_agent_name, file_path=argv_file_path, cwd_suffix=argv_cwd_suffix,
                  train_and_evaluate_func=train_and_evaluate_mp_async_timed)
    else:
        try_train(train_and_evaluate_func=train_and_evaluate_mp_async_timed)
