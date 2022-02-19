"""
统计多线程情况下训练过程每个阶段执行时间
"""

import multiprocessing as mp
import sys
import timeit

import torch
from elegantrl.run import PipeEvaluator, PipeLearner, process_safely_terminate, init_agent, PipeWorker, init_buffer
from step08_train_model import try_train


def train_and_evaluate_mp_timed(args):
    args.init_before_training()

    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    evaluator_pipe = PipeEvaluator()
    process.append(mp.Process(target=evaluator_pipe.run, args=(args,)))

    worker_pipe = PipeWorker(args.worker_num)
    process.extend([mp.Process(target=worker_pipe.run, args=(args, worker_id))
                    for worker_id in range(args.worker_num)])

    learner_pipe = TimeitPipeLearner()
    process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe)))

    [p.start() for p in process]
    process[-1].join()  # waiting for learner
    process_safely_terminate(process)


class TimeitPipeLearner:
    def __init__(self):
        pass

    @staticmethod
    def run(args, comm_eva, comm_exp):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        agent = init_agent(args, gpu_id)
        buffer = init_buffer(args, gpu_id)

        '''loop'''
        loop_counter = 0
        total_explore = 0
        total_update_buffer = 0
        total_update_net = 0
        total_evaluate = 0
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
            total_explore += explored - start
            total_update_buffer += buffer_updated - explored
            total_update_net += net_updated - buffer_updated
            total_evaluate += evaluated - net_updated
            loop_counter += 1
            if loop_counter % 1 == 0:
                print("Train loop %d, Increased steps %d, "
                      "Buffer size %d, Update net sample rounds %d, "
                      "avgExp %.4f, avgUpdBuf %.4f, avgUpdNet %.4f, avgEva %.4f ."
                      % (loop_counter,
                         steps,
                         buffer.now_len,
                         int(1 + buffer.now_len * args.repeat_times / args.batch_size),
                         total_evaluate / loop_counter,
                         total_update_buffer / loop_counter,
                         total_update_net / loop_counter,
                         total_evaluate / loop_counter))
        agent.save_or_load_agent(args.cwd, if_save=True)
        print(f'| Learner: Save in {args.cwd}')

        if hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {args.cwd}")
            buffer.save_or_load_history(args.cwd, if_save=True)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        argv_file_path = sys.argv[1]
        try_train(file_path=argv_file_path,
                  train_and_evaluate_func=train_and_evaluate_mp_timed)
    elif len(sys.argv) == 3:
        argv_file_path = sys.argv[1]
        argv_cwd_suffix = sys.argv[2]
        try_train(file_path=argv_file_path, cwd_suffix=argv_cwd_suffix,
                  train_and_evaluate_func=train_and_evaluate_mp_timed)
    elif len(sys.argv) == 4:
        argv_agent_name = sys.argv[1]
        argv_file_path = sys.argv[2]
        argv_cwd_suffix = sys.argv[3]

        try_train(agent_name=argv_agent_name, file_path=argv_file_path, cwd_suffix=argv_cwd_suffix,
                  train_and_evaluate_func=train_and_evaluate_mp_timed)
    else:
        try_train(train_and_evaluate_func=train_and_evaluate_mp_timed)
