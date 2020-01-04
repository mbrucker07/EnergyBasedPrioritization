import os
import sys
import pickle
import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork

import os.path as osp
import tempfile
import datetime

def send_key_value_pair(num_cpu, key, value):
    if num_cpu is None:
        raise Exception("Num cpu is none")
    if key is None:
        raise Exception("key is none")
    if value is None:
        value = [0.]
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert rank < num_cpu
    if rank > 0:
        data = [key, value]
        comm.send(data, dest=0, tag=rank)
        return None, None
    elif rank == 0:
        val_list = list()
        val_list.append(value)
        if num_cpu > 1:
            for i in range(1, num_cpu):
                data = comm.recv(source=i, tag=i)
                assert data[0] == key
                val_list.append(data[1])
        x = np.array(val_list)
        mean = np.mean(x)
        return key, mean


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluators, evaluators_names, n_epochs, n_test_rollouts, n_cycles, n_batches,
          policy_save_interval, save_policies, num_cpu, dump_buffer, w_potential, w_linear,
          w_rotational, rank_method, clip_energy, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    t = 1
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for cycle in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode, dump_buffer, w_potential, w_linear, w_rotational, rank_method, clip_energy)
            for batch in range(n_batches):
                t = ((epoch*n_cycles*n_batches)+(cycle*n_batches)+batch)*num_cpu
                policy.train(t, dump_buffer)

            policy.update_target_net()

        """
        # test

        evaluator = evaluators[0] # TODO: new

        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            if 'success_rate' in key:
                print("Test success: {} with history {}".format(val, list(evaluator.success_history)))  # TODO new
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            if 'success_rate' in key:
                print("Rollout success: {} with history {}".format(val, list(rollout_worker.success_history)))  # TODO new
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))


        # TODO NEW SECTION
        if len(evaluators) > 1:
            i = 1
            for eval, name in zip(evaluators[1:], evaluators_names[1:]):
                eval.clear_history()
                for _ in range(n_test_rollouts):
                    eval.generate_rollouts()
                # record logs
                for key, val in eval.logs(name):
                    if 'success_rate' in key:
                        print("{} success: {} with history {}".format(name, val, list(eval.success_history)))  # TODO new
                    logger.record_tabular(key, mpi_average(val))
                i += 1

        # TODO END NEW

        """
        # TODO NEW NEW

        # test

        evaluator = evaluators[0] # TODO: new

        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            if 'success_rate' in key:
                print("Test success: {} with history {}".format(val, list(evaluator.success_history)))  # TODO new
            key, mean = send_key_value_pair(num_cpu, key, val)
            if rank == 0:
                logger.record_tabular(key, mean)
        for key, val in rollout_worker.logs('train'):
            if 'success_rate' in key:
                print("Rollout success: {} with history {}".format(val, list(rollout_worker.success_history)))  # TODO new
            key, mean = send_key_value_pair(num_cpu, key, val)
            if rank == 0:
                logger.record_tabular(key, mean)
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))


        # TODO NEW SECTION
        if len(evaluators) > 1:
            i = 1
            for eval, name in zip(evaluators[1:], evaluators_names[1:]):
                eval.clear_history()
                for _ in range(n_test_rollouts):
                    eval.generate_rollouts()
                # record logs
                for key, val in eval.logs(name):
                    if 'success_rate' in key:
                        print("{} success: {} with history {}".format(name, val, list(eval.success_history)))  # TODO new
                    key, mean = send_key_value_pair(num_cpu, key, val)
                    if rank == 0:
                        logger.record_tabular(key, mean)
                i += 1
        # TODO END NEW NEW



        if rank == 0:
            logger.dump_tabular()
            if dump_buffer:
                policy.dump_buffer(epoch)

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(
    env_name, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return,
    temperature, prioritization, binding, logging, version, dump_buffer, n_cycles, rank_method,
    w_potential, w_linear, w_rotational, clip_energy, save_path, load_path, override_params={}, save_policies=True):

    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        whoami = mpi_fork(num_cpu, binding)
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging

    if logging: 
        logdir = 'logs/'+str(env_name)+'-temperature'+str(temperature)+\
                 '-prioritization'+str(prioritization)+'-replay_strategy'+str(replay_strategy)+\
                 '-n_epochs'+str(n_epochs)+'-num_cpu'+str(num_cpu)+'-seed'+str(seed)+\
                 '-n_cycles'+str(n_cycles)+'-rank_method'+str(rank_method)+\
                 '-w_potential'+str(w_potential)+'-w_linear'+str(w_linear)+'-w_rotational'+str(w_rotational)+\
                 '-clip_energy'+str(clip_energy)+\
                 '-version'+str(version)
    else:
        logdir = osp.join(tempfile.gettempdir(),
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))

    if save_path:
        logdir = save_path



    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    params['temperature'] = temperature
    params['prioritization'] = prioritization
    params['binding'] = binding
    params['max_timesteps'] = n_epochs * params['n_cycles'] *  params['n_batches'] * num_cpu
    params['version'] = version
    params['dump_buffer'] = dump_buffer
    params['n_cycles'] = n_cycles
    params['rank_method'] = rank_method
    params['w_potential'] = w_potential
    params['w_linear'] = w_linear
    params['w_rotational'] = w_rotational
    params['clip_energy'] = clip_energy
    params['n_epochs'] = n_epochs
    params['num_cpu'] = num_cpu
    params['eval_probs'] = None
    params['train_probs'] = None

    if params['dump_buffer']:
        params['alpha'] =0

    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)

    if load_path:
        # Load policy.
        with open(load_path, 'rb') as f:
            policy = pickle.load(f)
            print("Policy loaded from {}".format(load_path))

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
    rollout_worker.seed(rank_seed)

    evaluators = list()
    evaluators_names = list()
    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    if params["train_probs"]:
        train_dict = dict()
        train_dict["probs"] = params["train_probs"]
        rollout_worker.adapt_env(train_dict)
        evaluator.adapt_env(train_dict)
        print("Train mode probs: {}".format(train_dict["probs"]))
    evaluator.seed(rank_seed)
    evaluators.append(evaluator)
    evaluators_names.append("train")
    if params["eval_probs"]:
        for eval_mode, probs in params["eval_probs"].items():
            evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
            eval_dict = dict()
            eval_dict["probs"] = probs
            evaluator.adapt_env(eval_dict)
            print("Eval mode: {} with probs {}".format(eval_mode, probs))
            evaluator.seed(rank_seed)
            evaluators.append(evaluator)
            evaluators_names.append(eval_mode)


    train(
        logdir=logdir, policy=policy, rollout_worker=rollout_worker,
        evaluators=evaluators, evaluators_names=evaluators_names, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, save_policies=save_policies,
        num_cpu=num_cpu, dump_buffer=dump_buffer, w_potential=params['w_potential'], 
        w_linear=params['w_linear'], w_rotational=params['w_rotational'], rank_method=rank_method,
        clip_energy=clip_energy)


@click.command()
@click.option('--env_name', type=str, default='FetchPickAndPlace-v1', help='the name of the OpenAI Gym \
        environment that you want to train on. We tested EBP on four challenging robotic manipulation tasks, including: \
        FetchPickAndPlace-v1, HandManipulateBlockFull-v1, HandManipulateEggFull-v1, HandManipulatePenRotate-v1, FetchPickAndThrow-v1, FetchSlide-v1')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'final', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--temperature', type=float, default=1.0, help='temperature value for Enery-Based Prioritization (EBP)')
@click.option('--prioritization', type=click.Choice(['none', 'energy', 'tderror']), default='energy', help='the prioritization strategy to be used. "energy" uses EBP;\
                                                                                                             "none" is vanilla HER; tderror is Prioritized Experience Replay.')
@click.option('--binding', type=click.Choice(['none', 'core']), default='core', help='configure mpi using bind-to none or core.')
@click.option('--logging', type=bool, default=False, help='whether or not logging')
@click.option('--version', type=int, default=0, help='version')
@click.option('--dump_buffer', type=bool, default=False, help='dump buffer contains achieved goals, energy, tderrors for analysis')
@click.option('--n_cycles', type=int, default=50, help='n_cycles')
@click.option('--rank_method', type=click.Choice(['none', 'min', 'dense', 'average']), default='none', help='energy ranking method')
@click.option('--w_potential', type=float, default=1.0, help='w_potential')
@click.option('--w_linear', type=float, default=1.0, help='w_linear')
@click.option('--w_rotational', type=float, default=1.0, help='w_rotational')
@click.option('--clip_energy', type=float, default=999, help='clip_energy')
@click.option('--save_path', type=str, default=None, help='save_path')
@click.option('--load_path', type=str, default=None, help='load_path')


def main(**kwargs):
    launch(**kwargs)

if __name__ == '__main__':
    main()
