import os
import sys
import pickle
import click
import numpy as np
import json
from mpi4py import MPI
from collections import OrderedDict
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork

import os.path as osp
import tempfile
import datetime

def master_send_key_value_pair(num_cpu, id, key, value):
    if num_cpu is None:
        raise Exception("Num cpu is none")
    if key is None:
        raise Exception("key is none")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    assert rank < num_cpu
    if rank > 0:
        [key_rec, value_rec] = comm.recv(source=0, tag=rank+id)
        assert key_rec == key
        #print("Send: {}, id: {}, rank: {}".format(key, id, rank))
        return key_rec, value_rec
    elif rank == 0:
        if num_cpu > 1:
            for i in range(1, num_cpu):
                comm.send([key, value], dest=i, tag=i+id)
                #print("Recv: {}, Self: {}, Id: {}, i: {}".format(data[0], key, id, i))
        return None, None

def slaves_send_key_value_pair(num_cpu, id, key, value):
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
        comm.send(data, dest=0, tag=rank+id)
        #print("Send: {}, id: {}, rank: {}".format(key, id, rank))
        return None, None
    elif rank == 0:
        val_list = list()
        val_list.append(value)
        if num_cpu > 1:
            for i in range(1, num_cpu):
                data = comm.recv(source=i, tag=i+id)
                #print("Recv: {}, Self: {}, Id: {}, i: {}".format(data[0], key, id, i))
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


def train(policy, rollout_workers, evaluators, evaluators_names, min_successes, n_epochs, n_test_rollouts, n_cycles, n_batches,
          policy_save_interval, save_policies, num_cpu, dump_buffer, w_potential, w_linear,
          w_rotational, rank_method, clip_energy, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1
    success_rate = -1
    t = 1

    # initialize rollout_worker and evaluator
    train_index = 0
    rollout_worker = rollout_workers[train_index]

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

        # TODO NEW NEW

        # test
        # unique ids are necessary for mpi communication
        # record logs
        logger.record_tabular('epoch', epoch)
        id = 10
        """
        for key, val in evaluator.logs('test'):
            if 'success_rate' in key:
                print("Test success: {} with history {}".format(val, list(evaluator.success_history)))  # TODO new
            key, mean = send_key_value_pair(num_cpu, id, key, val)
            id += 10;
            if rank == 0:
                logger.record_tabular(key, mean)
        for key, val in rollout_worker.logs('train'):
            if 'success_rate' in key:
                print("Rollout success: {} with history {}".format(val, list(rollout_worker.success_history)))  # TODO new
            key, mean = send_key_value_pair(num_cpu, id, key, val)
            id += 10
            if rank == 0:
                logger.record_tabular(key, mean)
        """
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))


        # TODO NEW SECTION

        for k in range(0, train_index+1): # only evaluate probs that have been trained
            eval = evaluators[k]
            name = evaluators_names[k]
            eval.clear_history()
            for _ in range(n_test_rollouts):
                eval.generate_rollouts()
            # record logs
            for key, val in eval.logs(name):
                if 'success_rate' in key:
                    print("Rank {}: {} success: {}".format(rank, name, val))  # TODO new
                key, mean = slaves_send_key_value_pair(num_cpu, id, key, val)
                id += 10
                if rank == 0:
                    logger.record_tabular(key, mean)
                # update success rate if eval is the one related to training
                if rank == 0 and 'success_rate' in key and k == train_index:
                    success_rate = mean

        # TODO END NEW NEW



        if rank == 0:
            logger.record_tabular('train_index', train_index)
            logger.record_tabular('train_mode', evaluators_names[train_index])
            logger.dump_tabular()
            if dump_buffer:
                policy.dump_buffer(epoch)

        # save the policy if it's better than the previous ones
        #success_rate = mpi_average(evaluator.current_success_rate()) TODO: removed this, replaced by custom calculation
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            rollout_worker.save_policy(best_policy_path) # TODO replaced evaluator by rollout_worker (policies are the same)
            rollout_worker.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            rollout_worker.save_policy(policy_path)

        # Master (rank 0) advance to new training probs if min_success is reached
        if rank == 0 and best_success_rate >= min_successes[train_index] and train_index+1 < len(rollout_workers):
            train_index += 1
            print("Reached min_success {} > {} --> Changing train_probs to {} (unknown success rate)"
                  .format(best_success_rate, min_successes[train_index-1], evaluators_names[train_index]))
            best_success_rate = -1
            rollout_worker = rollout_workers[train_index]

        # Send train_index to slaves, let them confirm
        dummy, new_index = master_send_key_value_pair(num_cpu, 1e8, "train_index", train_index)
        if new_index == train_index+1:
            train_index = new_index
            best_success_rate = -1
            rollout_worker = rollout_workers[train_index]
            print("Rank {}: Changing train_probs to {}".format(rank, evaluators_names[train_index]))


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


    evaluators = list()
    evaluators_names = list()
    rollout_workers = list()
    min_successes = list()


    if params["train_probs"]:
        for train_mode, [probs, min_success] in sorted(params["train_probs"].items()):  # sorted is necessary for mpi communication
            config_dict = dict()
            config_dict["probs"] = probs
            # Rollout workers
            rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
            rollout_worker.adapt_env(config_dict)
            rollout_worker.seed(rank_seed)
            rollout_workers.append(rollout_worker)
            # Evaluators
            evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
            evaluator.adapt_env(config_dict)
            evaluator.seed(rank_seed)
            evaluators.append(evaluator)
            # Name
            evaluators_names.append(train_mode)
            # Min success
            min_successes.append(min_success)
            logger.info("Created Train and Eval mode: {} with probs {} and min_success {}".format(train_mode, probs, min_success))
    else:
        # Default Rollout workers in case no config file is specified
        rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
        rollout_worker.seed(rank_seed)
        rollout_workers.append(rollout_worker)
        evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
        evaluator.seed(rank_seed)
        evaluators.append(evaluator)
        evaluators_names.append("default")
        min_successes.append(1)

    logger.info("Modes: {}".format(evaluators_names))
    logger.info("Min_successes: {}".format(min_successes))

    train(
        logdir=logdir, policy=policy, rollout_workers=rollout_workers,
        evaluators=evaluators, evaluators_names=evaluators_names, min_successes= min_successes, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
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
