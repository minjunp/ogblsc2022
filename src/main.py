import numpy as np
import pandas as pd
from ogb.lsc import WikiKG90Mv2Dataset
import sys
import os
import time

from scipy.sparse import coo_matrix
import dgl.backend as F
import dgl

from model.model import KEModel
import argparse

from evaluator import sampler
import math

import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import torch as th
from functools import wraps
import traceback
from _thread import start_new_thread
from tqdm import tqdm
from evaluator.evaluator import Evaluator

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument(
            '--model_name',
            default='TransE',
            choices=[
                'TransE',
                'TransE_l1',
                'TransE_l2',
                'TransR',
                'RESCAL',
                'DistMult',
                'ComplEx',
                'RotatE',
                'SimplE',
            ],
            help='The models provided by DGL-KE.',
        )
        
        self.add_argument(
            '--task_type',
            type=str,
            default='valid',
            help='Task to perform: valid or test',
        )
        
        self.add_argument(
            '--save_path',
            type=str,
            default='/db2/users/minjunpark/ogb/ogblsc2022/src/output',
            help='Path to save model output'
        )
        
        self.add_argument(
            '--data_path',
            type=str,
            default='data',
            help='The path of the directory where DGL-KE loads knowledge graph data.',
        )
        self.add_argument(
            '--dataset',
            type=str,
            default='FB15k',
            help='The name of the builtin knowledge graph. Currently, the builtin knowledge '
            'graphs include FB15k, FB15k-237, wn18, wn18rr and Freebase. '
            'DGL-KE automatically downloads the knowledge graph and keep it under data_path.',
        )
        self.add_argument(
            '--format',
            type=str,
            default='built_in',
            help='The format of the dataset. For builtin knowledge graphs,'
            'the foramt should be built_in. For users own knowledge graphs,'
            'it needs to be raw_udd_{htr} or udd_{htr}.',
        )
        self.add_argument(
            '--model_path', type=str, default='ckpts', help='The path of the directory where models are saved.'
        )
        self.add_argument('--batch_size_eval', type=int, default=8, help='The batch size used for evaluation.')
        self.add_argument(
            '--neg_sample_size_eval', type=int, default=-1, help='The negative sampling size for evaluation.'
        )
        self.add_argument(
            '--neg_deg_sample_eval',
            action='store_true',
            help='Negative sampling proportional to vertex degree for evaluation.',
        )
        self.add_argument('--hidden_dim', type=int, default=256, help='The hidden dim used by relation and entity')
        self.add_argument(
            '-g',
            '--gamma',
            type=float,
            default=12.0,
            help='The margin value in the score function. It is used by TransX and RotatE.',
        )
        self.add_argument('--eval_percent', type=float, default=1, help='The percentage of data used for evaluation.')
        self.add_argument(
            '--no_eval_filter',
            action='store_true',
            help='Disable filter positive edges from randomly constructed negative edges for evaluation',
        )
        self.add_argument('--gpu', type=int, default=-1, help='a list of active gpu ids, e.g. 0')
        self.add_argument(
            '--mix_cpu_gpu',
            action='store_true',
            help='Evaluate a knowledge graph embedding model with both CPUs and GPUs.'
            'The embeddings are stored in CPU memory and the training is performed in GPUs.'
            'This is usually used for training a large knowledge graph embeddings.',
        )
        self.add_argument(
            '-de',
            '--double_ent',
            action='store_true',
            help='Double entitiy dim for complex number It is used by RotatE.',
        )
        self.add_argument('-dr', '--double_rel', action='store_true', help='Double relation dim for complex number.')
        self.add_argument(
            '--num_proc',
            type=int,
            default=1,
            help='The number of processes to evaluate the model in parallel.'
            'For multi-GPU, the number of processes by default is set to match the number of GPUs.'
            'If set explicitly, the number of processes needs to be divisible by the number of GPUs.',
        )
        self.add_argument(
            '--num_thread',
            type=int,
            default=1,
            help='The number of CPU threads to evaluate the model in each process.'
            'This argument is used for multiprocessing computation.',
        )

        self.add_argument(
            '-f', '--file', help='Path for input file. First line should contain number of lines to search in'
        )

    def parse_args(self):
        args = super().parse_args()

        return args


def get_compatible_batch_size(batch_size, neg_sample_size):
    if neg_sample_size < batch_size and batch_size % neg_sample_size != 0:
        old_batch_size = batch_size
        batch_size = int(math.ceil(batch_size / neg_sample_size) * neg_sample_size)
        print(
            'batch size ({}) is incompatible to the negative sample size ({}). Change the batch size to {}'.format(
                old_batch_size, neg_sample_size, batch_size
            )
        )
    return batch_size


def thread_wrapped_func(func):
    """Wrapped func for torch.multiprocessing.Process.
    With this wrapper we can use OMP threads in subprocesses
    otherwise, OMP_NUM_THREADS=1 is mandatory.
    How to use:
    @thread_wrapped_func
    def func_to_wrap(args ...):
    """

    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()

        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)

    return decorated_function


@thread_wrapped_func
def test_mp(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test(args, model, test_samplers, rank, mode, queue)


def test(args, model, test_samplers, rank=0, queue=None):
    if args.gpu > 0:
        gpu_id = args.gpu[rank % args.gpu] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu
    else:
        gpu_id = -1
    print(f'gpu_id: {gpu_id}')
    print(f'Task type: {args.task_type}')

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    with th.no_grad():
        logs = []
        answers = []
        top10s_all = []
        for sampler in test_samplers:
            query = sampler[0]
            ans = sampler[1]
            candidate = sampler[2]
            
            top10s = model.forward_test_wikikg(query, ans, candidate, args.task_type, logs, gpu_id)
            top10s_all.append(top10s)
            answers.append(ans)
        
        print("[{}] finished {} forward".format(rank, args.task_type))
        
        top10_predictions = np.concatenate(top10s_all, axis=0)
        true_labels = np.concatenate(answers, axis=0)

        test_samplers = test_samplers.reset()
        
        if args.task_type == "valid":
            dir_path = '/db2/users/minjunpark/ogb/ogblsc2022/src/output'
            eval = Evaluator(top10_predictions, true_labels, dir_path)
            
            mrr = eval.eval_func()
            print(f'MRR: {mrr}')
            
            metrics = {}
            if len(logs) > 0:
                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            if queue is not None:
                queue.put(logs)
            else:
                for k, v in metrics.items():
                    print('[{}]{} average {}: {}'.format(rank, args.task_type, k, v))
                    
        elif args.task_type == 'challenge':
            dir_path = '/db2/users/minjunpark/ogb/ogblsc2022/src/submission/v2'
            eval = Evaluator(top10_predictions, true_labels, dir_path)
            eval.submit_result()
            
        elif args.task_type == 'test':     
            input_dict = {}
            input_dict['h,r->t'] = {'t_correct_index': th.cat(answers, 0), 't_pred_top10': th.cat(logs, 0)}
            th.save(input_dict, os.path.join(args.save_path, f"{args.task_type}_{rank}.pkl"))
        else:
            raise NameError(f'No task type {args.task_type}')


def run():
    if th.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f'Device using: {device}')

    rootdir = '/db2/users/minjunpark/ogb/rawdata'
    dataset = WikiKG90Mv2Dataset(root=str(rootdir))

    print(dataset.num_entities)  # number of entities -- > 91230610
    print(dataset.num_relations)  # number of relation types --> 1387
    print(dataset.num_feat_dims)  # dimensionality of entity/relation features.

    src = dataset.train_hrt[:, 0]
    etype_id = dataset.train_hrt[:, 1]
    dst = dataset.train_hrt[:, 2]
    n_entities = dataset.num_entities

    coo = coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])

    g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)

    g.edata['tid'] = F.tensor(etype_id, F.int64)

    args = ArgParser().parse_args()
    args.train = False
    args.valid = False
    args.test = True
    args.strict_rel_part = False
    args.soft_rel_part = False
    args.async_update = False
    args.has_edge_importance = False
    args.eval_filter = not args.no_eval_filter

    def load_model_from_checkpoint(args, n_entities, n_relations, ckpt_path):
        model = load_model(args, n_entities, n_relations)
        model.load_emb(ckpt_path, args.dataset)
        return model

    def load_model(args, n_entities, n_relations):
        model = KEModel(
            args,
            args.model_name,
            n_entities,
            n_relations,
            args.hidden_dim,
            args.gamma,
            double_entity_emb=args.double_ent,
            double_relation_emb=args.double_rel,
        )
        return model

    # load model
    n_entities = dataset.num_entities
    n_relations = dataset.num_relations
    ckpt_path = '/db2/users/minjunpark/ogb/ogblsc2022/ckpts/wikikg90m-v2'

    print('Loading model from checkpoint...')
    model = load_model_from_checkpoint(args, n_entities, n_relations, ckpt_path)
    print(sum(p.numel() for p in model.parameters()))
    print('Model loaded successfully!')

    # get Eval dataset
    args.eval_percent = 1
    eval_dataset = sampler.EvalDataset(g, dataset, args)

    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = args.neg_sample_size = eval_dataset.g.number_of_nodes()
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)

    if args.num_proc > 1:
        test_sampler_tails = []
        for i in range(args.num_proc):
            test_sampler_tail = eval_dataset.create_sampler_wikikg90Mv2(
                args.batch_size_eval,
                rank=i,
                ranks=args.num_proc,
            )
            test_sampler_tails.append(test_sampler_tail)
    else:
        test_sampler_tail = eval_dataset.create_sampler_wikikg90Mv2(
            args.batch_size_eval,
            rank=0,
            ranks=1,
        )

    if args.num_proc > 1:
        model.share_memory()
        
    # test
    args.step = 0
    args.max_step = 0
    start = time.time()
    
    th.multiprocessing.set_start_method('spawn', force=True)
    if args.num_proc > 1:
        # Move data into shared memory and will only send a handle to another process.
        queue = mp.Queue(args.num_proc)
        procs = []
        print('Evaluation in process...')
        for i in range(args.num_proc):
            print(f'num proc: {i}')
            proc = mp.Process(target=test_mp, args=(args, model, test_sampler_tails[i], i, 'Test', queue))
            procs.append(proc)

        metrics = {}
        logs = []
        print('Getting metrics...')
        for i in tqdm(range(args.num_proc)):
            print(f'metric proc: {i}')
            log = queue.get()
            logs = logs + log

        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        print("-------------- Test result --------------")
        for k, v in metrics.items():
            print('Test average {}: {}'.format(k, v))
        print("-----------------------------------------")

        for proc in procs:
            proc.join()
    else:
        test(args, model, test_sampler_tail)
    print('Test takes {:.3f} seconds'.format(time.time() - start))


if __name__ == '__main__':
    sys.exit(run())