# -*- coding: utf-8 -*-
#
# sampler.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import scipy as sp
import dgl.backend as F
import dgl
import numpy as np
import operator
import sys
import pickle
import time
import math

# class ChunkNegEdgeSubgraph(dgl.DGLGraph):
#     """Wrapper for negative graph
#         Parameters
#         ----------
#         neg_g : DGLGraph
#             Graph holding negative edges.
#         num_chunks : int
#             Number of chunks in sampled graph.
#         chunk_size : int
#             Info of chunk_size.
#         neg_sample_size : int
#             Info of neg_sample_size.
#         neg_head : bool
#             If True, negative_mode is 'head'
#             If False, negative_mode is 'tail'
#     """
#     def __init__(self, subg, num_chunks, chunk_size,
#                  neg_sample_size, neg_head):
#         super(ChunkNegEdgeSubgraph, self).__init__(graph_data=subg.sgi.graph,
#                                                    readonly=True,
#                                                    parent=subg._parent)
#         self.ndata[NID] = subg.sgi.induced_nodes.tousertensor()
#         self.edata[EID] = subg.sgi.induced_edges.tousertensor()
#         self.subg = subg
#         self.num_chunks = num_chunks
#         self.chunk_size = chunk_size
#         self.neg_sample_size = neg_sample_size
#         self.neg_head = neg_head

#     @property
#     def head_nid(self):
#         return self.subg.head_nid

#     @property
#     def tail_nid(self):
#         return self.subg.tail_nid

# def create_neg_subgraph(pos_g, neg_g, chunk_size, neg_sample_size, is_chunked,
#                         neg_head, num_nodes):
#     """KG models need to know the number of chunks, the chunk size and negative sample size
#     of a negative subgraph to perform the computation more efficiently.
#     This function tries to infer all of these information of the negative subgraph
#     and create a wrapper class that contains all of the information.
#     Parameters
#     ----------
#     pos_g : DGLGraph
#         Graph holding positive edges.
#     neg_g : DGLGraph
#         Graph holding negative edges.
#     chunk_size : int
#         Chunk size of negative subgrap.
#     neg_sample_size : int
#         Negative sample size of negative subgrap.
#     is_chunked : bool
#         If True, the sampled batch is chunked.
#     neg_head : bool
#         If True, negative_mode is 'head'
#         If False, negative_mode is 'tail'
#     num_nodes: int
#         Total number of nodes in the whole graph.
#     Returns
#     -------
#     ChunkNegEdgeSubgraph
#         Negative graph wrapper
#     """
#     assert neg_g.number_of_edges() % pos_g.number_of_edges() == 0
#     # We use all nodes to create negative edges. Regardless of the sampling algorithm,
#     # we can always view the subgraph with one chunk.
#     if (neg_head and len(neg_g.head_nid) == num_nodes) \
#             or (not neg_head and len(neg_g.tail_nid) == num_nodes):
#         num_chunks = 1
#         chunk_size = pos_g.number_of_edges()
#     elif is_chunked:
#         # This is probably for evaluation.
#         if pos_g.number_of_edges() < chunk_size \
#                 and neg_g.number_of_edges() % neg_sample_size == 0:
#             num_chunks = 1
#             chunk_size = pos_g.number_of_edges()
#         # This is probably the last batch in the training. Let's ignore it.
#         elif pos_g.number_of_edges() % chunk_size > 0:
#             return None
#         else:
#             num_chunks = int(pos_g.number_of_edges() / chunk_size)
#         assert num_chunks * chunk_size == pos_g.number_of_edges()
#     else:
#         num_chunks = pos_g.number_of_edges()
#         chunk_size = 1
#     return ChunkNegEdgeSubgraph(neg_g, num_chunks, chunk_size,
#                                 neg_sample_size, neg_head)


class WikiEvalSampler(object):
    """Sampler for validation and testing for wikikg90M dataset
    Parameters
    ----------
    edges : tensor
        sampled test data
    batch_size : int
        Batch size of each mini batch.
    mode : str
        Sampling mode.
    """

    def __init__(self, edges, batch_size, candidate_index):
        self.edges = edges
        self.batch_size = batch_size
        self.candidate_index = candidate_index
        self.cnt = 0
        self.num_edges = len(self.edges['h,r->t']['hr'])

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch
        Returns
        -------
        tensor of size (batch_size, 2)
            sampled head and relation pair
        tensor of size (batchsize, 1)
            the index of the true tail entity
        tensor of size (bath_size, num_candidates)
            candidates from faiss
        """
        if self.cnt == self.num_edges:
            raise StopIteration
        beg = self.cnt
        if self.cnt + self.batch_size > self.num_edges:
            self.cnt = self.num_edges
        else:
            self.cnt += self.batch_size

        query = F.tensor(self.edges['h,r->t']['hr'][beg : self.cnt], F.int64)
        ans = F.tensor(self.edges['h,r->t']['t_correct_index'][beg : self.cnt], F.int64)
        candidates = F.tensor(
            operator.itemgetter(*self.candidate_index)(self.edges['h,r->t']['t_candidate'])[beg : self.cnt], F.int64
        )

        return query, ans, candidates

    def reset(self):
        """Reset the sampler"""
        self.cnt = 0
        return self


class EvalDataset(object):
    """Dataset for validation or testing
    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    """

    def __init__(self, g, dataset, args):
        self.num_train = len(dataset.train_hrt[:, 0])
        self.valid_dict = dataset.valid_dict
        self.num_valid = len(self.valid_dict['h,r->t']['hr'])
        self.test_dict = dataset.test_dict(mode='test-dev')
        self.num_test = len(self.test_dict['h,r->t']['hr'])
        self.challenge_dict = dataset.test_dict(mode = 'test-challenge')
        self.num_challenge = len(self.challenge_dict['h,r->t']['hr'])
        self.args = args

        print('|valid|:', self.num_valid)
        print('|test|:', self.num_test)
        print('|challenge|:', self.num_challenge)

        self.g = g

    def get_dicts(self):
        """Get all edges dict in this dataset
        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        Returns
        -------
        dict
            all edges
        """
        if self.args.task_type == 'valid':
            return self.valid_dict
        elif self.args.task_type == 'test':
            return self.test_dict
        elif self.args.task_type == 'challenge':
            return self.challenge_dict
        else:
            raise Exception('get invalid type: ' + self.args.task_type)

    def create_sampler_wikikg90Mv2(self, batch_size, rank=0, ranks=1):
        """Create sampler for validation and testing of wikikg90M dataset.
        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        batch_size : int
            Batch size of each mini batch.
        mode : str
            Sampling mode.
        rank : int
            Which partition to sample.
        ranks : int
            Total number of partitions.
        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        edges = self.get_dicts()
        new_edges = {}
        num_entities = 91230610

        """
        This function will split the edges into total number of partitions parts. And then calculate the
        corresponding begin and end index for each part to create evaluate sampler.
        """
        beg = edges['h,r->t']['hr'].shape[0] * rank // ranks
        end = min(edges['h,r->t']['hr'].shape[0] * (rank + 1) // ranks, edges['h,r->t']['hr'].shape[0])
        basepath = '/db2/users/minjunpark/ogb/faiss/processed_data'

        if self.args.task_type == 'valid':
            new_edges['h,r->t'] = {
                'hr': edges['h,r->t']['hr'][beg:end],
                't_correct_index': edges['h,r->t']['t'][beg:end],
                't_candidate': np.arange(0, num_entities),
            }
            validfile = 'I_valid_candidates.npy'
            # validfile = 'I_valid_candidates_100000.npy'
            
            valid_candidates = np.load(f'{basepath}/{validfile}')
            candidate_index = valid_candidates[beg:end, :]

        elif self.args.task_type == 'challenge':
            new_edges['h,r->t'] = {
                'hr': edges['h,r->t']['hr'][beg:end],
                't_correct_index': np.arange(beg, end),
                't_candidate': np.arange(0, num_entities),
            }

            basepath = '/db2/users/minjunpark/ogb/faiss/processed_data'
            # validfile = 'I_challenge_candidates.npy'
            validfile = 'I_challenge_candidates_100000.npy'
            valid_candidates = np.load(f'{basepath}/{validfile}')
            candidate_index = valid_candidates[beg:end, :]

        return WikiEvalSampler(new_edges, batch_size, candidate_index)
