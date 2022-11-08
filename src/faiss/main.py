import numpy as np
import faiss  
import pandas as pd
from ogb.lsc import WikiKG90Mv2Dataset
import sys
import os
import argparse
import time

datapath = '/db2/users/minjunpark/ogb/faiss/processed_data'

# Argparse
class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument(
            '--datapath',
            type=str,
            default='/db2/users/minjunpark/ogb/faiss/processed_data',
            help='Path to processed faiss data.',
        )
        
        self.add_argument(
            '--task',
            type=str,
            default='valid',
            help='Task type for data process.',
        )
        
        self.add_argument(
            '--num_cand',
            type=int,
            default=1000,
            help='Task type for data process.',
        )

    def parse_args(self):
        args = super().parse_args()

        return args

args = ArgParser().parse_args()

def get_candidates(k, index, entity_feat_subs, name):
    print('Getting candidates...')
    start_time = time.time()
    D, I = index.search(entity_feat_subs, k) # sanity check
    print("--- %s seconds to get candidates---" % (time.time() - start_time))
    
    np.save(f'{args.datapath}/I_{name}_candidates_{args.num_cand}', I)
    np.save(f'{args.datapath}/D_{name}_candidates_{args.num_cand}', D)
    return None

def run():
    print(f'task type: {args.task}')
    # Get data
    print('Load entity features...')
    entity_feat = np.load(f'{args.datapath}/entity_feat_float32.npy')
    print('Successfully loaded!')
    
    rootdir = '/db2/users/minjunpark/ogb/rawdata'
    dataset = WikiKG90Mv2Dataset(root=str(rootdir))
    
    if args.task == 'valid':
        # Validation dataset
        dic = dataset.valid_dict['h,r->t'] # get a dictionary storing the h,r->t task.
        hr = dic['hr']
        h = hr[:,0]
        
    if args.task == 'valid_tail_drop_duplicates':
        train_task = dataset.train_hrt
        df = pd.DataFrame(train_task, columns=['Head', 'Relation', 'Tail'])
        df_uniq = df.drop_duplicates(subset='Head').reset_index(drop=True)
        
        dic = dataset.valid_dict['h,r->t'] # get a dictionary storing the h,r->t task.
        hr = dic['hr']
        h2 = hr[:,0]
        
        h = []
        for i in h2:
            # If there's no information for head
            if i not in df_uniq.Head.to_numpy():
                print(f'No head: {i}')
                h.append(i)
            else:
                v = df_uniq[df_uniq.Head==i].Tail.tolist()[0]
                h.append(v)
                
        h = np.array(h)
        # h = np.array([df_uniq[df_uniq.Head==i].Tail.tolist()[0] for i in h2])
        
    if args.task == 'challenge':
        dic = dataset.test_dict(mode = 'test-challenge')['h,r->t']['hr']
        h = dic[:,0]
        
    if args.task == 'test':
        dic = dataset.valid_dict['h,r->t'] # get a dictionary storing the h,r->t task.
        hr = dic['hr']
        h = hr[:,0][:100]
        args.num_cand = 5

    # Define query vectors
    entity_feat_subs = entity_feat[[h.tolist()]]
    
    # Build index
    d = 768 
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(entity_feat)                  # add vectors to the index
    print(index.ntotal)
    
    # Find candidates
    get_candidates(args.num_cand, index, entity_feat_subs, args.task)
    
if __name__ == '__main__':
    sys.exit(run())