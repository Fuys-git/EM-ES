#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:12:22 2019

@author: weetee
"""
import os
import re
import random
import copy
import pandas as pd
import json
import torch
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ..misc import save_as_pickle, load_pickle
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def process_text(text):
    sents, relations, comments, blanks = [], [], [], []
    for i in range(int(len(text)/4)):
        sent = text[4*i]
        relation = text[4*i + 1]
        comment = text[4*i + 2]
        blank = text[4*i + 3]
        sent = re.findall("\"(.+)\"", sent)[0]
        sent = re.sub('<e1>', '[E1]', sent)
        sent = re.sub('</e1>', '[/E1]', sent)
        sent = re.sub('<e2>', '[E2]', sent)
        sent = re.sub('</e2>', '[/E2]', sent)
        sents.append(sent); relations.append(relation), comments.append(comment); blanks.append(blank)
    return sents, relations, comments, blanks

def preprocess_semeval2010_8(args):
    '''
    Data preprocessing for SemEval2010 task 8 dataset
    '''
    data_path = './data/' + args.task + '/train.txt'
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    sents, relations, comments, blanks = process_text(text)
    df_train = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    data_path = './data/' + args.task + '/test.txt'
    logger.info("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        text = f.readlines()
    
    sents, relations, comments, blanks = process_text(text)
    df_test = pd.DataFrame(data={'sents': sents, 'relations': relations})
    
    rm = Relations_Mapper(df_train['relations'])
    save_as_pickle('./data/{}/relations.pkl'.format(args.task), rm)
    df_test['relations_id'] = df_test.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    df_train['relations_id'] = df_train.progress_apply(lambda x: rm.rel2idx[x['relations']], axis=1)
    save_as_pickle('./data/{}/df_train.pkl'.format(args.task), df_train)
    save_as_pickle('./data/{}/df_test.pkl'.format(args.task), df_test)
    logger.info("Finished and saved!")
    
    return df_train, df_test, rm

class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}
        
        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations):
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1
        
        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key

class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=-1, label2_pad_value=-1,\
                 ):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])
        
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])
        
        return seqs_padded, labels_padded, labels2_padded, \
                x_lengths, y_lengths, y2_lengths

def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],\
                        [i for i, e in enumerate(x) if e == e2_id][0])
    except Exception as e:
        e1_e2_start = None
        print(e)
    return e1_e2_start

class semeval_dataset(Dataset):
    def __init__(self, df, tokenizer, e1_id, e2_id, type, task, model):
        if type != 'train' and type != 'test':
            raise ValueError("'type' should be 'train' or 'test'")

        self.e1_id = e1_id
        self.e2_id = e2_id
        self.df = df
        logger.info("Read tokenized data or tokenize data...")

        _path = './data/{}/tokenized_{}_for_model{}.pkl'.format(task, type, model)
        if not os.path.exists(_path):
            logger.info('Begin tokenizing data...')
            self.df['input'] = self.df.progress_apply(lambda x: tokenizer.encode(x['sents']), \
                                                      axis=1)
            with open(_path, 'wb') as f:
                pkl.dump(self.df['input'], f)

        logger.info('Read tokenized data from {}'.format(_path))
        with open(_path, 'rb') as f:
            self.df['input'] = pkl.load(f)
        self.df['e1_e2_start'] = self.df.progress_apply(lambda x: get_e1e2_start(x['input'],\
                                                       e1_id=self.e1_id, e2_id=self.e2_id), axis=1)
        print("\nInvalid rows/total: %d/%d" % (df['e1_e2_start'].isnull().sum(), len(df)))
        self.df.dropna(axis=0, inplace=True)
    
    def __len__(self,):
        return len(self.df)
        
    def __getitem__(self, idx):
        return torch.LongTensor(self.df.iloc[idx]['input']),\
                torch.LongTensor(self.df.iloc[idx]['e1_e2_start']),\
                torch.LongTensor([self.df.iloc[idx]['relations_id']])


def load_dataloaders(args):
    if args.model_no == 0:
        from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer
        model_name = 'bert'
    elif args.model_no == 1:
        from ..model.ALBERT.tokenization_albert import AlbertTokenizer as Tokenizer
        model_name = 'albert'
    elif args.model_no == 2:
        from ..transformers.models.roberta.tokenization_roberta import RobertaTokenizer as Tokenizer
        model_name = 'roberta'
    elif args.model_no == 3:
        # from ..gaojie_transformers.models.bert.tokenization_bert import BertTokenizer as Tokenizer
        from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer
        model_name = 'bert_large'
        
    if os.path.isfile("./pretrained/{}/{}_tokenizer.pkl".format(model_name, model_name)):
        tokenizer = load_pickle("./pretrained/{}/{}_tokenizer.pkl".format(model_name, model_name))
        logger.info("Loaded tokenizer from pre-trained tokenizer")
    else:
        logger.info("Pre-trained tokenizer not found, initializing new tokenizer...")
        tokenizer = Tokenizer.from_pretrained("./pretrained/{}".format(model_name))
        print(tokenizer.convert_ids_to_tokens(100))
        tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])
        save_as_pickle("./pretrained/{}/{}_tokenizer.pkl".format(model_name, model_name), tokenizer)
        logger.info("Saved tokenizer at ./pretrained/{}/{}_tokenizer.pkl".format(model_name, model_name))
    
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1
    
    if args.task == 'SemEval' or args.task == 'TACRED' or args.task == 'KBP37' or args.task == 'FewRel':
        relations_path = './data/{}/relations.pkl'.format(args.task)
        train_path = './data/{}/df_train.pkl'.format(args.task)
        test_path = './data/{}/df_test.pkl'.format(args.task)
        if os.path.isfile(relations_path) and os.path.isfile(train_path) and os.path.isfile(test_path):
            rm = load_pickle('./data/{}/relations.pkl'.format(args.task))
            df_train = load_pickle('./data/{}/df_train.pkl'.format(args.task))
            df_test = load_pickle('./data/{}/df_test.pkl'.format(args.task))
            logger.info("Loaded preproccessed data.")
        else:
            df_train, df_test, rm = preprocess_semeval2010_8(args)
        
        train_set = semeval_dataset(df_train, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id, type='train', task=args.task, model=args.model_no)
        test_set = semeval_dataset(df_test, tokenizer=tokenizer, e1_id=e1_id, e2_id=e2_id, type='test', task=args.task, model=args.model_no)
        train_length = len(train_set); test_length = len(test_set)
        PS = Pad_Sequence(seq_pad_value=tokenizer.pad_token_id,\
                          label_pad_value=tokenizer.pad_token_id,\
                          label2_pad_value=-1)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, \
                                  num_workers=0, collate_fn=PS, pin_memory=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, \
                                  num_workers=0, collate_fn=PS, pin_memory=False)
        
    return train_loader, test_loader, train_length, test_length