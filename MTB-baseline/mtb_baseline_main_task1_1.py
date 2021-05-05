#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:40:16 2019

@author: weetee
"""
from src.tasks.trainer import train_and_fit
from src.tasks.infer import infer_from_trained, FewRel
import logging
from argparse import ArgumentParser

'''
This fine-tunes the BERT model on SemEval, FewRel tasks
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='FewRel', help='SemEval, KBP37, TACRED, or FewRel')
    parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=0, help="number of relation classes, automatically added by program")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")#3
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=10, help="No of epochs")#15
    parser.add_argument("--lr", type=float, default=0.00003, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, \
                        help='''Model ID: 0 - BERT\n
                                          1 - ALBERT\n
                                          2 - RoBERTa\n
                                          3 - BERT-large''')
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--infer", type=int, default=0, help="0: Don't infer, 1: Infer")
    parser.add_argument("--only_evaluate", type=int, default=0, \
                        help='''1: only evaluate, do not train\n
                                2: only evaluate a part of data''')
    parser.add_argument("--rel2idx", type=dict, help="Automatically added by program")
    parser.add_argument("--idx2rel", type=dict, help="Automatically added by program")
    parser.add_argument("--no_relation", type=str, help="Automatically added by program")
    args = parser.parse_args()
    if args.task == 'SemEval':
        args.num_classes = 19
    elif args.task == 'KBP37':
        args.num_classes = 38
    elif args.task == 'TACRED':
        args.num_classes = 42
    elif args.task == 'FewRel':
        args.num_classes = 80
    else:
        raise ValueError('args.task error')
    logging.info('Arguments:')
    for arg in vars(args):
        logging.info('    {}: {}'.format(arg, getattr(args, arg)))

    if (args.train == 1) and (args.task != 'fewrel'):
        net = train_and_fit(args)
