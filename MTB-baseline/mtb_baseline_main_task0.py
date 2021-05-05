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
    parser.add_argument("--task", type=str, default='fewrel')
    # parser.add_argument("--train_data", type=str, default='./data/SemEval/train.txt')
    # parser.add_argument("--test_data", type=str, default='./data/SemEval/test.txt')
    parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=42, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=11, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00003, help="learning rate")
    parser.add_argument("--model_no", type=int, default=2, \
                           help='''Model ID: 0 - BERT\n
                                             1 - ALBERT\n
                                             2 - RoBERTa\n
                                             3 - BERT-large''')
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--infer", type=int, default=0, help="0: Don't infer, 1: Infer")
    
    args = parser.parse_args()
    logging.info('Arguments:')
    for arg in vars(args):
        logging.info('    {}: {}'.format(arg, getattr(args, arg)))

    #if (args.train == 1) and (args.task != 'fewrel'):
    #    net = train_and_fit(args)

    if args.task == 'fewrel':
        fewrel = FewRel(args)
        meta_input, e1_e2_start, meta_labels, outputs = fewrel.evaluate()
