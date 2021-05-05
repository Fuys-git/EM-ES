#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:37:26 2019

@author: weetee
"""
import os
import math
import torch
import torch.nn as nn
from ..misc import save_as_pickle, load_pickle
from seqeval.metrics import precision_score, recall_score, f1_score
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "/my_model/5shot/"
    amp_checkpoint = None
    checkpoint_path = os.path.join(base_path,"task_test_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"task_test_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        amp_checkpoint = checkpoint['amp']
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred, amp_checkpoint

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./my_model/5shot/task_test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./my_model/5shot/task_train_accuracy_per_epoch_%d.pkl" % model_no
    f1_path = "./my_model/5shot/task_test_f1_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path) and os.path.isfile(f1_path):
        losses_per_epoch = load_pickle("task_test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("task_train_accuracy_per_epoch_%d.pkl" % model_no)
        f1_per_epoch = load_pickle("task_test_f1_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch, f1_per_epoch = [], [], []
    return losses_per_epoch, accuracy_per_epoch, f1_per_epoch


def evaluate_(predic_labels , true_labels, ignore_idx):
    ### ignore index 0 (padding) when calculating accuracy
    #idxs = (labels != ignore_idx).squeeze()
    hits=0
    pred=[]
    for k, predic in enumerate(predic_labels):
        pred.append(predic_labels[k].argmax().cpu().item())
    for i,true in enumerate(true_labels):
        if (pred[i] == true_labels[i]):
            hits += 1
    accuracy = hits / (k + 1)
    return accuracy


def evaluate_results(net, test_loader, pad_id, cuda):
    logger.info("Evaluating test samples...")
    acc = 0; out_labels = []; true_labels = []
    hits=0
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, e1_e2_start, labels,_,_,_ = data #labels(batch_size,26)
            M = x.shape[1]  ## M=26
            batch_size = x.shape[0]  ##batch_size
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            e1_e2_start = e1_e2_start.reshape(e1_e2_start.shape[0] * e1_e2_start.shape[1], e1_e2_start.shape[2])

            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                
            outputs = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, \
                          e1_e2_start=e1_e2_start)#Q=None,

            outputs = outputs.reshape(batch_size, M, outputs.shape[1])  # (batch_size,26,1536)
            anchor_output = outputs[:, :(M - 1), :]  # (batch_size,25,1536) 前5个句子meta_train_input
            target_output = outputs[:, M - 1, :]  # (batch_size, 1536) 最后一个句子meta_test_input
            target_output = target_output.reshape(target_output.shape[0], 1,
                                                  target_output.shape[1])  # (batch_size, 1, 1536)
            # test句的预测
            target_matrix_product = torch.matmul(target_output, anchor_output.permute(0, 2,1))  # (batch_size,1,25) ; anchor_output.permute(0,2,1)->(batch_size,1536,5)
            #similar_scores = []

            predic_labels = target_matrix_product[0]  # (1,25)
            N = labels[0][M - 1].item() + 1
            K = 5
            true_labels = [N - 1, N - 1, N - 1, N - 1, N - 1]  # (1) 先转为list再转为tensor
            for j, x in enumerate(target_matrix_product):
                # similar_scores.append(target_matrix_product[j].argmax().cpu().item())
                if (j != 0):
                    # predic_labels= torch.cat( (predic_labels, target_matrix_product[j]/torch.sum(target_matrix_product[j])), axis=0)
                    predic_labels = torch.cat((predic_labels, target_matrix_product[j]), axis=0)
                    true_labels = true_labels + [N - 1, N - 1, N - 1, N - 1, N - 1]
            true_labels = torch.LongTensor(true_labels)  # (1,5)
            predic_labels = predic_labels.reshape(N, K).T  # (5,5)
            acc += evaluate_(predic_labels , true_labels, ignore_idx=-1)
        Acc = acc / (i + 1)
    results = {
        "accuracy": Acc,
    }
    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    path = r'/home/fuys/mtb/fr2/result/result_5shot.txt'
    doc = open(path, 'a')
    doc.write('evaluate' + '\n')
    for k, v in results.items():
        s2 = str(v)
        doc.write(k + '\n')
        doc.write(s2 + '\n')
    return results