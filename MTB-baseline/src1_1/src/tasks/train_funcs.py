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
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import precision_score as sklearn_precision_score
from sklearn.metrics import recall_score as sklearn_recall_score
import logging
from tqdm import tqdm
import numpy as np
from collections import Counter
import sys

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./my_model/"
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
    losses_path = "./my_model/task_test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./my_model/task_train_accuracy_per_epoch_%d.pkl" % model_no
    f1_path = "./my_model/task_test_f1_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path) and os.path.isfile(f1_path):
        losses_per_epoch = load_pickle("./my_model/task_test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("./my_model/task_train_accuracy_per_epoch_%d.pkl" % model_no)
        f1_per_epoch = load_pickle("./my_model/task_test_f1_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch, f1_per_epoch = [], [], []
    return losses_per_epoch, accuracy_per_epoch, f1_per_epoch


def evaluate_(output, labels, ignore_idx):
    ### ignore index 0 (padding) when calculating accuracy
    idxs = (labels != ignore_idx).squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]
    l = labels.squeeze()[idxs]; o = o_labels[idxs]

    if len(idxs) > 1:
        acc = (l == o).sum().item()/len(idxs)
    else:
        acc = (l == o).sum().item()
    l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
    o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

    return acc, (o, l)

def semeval_files(args, true_labels, pred_labels, epoch):
    if not os.path.exists('./semeval_files/'):
        os.makedirs('./semeval_files/')
    f = open("./semeval_files/true_labels_epoch{}.txt".format(epoch), "w")
    for i, true_label in enumerate(true_labels):
        f.write(str(i)+'\t'+args.idx2rel[true_label]+'\n')
    f.close()

def TACRED_scorer(args, key, prediction, verbose=True):
    if args.task == 'TACRED':
        NO_RELATION = "NA"
    elif args.task == 'KBP37':
        NO_RELATION = "no_relation"
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    f1_list = []
    p_list = []
    r_list = []

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = args.idx2rel[key[row]]
        guess = args.idx2rel[prediction[row]]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
            f1_list.append(f1)
            p_list.append(prec)
            r_list.append(recall)
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")

    if args.task == 'TACRED':
        prec_micro = 1.0
        if sum(guessed_by_relation.values()) > 0:
            prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
        recall_micro = 0.0
        if sum(gold_by_relation.values()) > 0:
            recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
        print("Precision (micro): {:.3%}".format(prec_micro))
        print("   Recall (micro): {:.3%}".format(recall_micro))
        print("       F1 (micro): {:.3%}".format(f1_micro))
        return prec_micro, recall_micro, f1_micro

    elif args.task == 'KBP37':
        prec_macro = sum(p_list) / len(p_list)
        recall_macro = sum(r_list) / len(r_list)
        f1_macro = sum(f1_list) / len(f1_list)
        print("Precision (macro): {:.3%}".format(prec_macro))
        print("   Recall (macro): {:.3%}".format(recall_macro))
        print("       F1 (macro): {:.3%}".format(f1_macro))
        return prec_macro, recall_macro, f1_macro


def evaluate_results(net, test_loader, pad_id, cuda, args, epoch):
    logger.info("Evaluating test samples...")
    acc = 0; out_labels = []; true_labels = []
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            x, e1_e2_start, labels, _,_,_ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if args.only_evaluate == 2 and i >= 10:
                break

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                
            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, \
                          e1_e2_start=e1_e2_start)
            
            accuracy, (o, l) = evaluate_(classification_logits, labels, ignore_idx=-1)
            out_labels.extend(o); true_labels.extend(l) 
            acc += accuracy

    accuracy = acc/(i + 1)

    results = {
        "accuracy": accuracy,
        "precision": sklearn_precision_score(true_labels, out_labels, labels=list(range(args.num_classes)), average='macro'),
        "recall": sklearn_recall_score(true_labels, out_labels, labels=list(range(args.num_classes)), average='macro'),
        "sklearn f1-macro": sklearn_f1_score(true_labels, out_labels, labels=list(range(args.num_classes)), average='macro'),
        "sklearn f1-micro": sklearn_f1_score(true_labels, out_labels, labels=list(range(args.num_classes)), average='micro')
    }

    if args.task == 'SemEval':
        logger.info("Generating additional files ...")
        semeval_files(args, true_labels, out_labels, epoch)
    elif args.task == 'TACRED' or args.task == 'KBP37':
        TACRED_scorer(args, true_labels, out_labels)


    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    
    return results
    