#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:53:55 2019

@author: weetee
"""



import os
import re
import json
import torch
import random
import copy
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_state, load_results, evaluate_, evaluate_results
from seqeval.metrics import precision_score, recall_score, f1_score
from ..misc import save_as_pickle, load_pickle
import matplotlib.pyplot as plt
import time
import logging
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler #新加
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence


logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


def train_and_fit(args):
    
    if args.fp16:    
        from apex import amp
    else:
        amp = None
    
    cuda = torch.cuda.is_available()

    logger.info("Loading Fewrel dataloaders...")
    train_loader, test_loader, train_len, test_len = load_dataloaders(args)

    logger.info("Loaded %d Training samples." % train_len)
    
    if args.model_no == 0:
        from ..model.BERT.modeling_bert import BertModel as Model
        model = args.model_size  # 'bert-base-uncased'
        lower_case = True
        model_name = 'bert'
        net = Model.from_pretrained(model, force_download=False, model_size=args.model_size,\
                                    task='fewrel', n_classes_=args.num_classes)
    elif args.model_no == 1:
        from ..model.ALBERT.modeling_albert import AlbertModel as Model
        lower_case = True
        model_name = 'albert'
        net = Model.from_pretrained(pretrained_model_name_or_path='./pretrained/albert/', \
                                    n_classes_=args.num_classes)
    elif args.model_no == 2: # bert-large
        from ..transformers.models.bert import BertModel as Model
        model_name = 'bert_large'
        net = Model.from_pretrained(pretrained_model_name_or_path='./pretrained/bert_large/',\
                                    num_classes=args.num_classes)
    elif args.model_no == 3:
        from ..gaojie_transformers.models.roberta import RobertaModel as Model
        model_name = 'roberta'
        net = Model.from_pretrained(pretrained_model_name_or_path='./pretrained/roberta/',\
                                    task='fewrel',num_classes=args.num_classes)
    elif args.model_no == 4:
        from ..transformers.models.roberta import RobertaModel as Model
        model_name = 'roberta_large'
        net = Model.from_pretrained(pretrained_model_name_or_path='./pretrained/roberta/',\
                                    task='fewrel',num_classes=args.num_classes)
    
    tokenizer = load_pickle("./pretrained/{}/{}_tokenizer.pkl".format(model_name, model_name))
    net.resize_token_embeddings(len(tokenizer))
    # funciont convert_tokens_to_ids is in tokenization_utils.py line 662
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1
    
    if cuda:
        net.cuda()

    if args.use_pretrained_blanks == 1:
        logger.info("Loading model pre-trained on blanks at ./data/test_checkpoint_%d.pth.tar..." % args.model_no)
        checkpoint_path = "./data/test_checkpoint_%d.pth.tar" % args.model_no
        checkpoint = torch.load(checkpoint_path)
         #读取参数
        model_dict = net.state_dict()
        #将checkpoint['state_dict']里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
         #更新现有的model_dict
        model_dict.update(pretrained_dict)

        net.load_state_dict(pretrained_dict, strict=False)
        del checkpoint, pretrained_dict, model_dict
    
    criterion = nn.CrossEntropyLoss()#ignore_index=-1
    #optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])
    optimizer = torch.optim.SGD([{"params":net.parameters(), "lr": args.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                      24,26,30], gamma=0.8)
    
    start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)  
    
    if (args.fp16) and (amp is not None):
        logger.info("Using fp16...")
        net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
        if amp_checkpoint is not None:
            amp.load_state_dict(amp_checkpoint)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                          24,26,30], gamma=0.8)
    
    losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch = load_results(args.model_no)

    model_name_list = ['bert-base-uncased', 'albert-base-v2', 'bert-large-uncased', 'roberta-base', 'roberta-large']
    logger.info('=============important information=============')
    logger.info('task : {}'.format(args.task))
    logger.info('model : {}'.format(model_name_list[args.model_no]))
    logger.info('train set len : {}'.format(len(train_loader)*args.batch_size))
    logger.info('test set len : {}'.format(len(test_loader)*args.batch_size))

    logger.info("Starting training process...")
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    update_size = len(train_loader)//10
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        net.train(); total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; accuracy_per_batch = []
        for i, data in enumerate(train_loader, 0):
            x, e1_e2_start, labels,_,_,_ = data#x[1,26,句长];e1_e2_start[1,26,2],labels (1,26) NK+1=26
            M = x.shape[1]#5*5+1=26
            batch_size = x.shape[0]# =1
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2]) #[26,句长]
            e1_e2_start = e1_e2_start.reshape(e1_e2_start.shape[0]*e1_e2_start.shape[1], e1_e2_start.shape[2]) #[26,2]

            attention_mask = (x != pad_id).float()#(batch_size*26,句子长度)
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()#(batch_size*26,句子长度)

            similar_scores = []

            if cuda:
                x = x.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                
            logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, \
                          e1_e2_start=e1_e2_start)#(batch_size*26,1536)=[26,1536]
            
            #return classification_logits, labels, net, tokenizer # for debugging now

            outputs = logits.reshape(batch_size, M, logits.shape[1])#(batch_size,26,1536)
            anchor_output = outputs[:, :(M - 1), :]  # (batch_size,25,1536) 前5个句子meta_train_input
            target_output = outputs[:, (M - 1), :]  # (batch_size,1536) 最后一个句子meta_test_input
            target_output = target_output.reshape(target_output.shape[0], 1, target_output.shape[1])#(batch_size, 1, 1536)
            target_matrix_product = torch.matmul(target_output, anchor_output.permute(0, 2, 1))#(batch_size,1,25) 相似度,test句的预测
            predic_labels = target_matrix_product[0]#(1,25)
            N=labels[0][M-1].item()+1
            K=5
            true_labels = [N-1,N-1,N-1,N-1,N-1] #(1) 先转为list再转为tensor
            for j, x in enumerate(target_matrix_product):
                #similar_scores.append(target_matrix_product[j].argmax().cpu().item())
                if (j!=0):
                    #predic_labels= torch.cat( (predic_labels, target_matrix_product[j]/torch.sum(target_matrix_product[j])), axis=0)
                    predic_labels = torch.cat((predic_labels, target_matrix_product[j] ), axis=0)#按行拼接
                    true_labels = true_labels+[N-1,N-1,N-1,N-1,N-1]
            true_labels=torch.LongTensor(true_labels)#(1,5)
            if cuda:
                true_labels=true_labels.cuda()
            predic_labels=predic_labels.reshape(N,K).T #(5,5)
            loss = criterion(predic_labels, true_labels) #predic_labels(batch_size,25) true_labels(batch_size)
            #loss = loss/len(predic_labels)
            loss = loss/args.gradient_acc_steps
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if args.fp16:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            else:
                grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)
            
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            total_acc += evaluate_(predic_labels , true_labels, ignore_idx=-1)
            
            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                accuracy_per_batch.append(total_acc/update_size)
                print('%d, %5d' % (i*1, 1*args.batch_size))
                print('[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f' %
                      (epoch + 1, (i + 1), train_len, losses_per_batch[-1], accuracy_per_batch[-1])) #(i+1)*args.batch_size
                total_loss = 0.0; total_acc = 0.0

        scheduler.step()
        results = evaluate_results(net, train_loader, pad_id, cuda)
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
        #test_f1_per_epoch.append(results['f1'])
        print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
        print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
        print("Train accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
        #print("Test f1 at Epoch %d: %.7f" % (epoch + 1, test_f1_per_epoch[-1]))

        if accuracy_per_epoch[-1] > best_pred:
            best_pred = accuracy_per_epoch[-1]
            torch.save({
                'epoch': epoch + 1, \
                'state_dict': net.state_dict(), \
                'best_acc': accuracy_per_epoch[-1], \
                'optimizer': optimizer.state_dict(), \
                'scheduler': scheduler.state_dict(), \
                'amp': amp.state_dict() if amp is not None else amp
            }, os.path.join("./my_model/5shot/" , "task_test_model_best_%d.pth.tar" % args.model_no))

        if (epoch % 1) == 0:
            save_as_pickle("./my_model/5shot/task_test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            save_as_pickle("./my_model/5shot/task_train_accuracy_per_epoch_%d.pkl" % args.model_no, accuracy_per_epoch)
            save_as_pickle("./my_model/5shot/task_test_f1_per_epoch_%d.pkl" % args.model_no, test_f1_per_epoch)
            torch.save({
                'epoch': epoch + 1, \
                'state_dict': net.state_dict(), \
                'best_acc': accuracy_per_epoch[-1], \
                'optimizer': optimizer.state_dict(), \
                'scheduler': scheduler.state_dict(), \
                'amp': amp.state_dict() if amp is not None else amp
            }, os.path.join("./my_model/5shot/", "task_test_checkpoint_%d.pth.tar" % args.model_no))

       # torch.cuda.empty_cache()

    logger.info("Finished Training!")

    path = r'/home/fuys/mtb/fr2/result/result_5shot.txt'
    doc = open(path, 'a')

    logger.info("Evaluating...")
    doc.write('\n'+'TEST' + '\n')
    results = evaluate_results(net, test_loader, pad_id, cuda)

    return net
