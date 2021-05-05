#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:53:55 2019

@author: weetee
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_state, load_results, evaluate_, evaluate_results
from ..misc import save_as_pickle, load_pickle
import matplotlib.pyplot as plt
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def train_and_fit(args):
    
    if args.fp16:    
        from apex import amp
    else:
        amp = None
    
    cuda = torch.cuda.is_available()
    
    train_loader, test_loader, train_len, test_len = load_dataloaders(args)
    #train_loader, test_loader, dev_loader, train_len, test_len, dev_len = load_dataloaders(args)
    logger.info("Loaded %d Training samples." % train_len)
    
    if args.model_no == 0:
        from ..model.BERT.modeling_bert import BertModel as Model
        model_name = 'bert'
        net = Model.from_pretrained(pretrained_model_name_or_path='./pretrained/bert/', \
                                    task='classification' ,\
                                    n_classes_=args.num_classes)
    elif args.model_no == 1:
        from ..model.ALBERT.modeling_albert import AlbertModel as Model
        lower_case = True
        model_name = 'albert'
        net = Model.from_pretrained(pretrained_model_name_or_path='./pretrained/albert/', \
                                    n_classes_=args.num_classes)
    elif args.model_no == 2:
        from ..gaojie_transformers.models.roberta import RobertaModel as Model
        model_name = 'roberta'
        net = Model.from_pretrained(pretrained_model_name_or_path='./pretrained/roberta/', \
                                    num_classes=args.num_classes)
    elif args.model_no == 3:
        from ..gaojie_transformers.models.bert import BertModel as Model
        model_name = 'bert_large'
        net = Model.from_pretrained(pretrained_model_name_or_path='./pretrained/bert_large/', \
                                    num_classes=args.num_classes)
    
    tokenizer = load_pickle("./pretrained/{}/{}_tokenizer.pkl".format(model_name, model_name))
    net.resize_token_embeddings(len(tokenizer))
    # funciont convert_tokens_to_ids is in tokenization_utils.py line 662
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1
    
    if cuda:
        net.cuda()
    ''' 
    logger.info("FREEZING MOST HIDDEN LAYERS...")
    if args.model_no == 0:
        unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                           "classification_layer", "blanks_linear", "lm_linear", "cls"]
    elif args.model_no == 1:
        unfrozen_layers = ["classifier", "pooler", "classification_layer",\
                           "blanks_linear", "lm_linear", "cls",\
                           "albert_layer_groups.0.albert_layers.0.ffn"]
    elif args.model_no == 2:
        unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                           "classification_layer", "blanks_linear", "lm_linear", "cls"]
        
    for name, param in net.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            # print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            # print("[FREE]: %s" % name)
            param.requires_grad = True
    '''
    for name, param in net.named_parameters():
       param.requires_grad = True

    if args.use_pretrained_blanks == 1:
        logger.info("Loading model pre-trained on blanks at ./data/test_checkpoint_%d.pth.tar..." % args.model_no)
        checkpoint_path = "./data/test_checkpoint_%d.pth.tar" % args.model_no
        checkpoint = torch.load(checkpoint_path)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        net.load_state_dict(pretrained_dict, strict=False)
        del checkpoint, pretrained_dict, model_dict
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])
    
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

    model_name_list = ['bert-base-uncased', 'albert-base-v2', 'roberta-base', 'bert-large-uncased']
    logger.info('=============important information=============')
    logger.info('task : {}'.format(args.task))
    logger.info('model : {}'.format(model_name_list[args.model_no]))
    logger.info('train set len : {}'.format(len(train_loader)*args.batch_size))
    logger.info('test set len : {}'.format(len(test_loader)*args.batch_size))
    logger.info('SemEval : train-8000, test-2717')
    logger.info('KBP37 : train-15917, test-3405, dev-1724')
    logger.info('TACRED : train-68124, test-15509, dev-22631')
    logger.info('===============================================')

    pad_id = tokenizer.pad_token_id
    if args.only_evaluate != 0:
        evaluate_results(net, test_loader, pad_id, cuda, args, 0)
        return 0

    logger.info("Starting training process...")
    mask_id = tokenizer.mask_token_id
    update_size = len(train_loader)//10
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        net.train(); total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; accuracy_per_batch = []
        for i, data in enumerate(train_loader, 0):
            x, e1_e2_start, labels, _,_,_ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                e1_e2_start = e1_e2_start.cuda()
            classification_logits = net(x, token_type_ids=token_type_ids, \
                                attention_mask=attention_mask, e1_e2_start=e1_e2_start)#（batch_size,80）
            
            loss = criterion(classification_logits, labels.squeeze(1))#labels （）
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
            total_acc += evaluate_(classification_logits, labels, \
                                   ignore_idx=-1)[0]
            
            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                accuracy_per_batch.append(total_acc/update_size)
                print('[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f' %
                      (epoch + 1, (i + 1)*args.batch_size, train_len, losses_per_batch[-1], accuracy_per_batch[-1]))
                total_loss = 0.0; total_acc = 0.0
        
        scheduler.step()
        results = evaluate_results(net, test_loader, pad_id, cuda, args, epoch)
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
        # test_f1_per_epoch.append(results['f1'])
        print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
        print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
        print("Train accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
        # print("Test f1 at Epoch %d: %.7f" % (epoch + 1, test_f1_per_epoch[-1]))
        
        if accuracy_per_epoch[-1] > best_pred:
            best_pred = accuracy_per_epoch[-1]
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./my_model/" , "task_test_model_best_%d.pth.tar" % args.model_no))
        
        if (epoch % 1) == 0:
            save_as_pickle("./my_model/task_test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            save_as_pickle("./my_model/task_train_accuracy_per_epoch_%d.pkl" % args.model_no, accuracy_per_epoch)
            save_as_pickle("./my_model/task_test_f1_per_epoch_%d.pkl" % args.model_no, test_f1_per_epoch)
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./my_model/" , "task_test_checkpoint_%d.pth.tar" % args.model_no))
    
    logger.info("Finished Training!")

    #results = evaluate_results(net, dev_loader, pad_id, cuda, args, epoch=None)
    path = r'/home/fuys/mtb/MTB-baseline/result.txt'
    doc = open(path, 'a')
    #doc.write('\n')
    #doc.write('DEV' + '\n')
    #for k, v in results.items():
    #    s2 = str(v)
    #    doc.write(k + '\n')
    #    doc.write(s2 + '\n')

    logger.info("Evaluating...")
    results = evaluate_results(net, test_loader, pad_id, cuda, args, epoch=None)
    doc.write('TEST' + '\n')
    for k, v in results.items():
        s2 = str(v)
        doc.write(k + '\n')
        doc.write(s2 + '\n')
    return net