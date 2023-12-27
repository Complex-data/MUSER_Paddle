import logging
from datetime import datetime
import paddle.nn as nn
import paddle
from tqdm import tqdm
import numpy as np
from paddlenlp.transformers import *
import math
import argparse
import random
import copy
import os
from nltk.tokenize import word_tokenize
from paddle import fluid

from utils.nli_data_reader import NLIDataReader
from utils.logging_handler import LoggingHandler
from bert_nli import BertNLIModel
from test_trained_model import evaluate


def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler
    """
    scheduler = scheduler.lower()
    if scheduler == 'constantlr':
        return paddle.optimizer.lr_scheduler.ConstantLearningRate(optimizer)
    elif scheduler == 'warmupconstant':
        return paddle.optimizer.lr_scheduler.LinearWarmup(optimizer, warmup_steps, t_total)
    elif scheduler == 'warmuplinear':
        return paddle.optimizer.lr_scheduler.LinearWarmup(optimizer, warmup_steps, t_total)
    elif scheduler == 'warmupcosine':
        return paddle.optimizer.lr_scheduler.CosineAnnealingWithWarmup(optimizer, warmup_steps, t_total)
    elif scheduler == 'warmupcosinewithhardrestarts':
        return paddle.optimizer.lr_scheduler.CosineAnnealingWithWarmup(optimizer, warmup_steps, t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))


def train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, checkpoint, gpu, max_grad_norm,
          best_acc):
    loss_fn = nn.CrossEntropyLoss()

    step_cnt = 0
    best_model_weights = None

    for pointer in tqdm(range(0, len(train_data), batch_size), desc='training'):
        model.train()  # model was in eval mode in evaluate(); re-activate the train mode

        # paddle.fluid.dygraph.release_memory()  # releases all unoccupied cached memory

        step_cnt += 1
        sent_pairs = []
        labels = []
        for i in range(pointer, pointer + batch_size):
            if i >= len(train_data): break
            sents = train_data[i].get_texts()
            if len(word_tokenize(' '.join(sents))) > 300: continue
            sent_pairs.append(sents)
            labels.append(train_data[i].get_label())
        logits, _ = model.ff(sent_pairs, checkpoint)
        if logits is None: continue
        true_labels = paddle.to_tensor(labels)
        if gpu:
            true_labels = true_labels.cuda()
        loss = loss_fn(logits, true_labels)
        logger.info(f'Loss: {float(loss.cpu()):.5f}')
        # back propagate
        if fp16:
            with paddle.amp.auto_cast():
                scaled_loss = optimizer.scale_loss(loss)
                scaled_loss.backward()
            paddle.nn.utils.clip_grad_norm_(paddle.amp.master_params(optimizer), max_grad_norm)
        else:
            loss.backward()
            paddle.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # update weights
        optimizer.step()

        # update training rate
        scheduler.step()

        optimizer.clear_grad()  # clear gradients first

        if step_cnt % 10 == 0:
            acc = evaluate(model, dev_data, checkpoint, mute=False)
            logger.info('==> step {} dev acc: {}'.format(step_cnt, acc))
            if acc > best_acc:
                best_acc = acc
                best_model_weights = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                model.save(model_save_path, best_model_weights, best_acc)
                model.to('gpu')

    return best_model_weights


def parse_args():
    parser = argparse.ArgumentParser("arguments for bert-nli training")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-ep', '--epoch_num', type=int, default=10, help='epoch num')
    parser.add_argument('--fp16', type=int, default=0,
                        help='use apex mixed precision training (1) or not (0); do not use this together with checkpoint')
    parser.add_argument('--check_point', '-cp', type=int, default=0,
                        help='use checkpoint (1) or not (0); this is required for training bert-large or larger models; do not use this together with apex fp16')
    parser.add_argument('--gpu', type=int, default=1, help='use gpu (1) or not (0)')
    parser.add_argument('-ss', '--scheduler_setting', type=str, default='WarmupLinear',
                        choices=['WarmupLinear', 'ConstantLR', 'WarmupConstant', 'WarmupCosine',
                                 'WarmupCosineWithHardRestarts'])
    parser.add_argument('-tm', '--trained_model', type=str, default='None',
                        help='path to the trained model; make sure the trained model is consistent with the model you want to train')
    parser.add_argument('-mg', '--max_grad_norm', type=float, default=1., help='maximum gradient norm')
    parser.add_argument('-wp', '--warmup_percent', type=float, default=0.2,
                        help='how many percentage of steps are used for warmup')
    parser.add_argument('-bt', '--bert_type', type=str, default='bert-large',
                        help='transformer (bert) pre-trained model you want to use',
                        choices=['bert-base', 'bert-large', 'albert-base-v2', 'albert-large-v2'])
    parser.add_argument('--hans', type=int, default=0, help='use hans data (1) or not (0)')
    parser.add_argument('-rl', '--reinit_layers', type=int, default=0, help='reinitialise the last N layers')
    parser.add_argument('-fl', '--freeze_layers', type=int, default=0,
                        help='whether to freeze all but the last few layers (1) or not (0)')
    parser.add_argument('-pl', '--platform', type=str, default='liar', help='liar politifact gossipcop weibo')

    args = parser.parse_args()
    return args.batch_size, args.epoch_num, args.fp16, args.check_point, args.gpu, args.scheduler_setting, args.max_grad_norm, args.warmup_percent, args.bert_type, args.trained_model, args.hans, args.reinit_layers, args.freeze_layers, args.platform


if __name__ == '__main__':
    # paddle.enable_static()

    batch_size, epoch_num, fp16, checkpoint, gpu, scheduler_setting, max_grad_norm, warmup_percent, bert_type, trained_model, hans, reinit_layers, freeze_layers, platform = parse_args()
    fp16 = bool(fp16)
    gpu = bool(gpu)
    hans = bool(hans)
    checkpoint = bool(checkpoint)
    if trained_model == 'None': trained_model = None

    print('=====Arguments=====')
    print('bert type:\t{}'.format(bert_type))
    print('trained model path:\t{}'.format(trained_model))
    print('batch size:\t{}'.format(batch_size))
    print('epoch num:\t{}'.format(epoch_num))
    print('fp16:\t{}'.format(fp16))
    print('check_point:\t{}'.format(checkpoint))
    print('scheduler setting:\t{}'.format(scheduler_setting))
    print('max grad norm:\t{}'.format(max_grad_norm))
    print('warmup percent:\t{}'.format(warmup_percent))
    print('using hans:\t{}'.format(hans))
    print('=====Arguments=====')

    label_num = 2
    # platform = "gossipcop"
    # platform = "politifact"
    # platform = "weibo"
    # platform = "liar"

    model_save_path = 'output/nli_model/' + platform

    print('model save path', model_save_path)

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    # Read the dataset
    nli_reader = NLIDataReader('datasets/' + platform)
    all_data = nli_reader.get_news_examples(platform + '_train.jsonl')  # ,max_examples=5000)

    random.shuffle(all_data)
    train_num = int(len(all_data) * 0.9)
    train_data = all_data[:train_num]
    dev_data = all_data[train_num:]

    logger.info('train data size {}'.format(len(train_data)))
    logger.info('dev data size {}'.format(len(dev_data)))

    total_steps = math.ceil(epoch_num * len(train_data) * 1. / batch_size)
    warmup_steps = int(total_steps * warmup_percent)

    model = BertNLIModel(gpu=gpu, batch_size=batch_size, bert_type=bert_type, model_path=trained_model,
                         reinit_num=reinit_layers, freeze_layers=freeze_layers)

                                     #grad_clip=fluid.clip.GradientClipByGlobalNorm(max_grad_norm))
    # scheduler = fluid.dygraph.learning_rate_scheduler.get_scheduler(scheduler_setting, warmup_steps=warmup_steps,
    #                                                                  total_steps=total_steps)
    scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=0.005, warmup_steps=20, start_lr=0, end_lr=0.05, verbose=True)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

    if fp16:
        model, optimizer = amp.decorate_model(model, optimizer, opt_level='O1')

    best_acc = -1.
    best_model_dic = None
    for ep in range(epoch_num):
        logger.info('\n=====epoch {}/{}====='.format(ep, epoch_num))
        model_dic = train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, checkpoint, gpu,
                          max_grad_norm, best_acc)
        if model_dic is not None:
            best_model_dic = model_dic
    assert best_model_dic is not None

    # for testing load the best model
    # model.load_model(best_model_dic)
    logger.info('\n=====Training finished. Now start test=====')
