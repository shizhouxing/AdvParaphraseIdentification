import os
import numpy as np
import sys, random, time, shutil, copy, nltk, json
from multiprocessing import Pool
from parser import Parser, update_arguments
from Logger import Logger
from utils import load_data, build_vocab, symbol2index, get_batches, save_json, \
                load_paws_test, sample_data
from Generator import Generator

argv = sys.argv[1:]
parser = Parser().getParser()
args, _ = parser.parse_known_args(argv)
random.seed(args.seed)
np.random.seed(args.seed)

args = update_arguments(args)

if args.adv_train or args.adv_generate:
    args.no_common_data = False
if not args.no_common_data:
    data_train, data_valid, data_test, vocab_char, vocab_word = load_data(args)

if args.sample_data:
    sample_data(args, data_test)
    exit(0)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with sess.as_default():
    if args.target == 'bimpm':
        from models.BiMPM.Model import BiMPM
        target = BiMPM(sess, args, vocab_char, vocab_word)
    elif args.target == 'diin':
        from models.DIIN.Model import DIIN
        target = DIIN(sess, args)
    elif args.target == 'drcn':
        from models.DRCN.Model import DRCN
        target = DRCN(sess, args, vocab_char, vocab_word)
    elif args.target == 'bert':
        from models.BERT.Model import BERT
        target = BERT(args, len(data_train))
    else:
        raise Exception('Unknown target model')

    if args.no_common_data:
        data_train, data_valid, data_test = target.data_train, target.data_valid, target.data_test
    random.shuffle(data_valid)
    random.shuffle(data_test)
    valid_batches = get_batches(args, data_valid, args.batch_size)
    test_batches = get_batches(args, data_test, args.batch_size)
    print('Dataset sizes: %d/%d/%d' % (len(data_train), len(data_valid), len(data_test)))

    summary_names = ['loss', 'accuracy', 'suc_pos', 'suc_neg']
    summary_num_pre = 2 

    logger = Logger(sess, args, summary_names, 1)

with sess.as_default():
    if args.pre_train:          
        while True:
            random.shuffle(data_train)
            train_batches = get_batches(args, data_train, args.batch_size)

            for i, batch in enumerate(train_batches):
                logger.next_step(target.step(batch, is_train=True)[:summary_num_pre])
                if (i + 1) % args.fine_tune_steps == 0 or (i + 1) == len(train_batches):
                    logger.next_epoch()
                    if args.use_torch:
                        target.save(logger.epoch.eval())
                    for batch in valid_batches:
                        logger.add_valid(target.step(batch)[:summary_num_pre])
                    logger.save_valid(log=True)    
                    for batch in test_batches:
                        logger.add_test(target.step(batch)[:summary_num_pre])
                    logger.save_test(log=True)           
           
if args.adv_train or args.adv_generate:
    vocab_word.build_candwords(data_train)
if args.adv_train or args.adv_generate:
    generator = Generator(args, target, vocab_char, vocab_word, data_train)    

with sess.as_default():
    if args.adv_train:
        print('Start adversarial training...')
        pool_adv_pos, pool_adv_neg = [], []
        while True:
            random.shuffle(data_train)
            train_batches = get_batches(args, data_train, args.batch_size)

            suc_pos_all, suc_neg_all = [], []
            for i, batch in enumerate(train_batches):   
                if i >= args.adv_steps_per_epoch: break

                while len(pool_adv_pos) < args.adv_examples_per_step:
                    b, suc_pos = generator.generate_batch(label=1, verbose=True)
                    suc_pos_all.append(suc_pos)
                    pool_adv_pos += b
                while len(pool_adv_neg) < args.adv_examples_per_step:
                    b, suc_neg = generator.generate_batch(label=0, verbose=True)
                    suc_neg_all.append(suc_neg)
                    pool_adv_neg += b

                batch += pool_adv_pos[:args.adv_examples_per_step] + \
                    pool_adv_neg[:args.adv_examples_per_step]
                pool_adv_pos = pool_adv_pos[args.adv_examples_per_step:]
                pool_adv_neg = pool_adv_neg[args.adv_examples_per_step:]

                logger.next_step(target.step(batch, is_train=True, attack=True)[:summary_num_pre])

                if i % args.display_interval == 0:
                    print('Suc %.5f %.5f' % (np.mean(suc_pos_all), np.mean(suc_neg_all)))

            logger.next_epoch()
            if args.use_torch:
                target.save(logger.epoch.eval())                
            for batch in test_batches:
                logger.add_test(target.step(batch, attack=True)[:summary_num_pre] + \
                    [np.mean(suc_pos_all), np.mean(suc_neg_all)])
            logger.save_test(log=True)

    if args.adv_generate:
        print('Evaluating...')
        if args.use_dev:
            generator.evaluate(data_valid, args.num_adv, verbose=True)
        else:
            generator.evaluate(data_test, args.num_adv, verbose=True)
        exit(0)
