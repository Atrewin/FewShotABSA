#!/usr/bin/env python
from typing import List, Tuple, Dict
import argparse, copy
import logging
import sys
import torch
import random
import os
import json
import pickle
# my staff
from utils.data_loader import FewShotRawDataLoader
from utils.preprocessor import BertInputBuilder, make_sent_dict, make_word_dict, \
    make_boundary_dict, make_preprocessor, make_label_mask
from utils.opt import define_args, basic_args, train_args, test_args, preprocess_args, model_args, option_check
from utils.device_helper import prepare_model, set_device_environment
from utils.trainer import FewShotTrainer, prepare_optimizer
from utils.tester import FewShotTester, eval_check_points
from utils.model_helper import make_model, load_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_training_dataloader(opt, data_loader, preprocessor):
    """ prepare feature and data """
    train_data_loader = data_loader.load_data(opt.train_path, preprocessor, opt, opt.train_batch_size)
    dev_data_loader = data_loader.load_data(opt.test_path, preprocessor, opt, opt.test_batch_size)

    logger.info(' Finish train dev dataloader ')

    return train_data_loader, dev_data_loader


def get_testing_dataloader(opt, data_loader, preprocessor):
    """ prepare feature and data """
    # @jiuhui 为什么还是training的path
    test_data_loader = data_loader.load_data(opt.train_path, preprocessor, opt, opt.test_batch_size)
    logger.info(' Finish test dataloader ')

    return test_data_loader


def main():
    """ to start the experiment """
    ''' set option '''
    parser = argparse.ArgumentParser()
    parser = define_args(parser, basic_args, train_args, test_args, preprocess_args, model_args)
    opt = parser.parse_args()
    print('Args:\n', json.dumps(vars(opt), indent=2))
    opt = option_check(opt)

    ''' device & environment '''
    device, n_gpu = set_device_environment(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    logger.info("Environment: device {}, n_gpu {}".format(device, n_gpu))

    ''' data & feature '''
    data_loader = FewShotRawDataLoader(opt)
    preprocessor = make_preprocessor(opt)
    sent_label2id, sent_id2label = make_sent_dict()
    opt.sent_label2id = sent_label2id
    opt.sent_id2label = sent_id2label
    word_label2id, word_id2label = make_word_dict()
    opt.word_label2id = word_label2id
    opt.word_id2label = word_id2label
    boundary_label2id, boundary_id2label = make_boundary_dict()
    opt.boundary_label2id = boundary_label2id
    opt.boundary_id2label = boundary_id2label
    if opt.do_train:
        train_data_loader, dev_data_loader = \
            get_training_dataloader(opt, data_loader, preprocessor)
    else:
        train_data_loader, dev_data_loader = [None] * 2

    if opt.do_predict:
        test_data_loader = get_testing_dataloader(opt, data_loader, preprocessor)
    else:
        test_data_loader = [None] * 1


    # ''' over fitting test '''
    # if opt.do_overfit_test:
    #     test_features, test_label2id, test_id2label = train_features, train_label2id, train_id2label
    #     dev_features, dev_label2id, dev_id2label = train_features, train_label2id, train_id2label

    ''' select training & testing mode '''
    trainer_class = FewShotTrainer
    tester_class = FewShotTester

    ''' training '''
    best_model = None
    if opt.do_train:
        logger.info("***** Perform training *****")
        if opt.restore_cpt:  # restart training from a check point.
            training_model = load_model(opt.saved_model_path)  # restore optimizer param is not support now.
            opt = training_model.opt
            opt.warmup_epoch = -1
        else:
            training_model = make_model(opt)
        training_model = prepare_model(opt, training_model, device, n_gpu)##有并行逻辑

        # prepare a set of name subseuqence/mark to use different learning rate for part of params
        upper_structures = ['span_detector', 'scale_rate', 'span_cls', 'crf', 'biaffine', 'char_embed']  # a set of parameter name subseuqence/mark
        param_to_optimize, optimizer, scheduler = prepare_optimizer(opt, training_model, upper_structures)

        tester = tester_class(opt, device, n_gpu)
        trainer = trainer_class(opt, optimizer, scheduler, param_to_optimize, device, n_gpu, tester=tester)
        if opt.warmup_epoch > 0:
            training_model.no_embedder_grad = True
            stage_1_param_to_optimize, stage_1_optimizer, stage_1_scheduler = prepare_optimizer(
                opt, training_model, upper_structures)
            stage_1_trainer = trainer_class(opt, stage_1_optimizer, stage_1_scheduler, stage_1_param_to_optimize, device, n_gpu, tester=None)
            trained_model, best_dev_score, test_score = stage_1_trainer.do_train(
                training_model, train_data_loader, opt.warmup_epoch)
            training_model = trained_model
            training_model.no_embedder_grad = False
            print('========== Warmup training finished! ==========')

        trained_model, best_dev_score, test_score = trainer.do_train(
            training_model, train_data_loader, opt.num_train_epochs, dev_data_loader, best_dev_score_now=0)

        exit()
        # decide the best model
        if not opt.eval_when_train:  # select best among check points
            best_model, best_score, test_score_then = trainer.select_model_from_check_point(
                train_id2label, dev_features, dev_id2label, test_features, test_id2label, rm_cpt=opt.delete_checkpoint)
        else:  # best model is selected during training
            best_model = trained_model
        logger.info('dev:{}, test:{}'.format(best_dev_score, test_score))
        print('dev:{}, test:{}'.format(best_dev_score, test_score))

    ''' testing '''
    if opt.do_predict:
        logger.info("***** Perform testing *****")
        print("***** Perform testing *****")
        tester = tester_class(opt, device, n_gpu)
        if not best_model:  # no trained model load it from disk.
            if not opt.saved_model_path or not os.path.exists(opt.saved_model_path):
                raise ValueError("No model trained and no trained model file given (or not exist)")
            if os.path.isdir(opt.saved_model_path):  # eval a list of checkpoints
                max_score = eval_check_points(opt, tester, test_features, test_id2label, device)
                print('best check points scores:{}'.format(max_score))
                exit(0)
            else:
                best_model = load_model(opt.saved_model_path)

        ''' test the best model '''
        testing_model = tester.clone_model(best_model, test_id2label)  # copy reusable params
        if opt.mask_transition and opt.task == 'sl':
            testing_model.label_mask = opt.test_label_mask.to(device)
        test_score = tester.do_test(testing_model, test_data_loader, test_id2label, log_mark='test_pred')
        logger.info('test:{}'.format(test_score))
        print('test:{}'.format(test_score))


if __name__ == "__main__":
    main()
