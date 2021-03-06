# coding: utf-8
from typing import List, Tuple, Dict
import torch
import logging
import sys
import time
import os
import copy
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# My Staff
from utils.iter_helper import PadCollate, FewShotDataset, SimilarLengthSampler
from utils.preprocessor import ModelInput
from utils.device_helper import prepare_model
from utils.model_helper import make_model, load_model
from models.few_shot_seq_labeler import FewShotSeqLabeler
import math


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    # stream=sys.stderr
    # stream=sys.stdout
)
logger = logging.getLogger(__name__)


class TrainerBase:
    """
    Build a pytorch trainer, it is design to be:
        - reusable for different training data
        - reusable for different training model instance
        - contains 2 model selection strategy:
            - dev and test(optional) during training. (not suitable when the model is very large)
            - store all checkpoint to disk.
    Support features:
        - multi-gpu [accelerating]
        - distributed gpu [accelerating]
        - 16bit-float training [save space]
        - split batch [save space]
        - model selection(dev & test) [better result & unexpected exit]
        - check-point [unexpected exit]
        - early stop [save time]
        - padding when forward [better result & save space]
        - grad clipping [better result]
        - step learning rate decay [better result]
    """
    def __init__(self, opt, optimizer, scheduler, param_to_optimize, device, n_gpu, tester=None):
        """
        :param opt: args
        :param optimizer:
        :param scheduler:
        :param param_to_optimize: model's params to optimize
        :param device: torch class for training device,
        :param n_gpu:  number of gpu used
        :param tester: class for evaluation
        """
        if opt.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                opt.gradient_accumulation_steps))

        self.opt = opt
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.param_to_optimize = param_to_optimize
        self.tester = tester  # for model selection, set 'None' to not select
        self.gradient_accumulation_steps = opt.gradient_accumulation_steps
        # Following is used to split the batch to save space
        self.batch_size = int(opt.train_batch_size / opt.gradient_accumulation_steps)
        self.device = device
        self.n_gpu = n_gpu

    def do_train(self, model, train_data_loader, num_train_epochs, dev_data_loader=None, best_dev_score_now=0):
        """
        do training and dev model selection
        :param model:
        :param train_features:
        :param dev_features:
        :param dev_id2label:
        :param test_features:
        :param test_id2label:
        :param best_dev_score_now:
        :return:
        """
        num_train_steps = int(self.opt.iter_num / self.opt.gradient_accumulation_steps) * self.opt.num_train_epochs

        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", self.batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        global_step = 0  # used for args.fp16
        total_step = 0
        best_dev_score_now = best_dev_score_now
        best_model_now = model
        test_score = None
        min_loss = 100000000000000
        loss_now = 0
        no_new_best_dev_num = 0
        no_loss_decay_steps = 0
        is_convergence = False

        model.train()

        for epoch_id in trange(int(num_train_epochs), desc="Epoch"):# batch_size=2
            # ??????????????????????????????????????????epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc="Train-Batch Progress", ncols=200)):
                # print("\r")
                # if self.n_gpu == -1:#if self.n_gpu == 1:#?????????????????????GPU?????????????????????????????????
                #     batch = tuple(t.to(self.device) for t in batch)  # multi-gpu does scattering it-self
                ''' loss '''
                loss = self.do_forward(batch, model, epoch_id, step)# loss ??????*(-1)??????????????????
                loss = self.process_special_loss(loss)  # for parallel process, split batch and so on
                loss.backward()
                # exit()
                # 04.27
                ''' optimizer step '''
                global_step, model, is_nan, update_model = self.optimizer_step(step, model, global_step)
                if is_nan:  # FP16 TRAINING: Nan in gradients, reducing loss scaling
                    continue
                total_step += 1

                ''' model selection '''
                if self.time_to_make_check_point(total_step, train_data_loader) or 1:
                    if self.tester and self.opt.eval_when_train:  # this is not suit for training big model
                        print("Start dev eval.")#@jinhui 7.23 ?????????????????????
                        dev_score, test_score, copied_best_model = self.model_selection(# ????????????????????????????????????target domain ??????acc ??????????????????????????????
                            model, best_dev_score_now, dev_data_loader, self.opt.sent_id2label)

                        if dev_score > best_dev_score_now:
                            best_dev_score_now = dev_score
                            best_model_now = copied_best_model
                            no_new_best_dev_num = 0
                        else:
                            no_new_best_dev_num += 1
                    else:
                        self.make_check_point_(model=model, step=total_step)

                ''' convergence detection & early stop '''
                loss_now = loss.item() if update_model else loss.item() + loss_now
                if self.opt.convergence_window > 0 and update_model:
                    if global_step % 100 == 0 or total_step % len(train_data_loader) == 0:
                        print('Current loss {}, global step {}, min loss now {}, no loss decay step {}'.format(
                            loss_now, global_step, min_loss, no_loss_decay_steps), end="")
                    if loss_now < min_loss:
                        min_loss = loss_now
                        no_loss_decay_steps = 0
                    else:
                        no_loss_decay_steps += 1
                        if no_loss_decay_steps >= self.opt.convergence_window:
                            logger.info('=== Reach convergence point!!!!!! ====')
                            print('=== Reach convergence point!!!!!! ====')
                            is_convergence = True
                if no_new_best_dev_num >= self.opt.convergence_dev_num > 0:
                    logger.info('=== Reach convergence point!!!!!! ====')
                    print('=== Reach convergence point!!!!!! ====')
                    is_convergence = True
                if is_convergence:
                    break
            if is_convergence:
                break
            print(" --- The {} epoch Finish --- ".format(epoch_id))

        return best_model_now, best_dev_score_now, test_score

    def time_to_make_check_point(self, step, data_loader):
        # interval_size = int(len(data_loader) / self.opt.cpt_per_epoch)
        # remained_step = len(data_loader) - (step % len(data_loader))  # remained step for current epoch
        return (step % 100 == 0)

    def get_dataset(self, features):
        return TensorDataset([self.unpack_feature(f) for f in features])

    def get_sampler(self, dataset):
        if self.opt.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        return sampler

    def get_data_loader(self, dataset, sampler):
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
        return data_loader

    def process_special_loss(self, loss):
        if self.n_gpu > 1:
            # loss = loss.sum()  # sum() to average on multi-gpu.
            loss = loss.mean()  # mean() to average on multi-gpu.
        if self.opt.fp16 and self.opt.loss_scale != 1.0:
            # rescale loss for fp16 training
            # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
            loss = loss * self.opt.loss_scale
        if self.opt.gradient_accumulation_steps > 1:
            loss = loss / self.opt.gradient_accumulation_steps
        return loss

    def set_optimizer_params_grad(self, param_to_optimize, named_params_model, test_nan=False):
        """ Utility function for optimize_on_cpu and 16-bits training.
            Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
        """
        is_nan = False
        for (name_opti, param_opti), (name_model, param_model) in zip(param_to_optimize, named_params_model):
            if name_opti != name_model:
                logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
                raise ValueError
            if param_model.grad is not None:
                if test_nan and torch.isnan(param_model.grad).sum() > 0:
                    is_nan = True
                if param_opti.grad is None:
                    param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
                param_opti.grad.data.copy_(param_model.grad.data)
            else:
                param_opti.grad = None
        return is_nan

    def copy_optimizer_params_to_model(self, named_params_model, named_params_optimizer):
        """ Utility function for optimize_on_cpu and 16-bits training.
            Copy the parameters optimized on CPU/RAM back to the model on GPU
        """
        for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
            if name_opti != name_model:
                logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
                raise ValueError
            param_model.data.copy_(param_opti.data)

    def make_check_point(self, model, step):
        logger.info("Save model check point to file:%s", os.path.join(
            self.opt.output_dir, 'model.step{}.cpt.pl'.format(step)))
        torch.save(
            self.check_point_content(model), os.path.join(self.opt.output_dir, 'model.step{}.cpt.pl'.format(step)))

    def make_check_point_(self, model, step):
        """ deal with IO error version """
        try:
            logger.info("Save model check point to file:%s", os.path.join(
                self.opt.output_dir, 'model.step{}.cpt.pl'.format(step)))
            torch.save(
                self.check_point_content(model), os.path.join(self.opt.output_dir, 'model.step{}.cpt.pl'.format(step)))
        except IOError:
            logger.info("Failed to make cpt, sleeping ...")
            time.sleep(300)
            self.make_check_point_(model, step)

    def model_selection(self, model, best_score, dev_data_loader, dev_id2label, test_features=None, test_id2label=None):
        """ do model selection during training"""
        print("Start dev model selection.")
        # do dev eval at every dev_interval point and every end of epoch
        # dev_model = self.tester.clone_model(model, dev_id2label)  # copy reusable params, for a different domain
        dev_model = model
        if self.opt.mask_transition and self.opt.task == 'sl':#@jinhui
            dev_model.label_mask = self.opt.dev_label_mask.to(self.device)

        dev_score = self.tester.do_test(dev_model, dev_data_loader, dev_id2label, log_mark='dev_pred')
        logger.info("  dev score(F1) = {}".format(dev_score))
        print("  dev score(F1) = {}".format(dev_score))
        best_model = None
        test_score = None
        if dev_score > best_score:
            logger.info(" === Found new best!! === ")
            ''' store new best model  '''
            best_model = self.clone_model(model)  # copy model to avoid writen by latter training
            ''' save model file '''
            logger.info("Save model to file:%s", os.path.join(self.opt.output_dir, 'model.pl'))
            torch.save(self.check_point_content(model), os.path.join(self.opt.output_dir, 'model.pl'))

            ''' get current best model's test score '''
            if test_features:
                test_model = self.tester.clone_model(model, test_id2label)  # copy reusable params for different domain
                if self.opt.mask_transition and self.opt.task == 'sl':
                    test_model.label_mask = self.opt.test_label_mask.to(self.device)

                test_score = self.tester.do_test(test_model, test_features, test_id2label, log_mark='test_pred')
                logger.info("  test score(F1) = {}".format(test_score))
                print("  test score(F1) = {}".format(test_score))
        # reset the model status
        model.train()
        return dev_score, test_score, best_model

    def check_point_content(self, model):
        """ necessary staff for rebuild the model """
        model = model
        return model.state_dict()

    def select_model_from_check_point(
            self, train_id2label, dev_features, dev_id2label, test_features=None, test_id2label=None, rm_cpt=True):
        all_cpt_file = list(filter(lambda x: '.cpt.pl' in x, os.listdir(self.opt.output_dir)))
        best_score = 0
        test_score_then = 0
        best_model = None
        all_cpt_file = sorted(all_cpt_file, key=lambda x: int(x.replace('model.step', '').replace('.cpt.pl', '')))
        for cpt_file in all_cpt_file:
            logger.info('testing check point: {}'.format(cpt_file))
            model = load_model(os.path.join(self.opt.output_dir, cpt_file))
            dev_score, test_score, copied_model = self.model_selection(
                model, best_score, dev_features, dev_id2label)
            if dev_score > best_score:
                best_score = dev_score
                test_score_then = test_score
                best_model = copied_model
        if rm_cpt:  # delete all check point
            for cpt_file in all_cpt_file:
                os.unlink(os.path.join(self.opt.output_dir, cpt_file))
        return best_model, best_score, test_score_then

    def unpack_feature(self, feature) -> List[torch.Tensor]:
        raise NotImplementedError

    def clone_model(self, model):
        # get a new instance
        return copy.deepcopy(model)

    def do_forward(self, batch, model, epoch_id, step):
        loss = model(*batch)
        return loss

    def optimizer_step(self, step, model, global_step):
        is_nan = False
        update_model = False
        if (step + 1) % self.gradient_accumulation_steps == 0:  # for both memory saving setting and normal setting
            if self.opt.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), self.opt.clip_grad)
            if self.opt.fp16 or self.opt.optimize_on_cpu:
                if self.opt.fp16 and self.opt.loss_scale != 1.0:
                    # scale down gradients for fp16 training
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data = param.grad.data / self.opt.loss_scale
                is_nan = self.set_optimizer_params_grad(self.param_to_optimize, model.named_parameters(), test_nan=True)
                if is_nan:
                    logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                    self.opt.loss_scale = self.opt.loss_scale / 2
                    model.zero_grad()
                    return global_step, model, is_nan
                self.optimizer.step()
                self.copy_optimizer_params_to_model(model.named_parameters(), self.param_to_optimize)
            else:
                self.optimizer.step()
            if self.scheduler:  # decay learning rate
                self.scheduler.step()
            model.zero_grad()
            global_step += 1
            update_model = True
        return global_step, model, is_nan, update_model


class FewShotTrainer(TrainerBase):
    """
    Support features:
        - multi-gpu [accelerating]
        - distributed gpu [accelerating]
        - 16bit-float training [save space]
        - split batch [save space]
        - model selection(dev & test) [better result & unexpected exit]
        - check-point [unexpected exit]
        - early stop [save time]
        - padding when forward [better result & save space]
    """
    def __init__(self, opt, optimizer, scheduler, param_to_optimize, device, n_gpu, tester=None):
        super(FewShotTrainer, self).__init__(opt, optimizer, scheduler, param_to_optimize, device, n_gpu, tester)

    def get_dataset(self, features):
        return FewShotDataset([self.unpack_feature(f) for f in features])

    def get_sampler(self, dataset):
        if self.opt.local_rank == -1:
            if self.opt.sampler_type == 'similar_len':
                sampler = SimilarLengthSampler(dataset, batch_size=self.batch_size)
            elif self.opt.sampler_type == 'random':
                sampler = RandomSampler(dataset)
            else:
                raise TypeError('the sampler_type is not true')
        else:
            sampler = DistributedSampler(dataset)
        return sampler

    def get_data_loader(self, dataset, sampler):
        pad_collate = PadCollate(dim=-1, sp_dim=-2, sp_item_idx=[3, 8, 12])  # nwp_index, spt_tgt need special padding
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=pad_collate)
        return data_loader


    def do_forward(self, batch, model, epoch_id, step):
        support_set, query_set, support_label, query_label = batch
        if torch.cuda.is_available():
            for k in support_set:
                if k != "sg":
                    support_set[k] = support_set[k].cuda()
                else:
                    for sg_index in range(len(support_set[k])):
                        support_set[k][sg_index] = support_set[k][sg_index].cuda()
            for k in query_set:
                if k != "sg":
                    query_set[k] = query_set[k].cuda()
                else:
                    for sg_index in range(len(query_set[k])):
                        query_set[k][sg_index] = query_set[k][sg_index].cuda()

            support_label = support_label.cuda()
            query_label = query_label.cuda()
        # ???????????????model?????????sg input ???????????????????????????sg????????????????????????????????????if-else??????????????????????????????????????????????????????????????????, ?????????????????????
        # @jinhui 0802 ????????????????????????????????????????????????
        loss = model(
            support_set=support_set,
            query_set=query_set,
            support_label=support_label,
            query_label=query_label,
        )
        return loss

    def check_point_content(self, model):
        """ save staff for rebuild a model """
        model = model  # save sub-module may cause issues
        sub_model = model if self.n_gpu <= 1 else model.module
        ret = {
            'state_dict': model.state_dict(),
            'opt': self.opt,
            # 'config': model.config,
        }
        return ret

    def get_value_from_order_dict(self, order_dict, key):
        """"""
        for k, v in order_dict.items():
            if key in k:
                return v
        return []

    def clone_model(self, model):
        # deal with data parallel model
        best_model: FewShotSeqLabeler
        old_model: FewShotSeqLabeler
        if self.opt.local_rank != -1 or self.n_gpu > 1:  # the model is parallel class here
            old_model = model.module
        else:
            old_model = model
        # get a new instance for different domain (cpu version to save resource)
        config = {'num_tags': old_model.config['num_tags']}
        if 'num_anchors' in old_model.config:
            config['num_anchors'] = old_model.config['num_anchors']  # Use previous model's random anchors.
        best_model = make_model(opt=old_model.opt, config=config)
        # copy weights and stuff
        best_model.load_state_dict(old_model.state_dict())
        return best_model


def prepare_optimizer(opt, model,  upper_structures=None):
    """
    :param opt:
    :param model:
    :param num_train_features:
    :param upper_structures: list of param name that use different learning rate. These names should be unique sub-str.
    :return:
    """
    num_train_steps = int(opt.iter_num / opt.gradient_accumulation_steps) * opt.num_train_epochs

    ''' special process for space saving '''
    if opt.fp16:
        param_to_optimize = [(n, param.clone().detach().to('cpu').float().requires_grad_())
                           for n, param in model.named_parameters()]
    elif opt.optimize_on_cpu:
        param_to_optimize = [(n, param.clone().detach().to('cpu').requires_grad_())
                           for n, param in model.named_parameters()]
    else:
        param_to_optimize = list(model.named_parameters())  # all parameter name and parameter

    ''' construct optimizer '''
    if upper_structures and opt.upper_lr > 0:  # use different learning rate for upper structure parameter
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_to_optimize if not any(nd in n for nd in upper_structures)],
             'weight_decay': 0.01, 'lr': opt.learning_rate},
            {'params': [p for n, p in param_to_optimize if any(nd in n for nd in upper_structures)],
             'weight_decay': 0.1, 'lr': opt.upper_lr},
        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_to_optimize], 'weight_decay': 0.01, 'lr': opt.learning_rate},
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, correct_bias=False)

    ''' construct scheduler '''
    num_warmup_steps = int(opt.warmup_proportion * num_train_steps)
    if opt.scheduler == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)  # PyTorch scheduler
    elif opt.scheduler == 'linear_decay':
        if 0 < opt.decay_lr < 1:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_epoch_size, gamma=opt.decay_lr)
        else:
            raise ValueError('illegal lr decay rate.')
    else:
        raise ValueError('Wrong scheduler')
    return param_to_optimize, optimizer, scheduler


