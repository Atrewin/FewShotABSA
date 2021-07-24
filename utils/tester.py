# coding: utf-8
from typing import List, Tuple, Dict
import torch
import logging
import sys
import os
import copy
import json
import collections
import subprocess
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# My Staff
from utils.iter_helper import PadCollate, FewShotDataset
from utils.device_helper import prepare_model
from utils.model_helper import make_model, load_model
from models.few_shot_seq_labeler import FewShotSeqLabeler


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


RawResult = collections.namedtuple("RawResult", ["feature", "prediction"])


class TesterBase:
    """
    Support features:
        - multi-gpu [accelerating]
        - distributed gpu [accelerating]
        - padding when forward [better result & save space]
    """
    def __init__(self, opt, device, n_gpu):
        if opt.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                opt.gradient_accumulation_steps))

        self.opt = opt
        # Following is used to split the batch to save space
        self.batch_size = opt.test_batch_size
        self.device = device
        self.n_gpu = n_gpu

    def do_test(self, model: torch.nn.Module, test_features, id2label: dict,
                log_mark: str = 'test_pred'):
        logger.info("***** Running eval *****")
        # print("***** Running eval *****")
        logger.info("  Num features = %d", len(test_features))
        logger.info("  Batch size = %d", self.batch_size)
        all_results = []

        model.eval()
        data_loader = test_features
        scores = []
        for batch in tqdm(data_loader, desc="Eval-Batch Progress"):

            with torch.no_grad():
                predictions = self.do_forward(batch, model) # 有了预测，还是下面要计算得分
                predictions = torch.tensor(predictions, dtype=torch.long)
                labels = batch[-1]# 有个问题是pad的部分好像还没移除掉
                score = self.accuracy(predictions, labels)# 里面没有实现
                scores.append(score)

        total_scores = 0
        for score in scores:
            total_scores += score
        return total_scores/len(scores)

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor)).item()#0.2500,

    def get_data_loader(self, features):
        dataset = TensorDataset([self.unpack_feature(f) for f in features])
        if self.opt.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
        return data_loader

    def clone_model(self, model, id2label):
        # get a new instance
        return copy.deepcopy(model)

    def unpack_feature(self, feature) -> List[torch.Tensor]:
        raise NotImplementedError

    def do_forward(self, batch, model):
        prediction = model(*batch)
        return prediction

    def eval_predictions(self, *args, **kwargs) -> float:



        raise NotImplementedError


class FewShotTester(TesterBase):
    """
        Support features:
            - multi-gpu [accelerating]
            - distributed gpu [accelerating]
            - padding when forward [better result & save space]
    """
    def __init__(self, opt, device, n_gpu):
        super(FewShotTester, self).__init__(opt, device, n_gpu)

    def get_data_loader(self, features):
        dataset = FewShotDataset([self.unpack_feature(f) for f in features])
        if self.opt.local_rank == -1:
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        pad_collate = PadCollate(dim=-1, sp_dim=-2, sp_item_idx=[3, 8, 12])  # nwp_index, spt_tgt need special padding
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=pad_collate)
        return data_loader

    def eval_predictions(self, all_results: List[RawResult], id2label: dict, log_mark: str) -> float:
        """ Our result score is average score of all few-shot batches. """
        all_batches = self.reform_few_shot_batch(all_results)
        all_scores = []
        for b_id, fs_batch in all_batches:
            f1 = self.eval_one_few_shot_batch(b_id, fs_batch, id2label, log_mark)
            all_scores.append(f1)
        return sum(all_scores) * 1.0 / len(all_scores)

    def eval_one_few_shot_batch(self, b_id, fs_batch: List[RawResult], id2label: dict, log_mark: str) -> float:
        pred_file_name = '{}.{}.txt'.format(log_mark, b_id)
        output_prediction_file = os.path.join(self.opt.output_dir, pred_file_name)
        if self.opt.task == 'sl':
            self.writing_sl_prediction(fs_batch, output_prediction_file, id2label)
            precision, recall, f1 = self.eval_with_script(output_prediction_file)
        elif self.opt.task == 'sc':
            precision, recall, f1 = self.writing_sc_prediction(fs_batch, output_prediction_file, id2label)
        else:
            raise ValueError("Wrong task.")
        return f1

    def writing_sc_prediction(self, fs_batch: List[RawResult], output_prediction_file: str, id2label: dict):
        tp, fp, fn = 0, 0, 0
        writing_content = []
        for result in fs_batch:
            pred_ids = result.prediction  # prediction is directly the predict ids [pad is removed in decoder]
            feature = result.feature
            pred_label = set([id2label[pred_id] for pred_id in pred_ids])
            label = set(feature.test_feature_item.data_item.label)
            writing_content.append({
                'seq_in': feature.test_feature_item.data_item.seq_in,
                'pred': list(pred_label),
                'label': list(label),
            })
            tp, fp, fn = self.update_f1_frag(pred_label, label, tp, fp, fn)  # update tp, fp, fn

        with open(output_prediction_file, "w") as writer:
            json.dump(writing_content, writer, indent=2)
        return self.compute_f1(tp, fp, fn)

    def update_f1_frag(self, pred_label, label, tp=0, fp=0, fn=0):
        tp += len(pred_label & label)
        fp += len(pred_label - label)
        fn += len(label - pred_label)
        return tp, fp, fn

    def compute_f1(self, tp, fp, fn):
        tp += 0.0000001  # to avoid zero division
        fp += 0.0000001
        fn += 0.0000001
        precision = 1.0 * tp / (tp + fp)
        recall = 1.0 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def writing_sl_prediction(self, fs_batch: List[RawResult], output_prediction_file: str, id2label: dict):
        writing_content = []
        for result in fs_batch:
            prediction = result.prediction
            feature = result.feature
            pred_ids = prediction  # prediction is directly the predict ids
            if len(pred_ids) != len(feature.test_feature_item.data_item.seq_in):
                raise RuntimeError("Failed to align the pred_ids to texts: {},{} \n{},{} \n{},{}".format(
                    len(pred_ids), pred_ids,
                    len(feature.test_feature_item.data_item.seq_in), feature.test_feature_item.data_item.seq_in,
                    len(feature.test_feature_item.data_item.seq_out), feature.test_feature_item.data_item.seq_out
                ))
            for pred_id, word, true_label in zip(pred_ids, feature.test_feature_item.data_item.seq_in, feature.test_feature_item.data_item.seq_out):
                pred_label = id2label[pred_id]
                writing_content.append('{0} {1} {2}'.format(word, true_label, pred_label))
            writing_content.append('')
        with open(output_prediction_file, "w") as writer:
            writer.write('\n'.join(writing_content))

    def eval_with_script(self, output_prediction_file):
        script_args = ['perl', self.opt.eval_script]
        with open(output_prediction_file, 'r') as res_file:
            p = subprocess.Popen(script_args, stdout=subprocess.PIPE, stdin=res_file)
            p.wait()

            std_results = p.stdout.readlines()
            if self.opt.verbose:
                for r in std_results:
                    print(r)
            std_results = str(std_results[1]).split()
        precision = float(std_results[3].replace('%;', ''))
        recall = float(std_results[5].replace('%;', ''))
        f1 = float(std_results[7].replace('%;', '').replace("\\n'", ''))
        return precision, recall, f1

    def reform_few_shot_batch(self, all_results: List[RawResult]) -> List[List[Tuple[int, RawResult]]]:
        """
        Our result score is average score of all few-shot batches.
        So here, we classify all result according to few-shot batch id.
        """
        all_batches = {}
        for result in all_results:
            b_id = result.feature.batch_gid
            if b_id not in all_batches:
                all_batches[b_id] = [result]
            else:
                all_batches[b_id].append(result)
        return sorted(all_batches.items(), key=lambda x: x[0])


    def do_forward(self, batch, model):

        support_set, query_set, support_label, query_label = batch
        if torch.cuda.is_available():
            for k in support_set:
                support_set[k] = support_set[k].cuda()
            for k in query_set:
                query_set[k] = query_set[k].cuda()
            support_label = support_label.cuda()
            query_label = query_label.cuda()

        prediction = model(
            query_set["token_ids"],
            query_set['wp_mask'],
            query_set['segment_ids'],
            query_set['input_mask'],
            query_set["output_mask"],
            support_set["token_ids"],
            support_set['wp_mask'],
            support_set['segment_ids'],
            support_set['input_mask'],
            support_set["output_mask"],
            support_label,
            query_label,
        )
        return prediction

    def get_value_from_order_dict(self, order_dict, key):
        """"""
        for k, v in order_dict.items():
            if key in k:
                return v
        return []

    def clone_model(self, model, id2label):
        """ clone only part of params """
        # deal with data parallel model
        new_model: FewShotSeqLabeler
        old_model: FewShotSeqLabeler
        if self.opt.local_rank != -1 or self.n_gpu > 1 and hasattr(model, 'module'):  # the model is parallel class here
            old_model = model.module
        else:
            old_model = model
        emission_dict = old_model.emission_scorer.state_dict()
        old_num_tags = len(self.get_value_from_order_dict(emission_dict, 'label_reps'))

        config = {'num_tags': len(id2label), 'id2label': id2label}
        # if 'num_anchors' in old_model.config:
        #     config['num_anchors'] = old_model.config['num_anchors']  # Use previous model's random anchors.
        # get a new instance for different domain
        new_model = make_model(opt=self.opt, config=config)
        new_model = prepare_model(self.opt, new_model, self.device, self.n_gpu)
        if self.opt.local_rank != -1 or self.n_gpu > 1:
            sub_new_model = new_model.module
        else:
            sub_new_model = new_model
        ''' copy weights and stuff '''
        if old_model.opt.task == 'sl' and old_model.transition_scorer:
            # copy one-by-one because target transition and decoder will be left un-assigned
            sub_new_model.context_embedder.load_state_dict(old_model.context_embedder.state_dict())
            sub_new_model.emission_scorer.load_state_dict(old_model.emission_scorer.state_dict())
            for param_name in ['backoff_trans_mat', 'backoff_start_trans_mat', 'backoff_end_trans_mat']:
                sub_new_model.transition_scorer.state_dict()[param_name].copy_(
                    old_model.transition_scorer.state_dict()[param_name].data)
        else:
            sub_new_model.load_state_dict(old_model.state_dict())

        return new_model


def eval_check_points(opt, tester, test_features, test_id2label, device):
    all_cpt_file = list(filter(lambda x: '.cpt.pl' in x, os.listdir(opt.saved_model_path)))
    all_cpt_file = sorted(all_cpt_file,
                          key=lambda x: int(x.replace('model.step', '').replace('.cpt.pl', '')))
    max_score = 0
    for cpt_file in all_cpt_file:
        cpt_model = load_model(os.path.join(opt.saved_model_path, cpt_file))
        testing_model = tester.clone_model(cpt_model, test_id2label)
        if opt.mask_transition and opt.task == 'sl':
            testing_model.label_mask = opt.test_label_mask.to(device)
        test_score = tester.do_test(testing_model, test_features, test_id2label, log_mark='test_pred')
        if test_score > max_score:
            max_score = test_score
        logger.info('cpt_file:{} - test:{}'.format(cpt_file, test_score))
    return max_score
