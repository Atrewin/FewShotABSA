# coding:utf-8
import json
import collections
import random
from typing import List, Tuple, Dict
import torch.utils.data as data
import os
import math
import torch
import numpy as np


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, examples: dict, preprocessor, opt):

        self.examples = examples
        self.classes = list(self.examples.keys())

        self.N = 3
        self.K = opt.K
        self.Q = opt.Q
        self.preprocessor = preprocessor
        self.opt = opt

    def get_nwp_index(self, word_piece_mark: list) -> torch.Tensor:
        """ get index of non-word-piece tokens, which is used to extract non-wp bert embedding in batch manner """
        return torch.nonzero(torch.LongTensor(word_piece_mark) - 1).tolist()  # wp mark word-piece with 1, so - 1


    def get_wp_label(self, label_lst, wp_text, wp_mark, label_pieced_words=False):
        """ get label on pieced words. """
        wp_label, label_idx = [], 0
        for ind, mark in enumerate(wp_mark):
            if mark == 0:  # label non-pieced token with original label
                wp_label.append(label_lst[label_idx])
                label_idx += 1  # pointer on non-wp labels
            elif mark == 1:  # label word-piece with whole word's label or with  [PAD] label
                pieced_label = wp_label[-1] if label_pieced_words else '[PAD]'
                wp_label.append(pieced_label)
            if not wp_label[-1]:
                raise RuntimeError('Empty label')
        if not (len(wp_label) == len(wp_text) == len(wp_mark)):
            raise RuntimeError('ERROR: Failed to generate wp labels:{}{}{}{}{}{}{}{}{}{}{}'.format(
                len(wp_label), len(wp_text), len(wp_mark),
                '\nwp_lb', wp_label, '\nwp_text', wp_text, '\nwp_mk', wp_mark, '\nlabel', label_lst))
        return wp_label

    def digitizing_input(self, tokens: List[str], seg_id: int) -> (List[int], List[int]):
        token_ids = self.preprocessor.tokenizer.convert_tokens_to_ids(tokens)
        if seg_id == 1:
            print("*********************")
        segment_ids = [seg_id for _ in range(len(tokens))]
        return token_ids, segment_ids

    def __getraw__(self, item):
        '''
        :param index:
        "token_ids",  # token index list
        "nwp_index",  # non-word-piece word index to extract non-word-piece tokens' reps (only useful for bert).
        "labels",  # labels index list
        "input_mask",  # [1] * len(sent), 1 for valid (tokens, cls, sep, word piece), 0 is padding in batch construction
        "output_mask",  # [1] * len(sent), 1 for valid output, 0 for padding, eg: 1 for original tokens in sl task
        '''
        wp_text = self.preprocessor.tokenizer.wordpiece_tokenizer.tokenize(" ".join(item['text']))
        wp_mask = [int((len(w) > 2) and w[0] == '#' and w[1] == '#') for w in wp_text]
        labels = ["O"] + self.get_wp_label(item['labels'], wp_text, wp_mask) + ["O"]
        labels_index = [self.opt.word_label2id[label] for label in labels] # include the [cls] and [sep]
        tokens = ['[CLS]'] + wp_text + ['[SEP]']
        token_ids, segment_ids = self.digitizing_input(tokens=tokens, seg_id=0)
        # nwp_index = self.get_nwp_index(wp_mark)
        wp_mask = [0] + wp_mask + [0]
        input_mask = [1] * len(token_ids)
        output_mask = [1] * len(labels_index)  # For sl: it is original tokens;

        # padding
        while len(token_ids) < self.opt.max_length:
            tokens.append("**NULL**")
            token_ids.append(0)
            wp_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            labels_index.append(0)
            input_mask.append(0)
            output_mask.append(0)

        assert len(token_ids) == self.opt.max_length
        assert len(wp_mask) == self.opt.max_length
        assert len(segment_ids) == self.opt.max_length
        assert len(labels_index) == self.opt.max_length
        assert len(input_mask) == self.opt.max_length
        assert len(output_mask) == self.opt.max_length
        return token_ids, wp_mask, segment_ids, labels_index, input_mask, output_mask

    def __additem__(self, d, token_ids, wp_mask, segment_ids, input_mask, output_mask):
        d['token_ids'].append(token_ids)
        d['wp_mask'].append(wp_mask)
        d['segment_ids'].append(segment_ids)
        d['input_mask'].append(input_mask)
        d['output_mask'].append(output_mask)

    def __getitem__(self, index):

        support_classes = self.classes.copy()
        support_classes.remove("EMPTY")

        support_set = {'token_ids': [], 'wp_mask': [], 'segment_ids': [], 'input_mask': [], 'output_mask': []}
        query_set = {'token_ids': [], 'wp_mask': [], 'segment_ids': [], 'input_mask': [], 'output_mask': []}
        query_label = []
        support_label = []

        for i, class_name in enumerate(support_classes):
            indices = np.random.choice(#@jinhui抽样
                list(range(len(self.examples[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                token_ids, wp_mask, segment_ids, labels, input_mask, output_mask = self.__getraw__(
                    self.examples[class_name][j])
                token_ids = torch.tensor(token_ids).long()
                wp_mask = torch.tensor(wp_mask).long()
                segment_ids = torch.tensor(segment_ids).long()
                labels = torch.tensor(labels).long()
                input_mask = torch.tensor(input_mask).long()
                output_mask = torch.tensor(output_mask).long()
                if count < self.K:
                    self.__additem__(support_set, token_ids, wp_mask, segment_ids, input_mask, output_mask)
                    support_label.append(labels)
                else:
                    self.__additem__(query_set, token_ids, wp_mask, segment_ids, input_mask, output_mask)
                    query_label.append(labels)
                count += 1

        # add EMPTY
        indices = np.random.choice(
            list(range(len(self.examples["EMPTY"]))), self.Q, False)
        for j in indices:
            token_ids, wp_mask, segment_ids, labels, input_mask, output_mask = self.__getraw__(
                self.examples["EMPTY"][j])
            token_ids = torch.tensor(token_ids).long()
            wp_mask = torch.tensor(wp_mask).long()
            segment_ids = torch.tensor(segment_ids).long()
            labels = torch.tensor(labels).long()
            input_mask = torch.tensor(input_mask).long()
            output_mask = torch.tensor(output_mask).long()

            self.__additem__(query_set, token_ids, wp_mask, segment_ids, input_mask, output_mask)
            query_label.append(labels)

        return support_set, query_set, support_label, query_label

    def __len__(self):
        return self.opt.iter_num


def collate_fn(data):#@改 jinghui 因为这里将batch 拉平了，导致后续的处理跟不上
    batch_support = {'token_ids': [], 'wp_mask': [], 'segment_ids': [], 'input_mask': [], 'output_mask': []}
    batch_query = {'token_ids': [], 'wp_mask': [], 'segment_ids': [], 'input_mask': [], 'output_mask': []}
    batch_query_labels = []
    batch_support_labels = []
    support_sets, query_sets, support_labels, query_labels = zip(*data)
    batch_size = len(support_sets)
    support_size = len(support_labels[0])
    query_size = len(query_labels[0])
    for i in range(len(support_sets)):# batch size
        for k in support_sets[i]:# key
            batch_support[k] += support_sets[i][k]# 这里把batch size拉平了
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_query_labels += query_labels[i]
        batch_support_labels += support_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0).reshape(batch_size,support_size,-1)# list[30, 128]
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0).reshape(batch_size,query_size,-1)
    batch_query_labels = torch.stack(batch_query_labels, 0).reshape(batch_size,query_size,-1)
    batch_support_labels = torch.stack(batch_support_labels, 0).reshape(batch_size,support_size,-1)

    # @jinhui 需要把batch, support size 重新构建好
    # .reshape(batch_size,query_size,-1)
    return batch_support, batch_query, batch_support_labels, batch_query_labels



class FewShotRawDataLoader():
    def __init__(self, opt):
        super(FewShotRawDataLoader, self).__init__()
        self.opt = opt
        self.debugging = opt.do_debug
        self.idx_dict = {'O': 0, 'T-POS': 1, 'T-NEU': 2, 'T-NEG': 3}#  delete '[PAD]': 0

    def load_data(self, path, preprocessor, opt, batch_size, num_workers=1):
        """
            load few shot data set
            input:
                path: file path
            output
                examples: a list, all example loaded from path
                few_shot_batches: a list, of fewshot batch, each batch is a list of examples
                max_len: max sentence length
                self.trans_mat = (trans_mat, start_trans_mat, end_trans_mat): the A prior joint distribution probability matrix for dataset
        """

        raw_data = json.load(open(path, 'r'))
        examples = {}
        for domain_name, domain_data in raw_data.items():
            examples = domain_data# 应该有循环逻辑进行拼接
        trans_mat = self.get_trans_mat(examples)
        dataset = FewRelDataset(examples, preprocessor, opt)
        # if self.debugging:
        #     examples, few_shot_batches = examples[:8], few_shot_batches[:2]
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn)
        return iter(data_loader), trans_mat



    def get_trans_mat(self, examples):
        """
            Calculated the prior joint distribution probability matrix between tags
            input:
                examples: Dict, key is tags type
            output
                trans_mat: (trans_mat, start_trans_mat, end_trans_mat): the A prior joint distribution probability matrix for dataset
        """

        # transition matrix
        num_tags = len(self.idx_dict)
        trans_mat = torch.zeros(num_tags, num_tags, dtype=torch.int32).tolist()
        start_trans_mat = torch.zeros(num_tags, dtype=torch.int32).tolist()
        end_trans_mat = torch.zeros(num_tags, dtype=torch.int32).tolist()
        for tag_type, supports in examples.items():
            # update transition matrix
            self.update_trans_mat(trans_mat, start_trans_mat, end_trans_mat, supports)

        return (trans_mat, start_trans_mat, end_trans_mat)

    def update_trans_mat(self,
                      trans_mat: List[List[int]],
                      start_trans_mat: List[int],
                      end_trans_mat: List[int],
                      support_data: List[str]) -> None:
        for support_data_item in support_data:
            labels = support_data_item["labels"]
            if labels[-1] not in self.idx_dict.keys():
                labels[-1] = "O" #源数据处理有问题
            s_idx = self.idx_dict[labels[0]]
            e_idx = self.idx_dict[labels[-1]]

            start_trans_mat[s_idx] += 1
            end_trans_mat[e_idx] += 1
            for i in range(len(labels) - 1):
                cur_label = labels[i]
                next_label = labels[i + 1]
                start_idx = self.idx_dict[cur_label]
                end_idx = self.idx_dict[next_label]

                trans_mat[start_idx][end_idx] += 1# count




