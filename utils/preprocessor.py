# coding:utf-8
import collections
from typing import List, Tuple, Dict
import torch
import pickle
import json
import os
from transformers import BertTokenizer
from utils.iter_helper import pad_tensor
from utils.config import SP_TOKEN_NO_MATCH, SP_LABEL_O, SP_TOKEN_O


FeatureItem = collections.namedtuple(   # text or raw features
    "FeatureItem",
    [
        "tokens",  # tokens corresponding to input token ids, eg: word_piece tokens with [CLS], [SEP]
        "labels",  # labels for all input position, eg; label for word_piece tokens
        "data_item",
        "token_ids",
        "segment_ids",
        "nwp_index",
        "input_mask",
        "output_mask"
    ]
)

ModelInput = collections.namedtuple(  # digit features for computation
    "ModelInput",   # all element shape: test: (1, test_len) support: (support_size, support_len)
    [
        "token_ids",  # token index list
        "segment_ids",  # bert [SEP] ids
        "nwp_index",  # non-word-piece word index to extract non-word-piece tokens' reps (only useful for bert).
        "input_mask",  # [1] * len(sent), 1 for valid (tokens, cls, sep, word piece), 0 is padding in batch construction
        "output_mask",  # [1] * len(sent), 1 for valid output, 0 for padding, eg: 1 for original tokens in sl task
    ]
)



class InputBuilderBase:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, example, max_support_size, label2id
    ) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        raise NotImplementedError

    def data_item2feature_item(self, data_item, seg_id: int) -> FeatureItem:
        raise NotImplementedError

    def get_test_model_input(self, feature_item: FeatureItem) -> ModelInput:
        ret = ModelInput(
            token_ids=torch.LongTensor(feature_item.token_ids),
            segment_ids=torch.LongTensor(feature_item.segment_ids),
            nwp_index=torch.LongTensor(feature_item.nwp_index),
            input_mask=torch.LongTensor(feature_item.input_mask),
            output_mask=torch.LongTensor(feature_item.output_mask)
        )
        return ret

    def get_support_model_input(self, feature_items: List[FeatureItem], max_support_size: int) -> ModelInput:
        pad_id = self.tokenizer.vocab['[PAD]']
        token_ids = self.pad_support_set([f.token_ids for f in feature_items], pad_id, max_support_size)
        segment_ids = self.pad_support_set([f.segment_ids for f in feature_items], 0, max_support_size)
        nwp_index = self.pad_support_set([f.nwp_index for f in feature_items], [0], max_support_size)
        input_mask = self.pad_support_set([f.input_mask for f in feature_items], 0, max_support_size)
        output_mask = self.pad_support_set([f.output_mask for f in feature_items], 0, max_support_size)
        ret = ModelInput(
            token_ids=torch.LongTensor(token_ids),
            segment_ids=torch.LongTensor(segment_ids),
            nwp_index=torch.LongTensor(nwp_index),
            input_mask=torch.LongTensor(input_mask),
            output_mask=torch.LongTensor(output_mask)
        )
        return ret

    def pad_support_set(self, item_lst: List[List[int]], pad_value: int, max_support_size: int) -> List[List[int]]:
        """
        pre-pad support set to insure: 1. each spt set has same sent num 2. each sent has same length
        (do padding here because: 1. all support sent are considered as one tensor input  2. support set size is small)
        :param item_lst:
        :param pad_value:
        :param max_support_size:
        :return:
        """
        ''' pad sentences '''
        max_sent_len = max([len(x) for x in item_lst])  # max length among one
        ret = []
        for sent in item_lst:
            temp = sent[:]
            while len(temp) < max_sent_len:
                temp.append(pad_value)
            ret.append(temp)
        ''' pad support set size '''
        pad_item = [pad_value for _ in range(max_sent_len)]
        while len(ret) < max_support_size:
            ret.append(pad_item)
        return ret

    def digitizing_input(self, tokens: List[str], seg_id: int) -> (List[int], List[int]):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [seg_id for _ in range(len(tokens))]
        return token_ids, segment_ids

    def tokenizing(self, item):
        """ Possible tokenizing for item """
        pass


class BertInputBuilder(InputBuilderBase):
    def __init__(self, tokenizer, opt):
        super(BertInputBuilder, self).__init__(tokenizer)
        self.opt = opt
        self.test_seg_id = 0
        self.support_seg_id = 0 if opt.context_emb == 'sep_bert' else 1  # 1 to cat support and query to get reps
        self.seq_ins = {}

    def __call__(self, example, max_support_size, label2id) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        test_feature_item, test_input = self.prepare_test(example)
        support_feature_items, support_input = self.prepare_support(example, max_support_size)
        return test_feature_item, test_input, support_feature_items, support_input

    def prepare_test(self, example):
        test_feature_item = self.data_item2feature_item(data_item=example.test_data_item, seg_id=0)
        test_input = self.get_test_model_input(test_feature_item)
        return test_feature_item, test_input

    def prepare_support(self, example, max_support_size):
        support_feature_items = [self.data_item2feature_item(data_item=s_item, seg_id=self.support_seg_id) for s_item in
                                 example.support_data_items]
        support_input = self.get_support_model_input(support_feature_items, max_support_size)
        return support_feature_items, support_input

    def data_item2feature_item(self, data_item, seg_id: int) -> FeatureItem:
        """ get feature_item for bert, steps: 1. do digitalizing 2. make mask """
        wp_mark, wp_text = self.tokenizing(data_item)
        if self.opt.task == 'sl':  # use word-level labels  [opt.label_wp is supported by model now.]
            labels = self.get_wp_label(data_item.seq_out, wp_text, wp_mark) if self.opt.label_wp else data_item.seq_out
        else:  # use sentence level labels
            labels = data_item.label
            if 'None' not in labels:
                # transfer label to index type such as `label_1`
                if self.opt.index_label:
                    labels = [self.opt.label2index_type[label] for label in labels]
                if self.opt.unused_label:
                    labels = [self.opt.label2unused_type[label] for label in labels]
        tokens = ['[CLS]'] + wp_text + ['[SEP]'] if seg_id == 0 else wp_text + ['[SEP]']
        token_ids, segment_ids = self.digitizing_input(tokens=tokens, seg_id=seg_id)
        nwp_index = self.get_nwp_index(wp_mark)
        input_mask = [1] * len(token_ids)
        output_mask = [1] * len(labels)   # For sl: it is original tokens; For sc: it is labels
        ret = FeatureItem(
            tokens=tokens,
            labels=labels,
            data_item=data_item,
            token_ids=token_ids,
            segment_ids=segment_ids,
            nwp_index=nwp_index,
            input_mask=input_mask,
            output_mask=output_mask,
        )
        return ret

    def get_nwp_index(self, word_piece_mark: list) -> torch.Tensor:
        """ get index of non-word-piece tokens, which is used to extract non-wp bert embedding in batch manner """
        return torch.nonzero(torch.LongTensor(word_piece_mark) - 1).tolist()  # wp mark word-piece with 1, so - 1

    def tokenizing(self, item):
        """ Do tokenizing and get word piece data and get label on pieced words. """
        wp_text = self.tokenizer.wordpiece_tokenizer.tokenize(' '.join(item.seq_in))
        wp_mark = [int((len(w) > 2) and w[0] == '#' and w[1] == '#') for w in wp_text]  # mark wp as 1
        return wp_mark, wp_text

    def get_wp_label(self, label_lst, wp_text, wp_mark, label_pieced_words=False):
        """ get label on pieced words. """
        wp_label, label_idx = [], 0
        for ind, mark in enumerate(wp_mark):
            if mark == 0:  # label non-pieced token with original label
                wp_label.append(label_lst[label_idx])
                label_idx += 1  # pointer on non-wp labels
            elif mark == 1:  # label word-piece with whole word's label or with  [PAD] label
                pieced_label = wp_label[-1].replace('B-', 'I-') if label_pieced_words else '[PAD]'
                wp_label.append(pieced_label)
            if not wp_label[-1]:
                raise RuntimeError('Empty label')
        if not (len(wp_label) == len(wp_text) == len(wp_mark)):
            raise RuntimeError('ERROR: Failed to generate wp labels:{}{}{}{}{}{}{}{}{}{}{}'.format(
                len(wp_label), len(wp_text), len(wp_mark),
                '\nwp_lb', wp_label, '\nwp_text', wp_text, '\nwp_mk', wp_mark, '\nlabel', label_lst))
        return wp_label


class NormalInputBuilder(InputBuilderBase):
    def __init__(self, tokenizer):
        super(NormalInputBuilder, self).__init__(tokenizer)

    def __call__(self, example, max_support_size, label2id) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        test_feature_item = self.data_item2feature_item(data_item=example.test_data_item, seg_id=0)
        test_input = self.get_test_model_input(test_feature_item)
        support_feature_items = [self.data_item2feature_item(data_item=s_item, seg_id=1) for s_item in
                                 example.support_data_items]
        support_input = self.get_support_model_input(support_feature_items, max_support_size)
        return test_feature_item, test_input, support_feature_items, support_input

    def data_item2feature_item(self, data_item, seg_id: int) -> FeatureItem:
        """ get feature_item for bert, steps: 1. do padding 2. do digitalizing 3. make mask """
        tokens, labels = data_item.seq_in, data_item.seq_out
        token_ids, segment_ids = self.digitizing_input(tokens=tokens, seg_id=seg_id)
        nwp_index = [[i] for i in range(len(token_ids))]
        input_mask = [1] * len(token_ids)
        output_mask = [1] * len(data_item.seq_in)
        ret = FeatureItem(
            tokens=tokens,
            labels=labels,
            data_item=data_item,
            token_ids=token_ids,
            segment_ids=segment_ids,
            nwp_index=nwp_index,
            input_mask=input_mask,
            output_mask=output_mask,
        )
        return ret


class Preprocessor:
    """
    Class for build feature and label2id dict
    Main function:
        construct_feature
        make_dict
    """
    def __init__(
            self,
            input_builder: InputBuilderBase,
            tokenizer
    ):
        self.input_builder = input_builder
        self.tokenizer = tokenizer



def flatten(l):
    """ convert list of list to list"""
    return [item for sublist in l for item in sublist]


def make_sent_dict() -> (Dict[str, int], Dict[int, str]):
    """
    make label2id dict
    label2id must follow rules:
    For sequence labeling:
        1. id(PAD)=0 id(O)=1  2. id(B-X)=i  id(I-X)=i+1
    For (multi-label) text classification:
        1. id(PAD)=0
    """
    sent_label2id = {'[PAD]': 0, 'EMPTY': 1, 'POS': 2, 'NEU': 3, 'NEG': 4}
    sent_id2label = {0: '[PAD]', 1: 'EMPTY', 2: 'POS', 3: 'NEU', 4: 'NEG'}
    return sent_label2id, sent_id2label

def make_word_dict() -> (Dict[str, int], Dict[int, str]):
    """
    make label2id dict
    label2id must follow rules:
    For sequence labeling:
        1. id(PAD)=0 id(O)=1  2. id(B-X)=i  id(I-X)=i+1
    For (multi-label) text classification:
        1. id(PAD)=0
    """
    word_label2id = {'[PAD]': 0, 'O': 1, 'T-POS': 2, 'T-NEU': 3, 'T-NEG': 4}
    word_id2label = {0: '[PAD]', 1: "O",  2: 'T-POS', 3: 'T-NEU', 4: 'T-NEG'}
    return word_label2id, word_id2label

def make_boundary_dict() -> (Dict[str, int], Dict[int, str]):
    """
    make label2id dict
    label2id must follow rules:
    For sequence labeling:
        1. id(PAD)=0 id(O)=1  2. id(B-X)=i  id(I-X)=i+1
    For (multi-label) text classification:
        1. id(PAD)=0
    """
    boundary_label2id = {'[PAD]': 0, 'O': 1, 'T': 2}
    boundary_id2label = {0: '[PAD]', 1: "O",  2: 'T'}
    return boundary_label2id, boundary_id2label


def make_word_dict_by_file(all_files: List[str]) -> (Dict[str, int], Dict[int, str]):
    all_words = []
    word2id = {}
    for file in all_files:
        with open(file, 'r') as reader:
            raw_data = json.load(reader)
        for domain_n, domain in raw_data.items():
            # Notice: the batch here means few shot batch, not training batch
            for batch_id, batch in enumerate(domain):
                all_words.extend(batch['support']['seq_ins'])
                all_words.extend(batch['query']['seq_ins'])
    word_set = sorted(list(set(flatten(all_words))))  # sort to make embedding id fixed
    for word in ['[PAD]', '[OOV]'] + word_set:
        word2id[word] = len(word2id)
    id2word = dict([(idx, word) for word, idx in word2id.items()])
    return word2id, id2word

def make_span_dict(examples) -> (Dict[str, int], Dict[int, str]):
    """
    make label2id dict
    label2id must follow rules:
    1. id(PAD)=0 id(O)=1  2. id(B-X)=i  id(I-X)=i+1
    """
    def purify(l):
        """ remove B- and I- """
        return set([item.replace('B-', '').replace('I-', '') for item in l])

    ''' collect all label from: all test set & all support set '''
    all_labels = []
    label2id = {}
    for example in examples:
        all_labels.append(example.test_data_item.label)
        all_labels.extend([data_item.label for data_item in example.support_data_items])
    ''' collect label word set '''
    label_set = sorted(list(purify(set(flatten(all_labels)))))  # sort to make embedding id fixed
    ''' build dict '''
    label2id['[PAD]'] = len(label2id)  # '[PAD]' in first position and id is 0
    label2id['O'] = len(label2id)
    for label in label_set:
        if label == 'O':
            continue
        label2id[label] = len(label2id)
    ''' reverse the label2id '''
    id2label = dict([(idx, label) for label, idx in label2id.items()])
    return label2id, id2label

def make_mask(token_ids: torch.Tensor, label_ids: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    input_mask = (token_ids != 0).long()
    output_mask = (label_ids != 0).long()  # mask
    return input_mask, output_mask


def save_feature(path, features, label2id, id2label):
    with open(path, 'wb') as writer:
        saved_features = {
            'features': features,
            'label2id': label2id,
            'id2label': id2label,
        }
        pickle.dump(saved_features, writer)


def load_feature(path):
    with open(path, 'rb') as reader:
        saved_feature = pickle.load(reader)
        return saved_feature['features'], saved_feature['label2id'], saved_feature['id2label']


def make_preprocessor(opt):
    """ make preprocessor """
    transformer_style_embs = ['bert', 'sep_bert', 'electra']

    ''' select input_builder '''
    if opt.context_emb in transformer_style_embs:
        tokenizer = BertTokenizer.from_pretrained(opt.bert_vocab, use_multiprocessing=False)
        input_builder = BertInputBuilder(tokenizer=tokenizer, opt=opt)

    elif opt.context_emb in ['glove', 'raw']:
        word2id, id2word = make_word_dict_by_file([opt.train_path, opt.dev_path, opt.test_path])
        opt.word2id = word2id
        tokenizer = MyTokenizer(word2id=word2id, id2word=id2word)
        input_builder = NormalInputBuilder(tokenizer=tokenizer)

    elif opt.context_emb == 'elmo':
        raise NotImplementedError
    else:
        raise TypeError('wrong word representation type')

    ''' build preprocessor '''

    preprocessor = Preprocessor(input_builder=input_builder, tokenizer=tokenizer)
    return preprocessor


def make_label_mask(opt, path, label2id):
    """ disable cross domain transition """
    label_mask = [[0] * len(label2id) for _ in range(len(label2id))]
    with open(path, 'r') as reader:
        raw_data = json.load(reader)
        for domain_n, domain in raw_data.items():
            # Notice: the batch here means few shot batch, not training batch
            batch = domain[0]
            supports_labels = batch['support']['seq_outs']
            all_support_labels = set(collections._chain.from_iterable(supports_labels))
            for lb_from in all_support_labels:
                for lb_to in all_support_labels:
                    if opt.do_debug:  # when debuging, only part of labels are leveraged
                        if lb_from not in label2id or lb_to not in label2id:
                            continue
                    label_mask[label2id[lb_from]][label2id[lb_to]] = 1
    return torch.LongTensor(label_mask)


class MyTokenizer(object):
    def __init__(self, word2id, id2word):
        self.word2id = word2id
        self.id2word = id2word
        self.vocab = word2id

    def convert_tokens_to_ids(self, tokens):
        return [self.word2id[token] for token in tokens]
