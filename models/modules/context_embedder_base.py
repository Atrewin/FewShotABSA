#!/usr/bin/env python

import torch
import logging
import sys
from pytorch_pretrained_bert.modeling import BertModel
from torchnlp.word_to_vector import GloVe
from fastNLP.embeddings import char_embedding
from fastNLP import Vocabulary

from utils.graph_util import getGraphMaps

from extention.Graph_Embedding.rgcn import RGCN



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

vocab = Vocabulary()


class ContextEmbedderBase(torch.nn.Module):
    def __init__(self):
        super(ContextEmbedderBase, self).__init__()


    def forward(self, *args, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        :param args:
        :param kwargs:
        :return: test_sent_reps, support_sent_reps
        """
        raise NotImplementedError()

    def expand_it(self, item: torch.Tensor, support_size):
        expand_shape = list(item.unsqueeze_(1).shape)  # (b*q, 1, l, e)
        expand_shape[1] = support_size
        return item.expand(expand_shape)

    def expand_support(self, item: torch.Tensor, query_size, support_size):
        #@jinhui 0802
        expand_shape = list(item.unsqueeze_(0).shape)  # (1, b*s, l, e)
        expand_shape[0] = query_size  #
        return item.expand(expand_shape).reshape(-1, support_size,expand_shape[-2], expand_shape[-1] )

    def cat_test_and_support(self, test_item, support_item):
        return torch.cat([test_item, support_item], dim=-1)

    def flatten_input(self, input_ids, segment_ids, input_mask):  # 感觉将来前已经摊平了
        """ resize shape (batch_size, support_size, cat_len) to shape (batch_size * support_size, sent_len) """
        sent_len = input_ids.shape[-1]
        input_ids = input_ids.view(-1, sent_len)
        segment_ids = segment_ids.view(-1, sent_len)
        input_mask = input_mask.view(-1, sent_len)
        return input_ids, segment_ids, input_mask

    def flatten_index(self, nwp_index):
        """ resize shape (batch_size, support_size, index_len) to shape (batch_size * support_size, index_len, 1) """
        " resize shape (batch_size*support_size, index_len) to shape (batch_size * support_size, index_len, 1) "
        nwp_sent_len = nwp_index.shape[-1]
        return nwp_index.contiguous().view(-1, nwp_sent_len, 1)

    def extract_non_word_piece_reps(self, reps, index):
        """
        Use the first word piece as entire word representation
        As we have only one index for each token, we need to expand to the size of reps dim.
        """
        # index = index.unsqueeze(-1)#@jinhui 改 发现下面的扩充维度不一致
        expand_shape = list(index.shape)  # [16,128]#128 vs 126 并不是个问题dim=-2
        expand_shape[-1] = reps.shape[-1]  # [16,126,768]  # expend index over embedding dim
        index = index.expand(expand_shape)
        nwp_reps = torch.gather(input=reps, index=index, dim=-2)  # extract over token level
        return nwp_reps


class BertContextEmbedder(ContextEmbedderBase):
    def __init__(self, opt):
        super(BertContextEmbedder, self).__init__()
        self.opt = opt
        ''' load bert '''
        self.bert = BertModel.from_pretrained(opt.bert_path)

        '''build feature maps for cat test and support reps '''

        h = self.bert.config.hidden_size
        self.feature_maps = nn.Linear(h * 2, h)






    def forward(
            self,
            query_set,
            support_set = None
    ) -> (torch.Tensor, torch.Tensor):

        """
        get context representation

        :return:
            if do concatenating representation:
                return (test_reps, support_reps):  all their shape are (batch_size, support_size, nwp_sent_len, emb_len)
            else do representation for a single sent:
                return test_reps, shape is (batch_size, nwp_sent_len, emb_len)
        """
        if support_set is not None:
            return self.concatenating_reps(
                query_set, support_set
            )
        else:
            return self.single_reps(query_set)

    def concatenating_reps(
            self,
            query_set,
            support_set
    ) -> (torch.Tensor, torch.Tensor):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len, 1)
        :param test_input_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len, 1)
        :param support_input_mask: (batch_size, support_size, support_len)
        """
        test_token_ids = query_set["token_ids"]
        test_nwp_index = query_set['wp_mask']
        test_segment_ids = query_set['segment_ids']
        test_input_mask = query_set['input_mask']
        support_token_ids = support_set["token_ids"]
        support_nwp_index = support_set['wp_mask']
        support_segment_ids = support_set['segment_ids']
        support_input_mask = support_set['input_mask']

        support_size = support_token_ids.shape[1]
        test_len = test_token_ids.shape[-1] - 2  # max len, exclude [CLS] and [SEP] token
        support_len = support_token_ids.shape[-1] - 1  # max len, exclude [SEP] token
        batch_size = support_token_ids.shape[0]
        ''' expand test input to shape: (batch_size, support_size, test_len)'''
        test_token_ids, test_segment_ids, test_input_mask, test_nwp_index = self.expand_test_item(
            test_token_ids, test_segment_ids, test_input_mask, test_nwp_index, support_size)
        ''' concat test and support '''
        input_ids = self.cat_test_and_support(test_token_ids, support_token_ids)
        segment_ids = self.cat_test_and_support(test_segment_ids, support_segment_ids)
        input_mask = self.cat_test_and_support(test_input_mask, support_input_mask)
        ''' flatten input '''
        input_ids, segment_ids, input_mask = self.flatten_input(input_ids, segment_ids, input_mask)
        test_nwp_index, support_nwp_index = self.flatten_index(test_nwp_index), self.flatten_index(support_nwp_index)
        ''' get concat reps '''
        sequence_output, _ = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        ''' extract reps '''
        # select pure sent part, remove [SEP] and [CLS], notice: seq_len1 == seq_len2 == max_len.
        test_reps = sequence_output.narrow(-2, 1, test_len)  # shape:(batch, test_len, rep_size)
        support_reps = sequence_output.narrow(-2, 2 + test_len, support_len)  # shape:(batch, support_len, rep_size)
        # select non-word-piece tokens' representation
        test_reps = self.extract_non_word_piece_reps(test_reps, test_nwp_index)
        support_reps = self.extract_non_word_piece_reps(support_reps, support_nwp_index)
        # resize to shape (batch_size, support_size, sent_len, emb_len)
        reps_size = test_reps.shape[-1]
        test_reps = test_reps.view(batch_size, support_size, -1, reps_size)
        support_reps = support_reps.view(batch_size, support_size, -1, reps_size)
        return test_reps, support_reps

    def single_reps(
            self,
            query_set
    ) -> (torch.Tensor, torch.Tensor):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len, 1)
        :param test_input_mask: (batch_size, test_len)

        """

        test_token_ids = query_set["token_ids"]
        test_nwp_index = query_set['wp_mask']
        test_segment_ids = query_set['segment_ids']
        test_input_mask = query_set['input_mask']


        test_len = test_token_ids.shape[-1] - 2  # max len, exclude [CLS] and [SEP] token
        batch_size = test_token_ids.shape[0]
        ''' get bert reps '''
        test_sequence_output, _ = self.bert(
            test_token_ids, test_segment_ids, test_input_mask, output_all_encoded_layers=False)
        ''' extract reps '''
        # select pure sent part, remove [SEP] and [CLS], notice: seq_len1 == seq_len2 == max_len.
        test_reps = test_sequence_output.narrow(-2, 1, test_len)  # shape:(batch, test_len, rep_size)
        # select non-word-piece tokens' representation
        test_reps = self.extract_non_word_piece_reps(test_reps, test_nwp_index)
        return test_reps

    def expand_test_item(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_input_mask: torch.Tensor,
            test_nwp_index: torch.Tensor,
            support_size: int,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        return self.expand_it(test_token_ids, support_size), self.expand_it(test_segment_ids, support_size), \
               self.expand_it(test_input_mask, support_size), self.expand_it(test_nwp_index, support_size)



class BertCatGraphContextEmbedder(ContextEmbedderBase):
    # 需要保持统一的对外API，（但是input有多余的参数，不知道是否需要改变其他，还是在入口处变为if-else）# 当前变为if-else
    def __init__(self, opt):
        super(BertCatGraphContextEmbedder, self).__init__()
        self.opt = opt
        ''' load bert '''
        self.bert = BertModel.from_pretrained(opt.bert_path)

        ''' load rcnn '''
        self.rcnn = self.load_rcnn(opt.maps_root, opt.rcnn_checkpoint)#@jinhui 0731


    def forward(
            self,
            query_set,
            support_set = None,

    ) -> (torch.Tensor, torch.Tensor):

        """
        get context representation

        :return:
            if do concatenating representation:
                return (test_reps, support_reps):  all their shape are (batch_size, support_size, nwp_sent_len, emb_len)
            else do representation for a single sent:
                return test_reps, shape is (batch_size, nwp_sent_len, emb_len)
        """
        if support_set is not None:
            return self.concatenating_reps(#是Tagging 他们的一个motivation 感觉过于复杂，不一定要使用
            query_set=query_set,support_set=support_set)
        else:
            return self.single_reps(query_set=query_set)

    def concatenating_reps(
            self,
            query_set,
            support_set
    ) -> (torch.Tensor, torch.Tensor):
        """
        :param test_token_ids: (batch_size*query_size, test_len)
        :param test_segment_ids: (batch_size*query_size, test_len)
        :param test_nwp_index: (batch_size*query_size, test_len, 1)
        :param test_input_mask: (batch_size*query_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len, 1)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param support_set_sg: batch_size*support_size*[sg: {entity, edges_index, edges_type, edges_norm}]
        :param query_set_sg: batch_size*support_size*[sg: {entity, edges_index, edges_type, edges_norm}]

        """
        query_set_shape = list(query_set["token_ids"].shape)
        test_token_ids = query_set["token_ids"].reshape(-1, query_set_shape[-1]) #(batch_size,query_size, test_len) = (batch_size*query_size, test_len)
        test_nwp_index = query_set['wp_mask'].reshape(-1, query_set_shape[-1])
        test_segment_ids = query_set['segment_ids'].reshape(-1, query_set_shape[-1])
        test_input_mask = query_set['input_mask'].reshape(-1, query_set_shape[-1])
        support_token_ids = support_set["token_ids"]
        support_nwp_index = support_set['wp_mask']
        support_segment_ids = support_set['segment_ids']
        support_input_mask = support_set['input_mask']
        support_set_sg = support_set["sg"]
        query_set_sg = query_set["sg"]

        support_size = support_token_ids.shape[1]
        test_len = test_token_ids.shape[-1] - 2  # max len, exclude [CLS] and [SEP] token
        support_len = support_token_ids.shape[-1] - 1  # max len, exclude [SEP] token
        batch_size = support_token_ids.shape[0]
        ''' expand test input to shape: (batch_size*qurey_size , support_size, test_len)'''
        test_token_ids, test_segment_ids, test_input_mask, test_nwp_index = self.expand_test_item(
            test_token_ids, test_segment_ids, test_input_mask, test_nwp_index, support_size)

        #@jinhui 0802 认为依然需要expand_support_item 未实现

        # @jinhui 0802 不写先 未实现完整
        print("这里考虑到太过复杂， 未必会使用，暂时不实现")
        ''' concat test and support '''
        input_ids = self.cat_test_and_support(test_token_ids, support_token_ids)
        segment_ids = self.cat_test_and_support(test_segment_ids, support_segment_ids)
        input_mask = self.cat_test_and_support(test_input_mask, support_input_mask)
        ''' flatten input '''
        input_ids, segment_ids, input_mask = self.flatten_input(input_ids, segment_ids, input_mask)
        test_nwp_index, support_nwp_index = self.flatten_index(test_nwp_index), self.flatten_index(support_nwp_index)
        ''' get concat reps '''
        sequence_output, _ = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)

        # 这里的batch 操作不知道RGCNN是否会是一致的。#bert 通过flatten_input来保持一致性了

        ''' extract bert reps '''
        # select pure sent part, remove [SEP] and [CLS], notice: seq_len1 == seq_len2 == max_len.
        test_reps = sequence_output.narrow(-2, 1, test_len)  # shape:(batch, test_len, rep_size)
        support_reps = sequence_output.narrow(-2, 2 + test_len, support_len)  # shape:(batch, support_len, rep_size)
        # select non-word-piece tokens' representation
        test_reps = self.extract_non_word_piece_reps(test_reps, test_nwp_index)
        support_reps = self.extract_non_word_piece_reps(support_reps, support_nwp_index)

        ''' extract graph reps ''' #jinhui 未实现
        test_graph_reps = self.extract_non_word_piece_graph_seq_reps(query_set_sg)
        support_graph_reps = self.extract_non_word_piece_graph_seq_reps(support_set_sg)


        '''cat bert and graph reps'''




        # resize to shape (batch_size, support_size, sent_len, emb_len)
        reps_size = test_reps.shape[-1]
        test_reps = test_reps.view(batch_size, support_size, -1, reps_size)
        support_reps = support_reps.view(batch_size, support_size, -1, reps_size)
        return test_reps, support_reps

    def single_reps(
            self,
            query_set
    ) -> (torch.Tensor, torch.Tensor):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len, 1)
        :param test_input_mask: (batch_size, test_len)
        """

        test_token_ids = query_set["token_ids"]
        test_nwp_index = query_set['wp_mask']
        test_segment_ids = query_set['segment_ids']
        test_input_mask = query_set['input_mask']

        test_len = test_token_ids.shape[-1] - 2  # max len, exclude [CLS] and [SEP] token
        batch_size = test_token_ids.shape[0]
        ''' get bert reps '''
        test_sequence_output, _ = self.bert(
            test_token_ids, test_segment_ids, test_input_mask, output_all_encoded_layers=False)
        ''' extract reps '''
        # select pure sent part, remove [SEP] and [CLS], notice: seq_len1 == seq_len2 == max_len.
        test_reps = test_sequence_output.narrow(-2, 1, test_len)  # shape:(batch, test_len, rep_size)
        # select non-word-piece tokens' representation
        test_reps = self.extract_non_word_piece_reps(test_reps, test_nwp_index)
        return test_reps

    def expand_test_item(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_input_mask: torch.Tensor,
            test_nwp_index: torch.Tensor,
            support_size: int,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        return self.expand_it(test_token_ids, support_size), self.expand_it(test_segment_ids, support_size), \
               self.expand_it(test_input_mask, support_size), self.expand_it(test_nwp_index, support_size)

    def expand_it(self, item: torch.Tensor, support_size):
        expand_shape = list(item.unsqueeze_(1).shape)
        expand_shape[1] = support_size #NQ*NS
        return item.expand(expand_shape)

    def cat_test_and_support(self, test_item, support_item):
        return torch.cat([test_item, support_item], dim=-1)

    def flatten_input(self, input_ids, segment_ids, input_mask):#感觉将来前已经摊平了
        """ resize shape (batch_size, support_size, cat_len) to shape (batch_size * support_size, sent_len) """
        sent_len = input_ids.shape[-1]
        input_ids = input_ids.view(-1, sent_len)
        segment_ids = segment_ids.view(-1, sent_len)
        input_mask = input_mask.view(-1, sent_len)
        return input_ids, segment_ids, input_mask

    def flatten_index(self, nwp_index):
        """ resize shape (batch_size, support_size, index_len) to shape (batch_size * support_size, index_len, 1) """
        " resize shape (batch_size*support_size, index_len) to shape (batch_size * support_size, index_len, 1) "
        nwp_sent_len = nwp_index.shape[-1]
        return nwp_index.contiguous().view(-1, nwp_sent_len, 1)

    def extract_non_word_piece_reps(self, reps, index):
        """
        Use the first word piece as entire word representation
        As we have only one index for each token, we need to expand to the size of reps dim.
        """
        # index = index.unsqueeze(-1)#@jinhui 改 发现下面的扩充维度不一致
        expand_shape = list(index.shape)#[16,128]#128 vs 126 并不是个问题dim=-2
        expand_shape[-1] = reps.shape[-1] #[16,126,768]  # expend index over embedding dim
        index = index.expand(expand_shape)
        nwp_reps = torch.gather(input=reps, index=index, dim=-2)  # extract over token level
        return nwp_reps

    def load_rcnn(self, maps_root="None", checkpoint="none"):
        n_bases = 4
        relation_map, concept_map, unique_nodes_mapping = getGraphMaps(maps_root)
                                                #dropout=0, 与训练时的0.5 不一致导致模型参数无法载入
        model = RGCN(len(unique_nodes_mapping), len(relation_map), num_bases=n_bases, dropout=0).cuda()

        model.load_state_dict(torch.load(checkpoint))
        return model

    def extract_non_word_piece_graph_seq_reps(self, sg_list: []):
        """
        parm: sg_list： a list for samples sg by batch_size*support_size*[sg: {entity, edges_index, edges_type, edges_norm}]
        """

        # @jinhui 未实现
        return a




import torch.nn as nn
class BertSeparateContextEmbedder(BertContextEmbedder):
    def __init__(self, opt):
        super(BertSeparateContextEmbedder, self).__init__(opt)


    def forward(
            self,query_set, support_set=None
    ) -> (torch.Tensor, torch.Tensor):
        """
        get context representation
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len, 1)
        :param test_input_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len, 1)
        :param support_input_mask: (batch_size, support_size, support_len)
        :return: (test_reps, support_reps):  all their shape are (batch_size, support_size, nwp_sent_len, emb_len)
        """

        if support_set is not None:
            return self.concatenating_reps(
                query_set=query_set, support_set=support_set
            )
        else:
            return self.single_reps(query_set=query_set)

    def concatenating_reps(
            self,
            query_set, support_set=None
    ) -> (torch.Tensor, torch.Tensor):
        query_set_shape = list(query_set["token_ids"].shape)
        test_token_ids = query_set["token_ids"].reshape(-1, query_set_shape[-1])  # (batch_size,query_size, test_len) = (batch_size*query_size, test_len)
        test_nwp_index = query_set['wp_mask'].reshape(-1, query_set_shape[-1])
        test_segment_ids = query_set['segment_ids'].reshape(-1, query_set_shape[-1])
        test_input_mask = query_set['input_mask'].reshape(-1, query_set_shape[-1])
        support_token_ids = support_set["token_ids"]
        support_nwp_index = support_set['wp_mask']
        support_segment_ids = support_set['segment_ids']
        support_input_mask = support_set['input_mask']


        # datalodar 不应该将 batch_size拉平 但是都写了就往下写 还是不太对，下面的逻辑要用到batch size
        support_size = support_token_ids.shape[1]#max length @jinhui 形状好像没有对齐, 看flatten_input内部
        test_len = test_token_ids.shape[-1] - 2  # max len, exclude [CLS] and [SEP] token
        support_len = support_token_ids.shape[-1] - 1  # max len, exclude [SEP] token
        batch_size = support_token_ids.shape[0]# batch_size*support_size
        query_size = query_set_shape[1]
        ''' flatten input '''
        support_token_ids, support_segment_ids, support_input_mask = self.flatten_input(# 进去前已经摊平 现在前面没有flatten
            support_token_ids, support_segment_ids, support_input_mask)
        support_nwp_index = self.flatten_index(support_nwp_index)#(batch_size, support_size, index_len) to (batch_size * support_size, index_len, 1)

        test_token_ids, test_segment_ids, test_input_mask = self.flatten_input(
            test_token_ids, test_segment_ids, test_input_mask)

        test_nwp_index = self.flatten_index(test_nwp_index)

        ''' get bert reps ''' # word reps, sentence reps
        test_sequence_output, _ = self.bert(
            test_token_ids, test_segment_ids, test_input_mask, output_all_encoded_layers=False)
        support_sequence_output, _ = self.bert(
            support_token_ids, support_segment_ids, support_input_mask, output_all_encoded_layers=False)
        ''' extract reps '''
        # select pure sent part, remove [SEP] and [CLS], notice: seq_len1 == seq_len2 == max_len.
        test_reps = test_sequence_output.narrow(-2, 1, test_len)  # shape:(batch, test_len, rep_size)
        support_reps = support_sequence_output.narrow(-2, 1, support_len)  # shape:(batch * support_size, support_len, rep_size)
        # select non-word-piece tokens' representation
        test_reps = self.extract_non_word_piece_reps(test_reps, test_nwp_index)#@jinhui 使用index进行选值填充
        support_reps = self.extract_non_word_piece_reps(support_reps, support_nwp_index)
        # resize to shape (batch_size, support_size, sent_len, emb_len)
        reps_size = test_reps.shape[-1]

        test_reps = self.expand_it(test_reps, support_size)# [q || s] #(b*q,s,len,emb)

        #@jinhui 感觉原代码没有实现cat 0802
        # @jinhui # 感觉还未到达[q || s]的目标 猜测意图为讲两个人的表的合并起来
        # 补充，需要把support to (b*q,s,len,emb)
        support_reps_temp = self.expand_support(support_reps, query_size=query_size, support_size=support_size)
        test_reps = self.cat_test_and_support(test_reps, support_reps_temp)  # (b*q,s,len,emb*2)
        test_reps = self.feature_maps_fc(test_reps)

        support_reps = support_reps.view(batch_size, support_size, -1, reps_size)
        return test_reps.contiguous(), support_reps.contiguous()


class BertGraphSeparateContextEmbedder(BertContextEmbedder):
    def __init__(self, opt):
        super(BertGraphSeparateContextEmbedder, self).__init__(opt)


        ''' load rcnn '''
        self.rcnn = self.load_rcnn(opt.maps_root, opt.rcnn_checkpoint) #@jinhui 0731

        bert_hidm = self.bert.config.hidden_size
        graph_hidm = 100
        self.fuse_bert_graph = nn.Linear(bert_hidm+graph_hidm, bert_hidm)

    def forward(
            self,query_set, support_set=None
    ) -> (torch.Tensor, torch.Tensor):
        """
        get context representation
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len, 1)
        :param test_input_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len, 1)
        :param support_input_mask: (batch_size, support_size, support_len)
        :return: (test_reps, support_reps):  all their shape are (batch_size, support_size, nwp_sent_len, emb_len)
        """

        if support_set is not None:
            return self.concatenating_reps(
                query_set=query_set, support_set=support_set
            )
        else:
            return self.single_reps(query_set=query_set)

    def concatenating_reps(
            self,
            query_set, support_set=None
    ) -> (torch.Tensor, torch.Tensor):
        query_set_shape = list(query_set["token_ids"].shape)
        test_token_ids = query_set["token_ids"].reshape(-1, query_set_shape[-1])  # (batch_size,query_size, test_len) = (batch_size*query_size, test_len)
        test_nwp_index = query_set['wp_mask'].reshape(-1, query_set_shape[-1])
        test_segment_ids = query_set['segment_ids'].reshape(-1, query_set_shape[-1])
        test_input_mask = query_set['input_mask'].reshape(-1, query_set_shape[-1])
        support_token_ids = support_set["token_ids"]
        support_nwp_index = support_set['wp_mask']
        support_segment_ids = support_set['segment_ids']
        support_input_mask = support_set['input_mask']
        support_set_sg = support_set["sg"]
        query_set_sg = query_set["sg"]

        # datalodar 不应该将 batch_size拉平 但是都写了就往下写 还是不太对，下面的逻辑要用到batch size
        support_size = support_token_ids.shape[1]#max length @jinhui 形状好像没有对齐, 看flatten_input内部
        test_len = test_token_ids.shape[-1] - 2  # max len, exclude [CLS] and [SEP] token
        support_len = support_token_ids.shape[-1] - 1  # max len, exclude [SEP] token
        batch_size = support_token_ids.shape[0]# batch_size*support_size
        query_size = query_set_shape[1]
        ''' flatten input '''
        support_token_ids, support_segment_ids, support_input_mask = self.flatten_input(# 进去前已经摊平 现在前面没有flatten
            support_token_ids, support_segment_ids, support_input_mask)
        support_nwp_index = self.flatten_index(support_nwp_index)#(batch_size, support_size, index_len) to (batch_size * support_size, index_len, 1)

        test_token_ids, test_segment_ids, test_input_mask = self.flatten_input(
            test_token_ids, test_segment_ids, test_input_mask)

        test_nwp_index = self.flatten_index(test_nwp_index)

        ''' get bert reps ''' # word reps, sentence reps
        test_sequence_output, _ = self.bert(
            test_token_ids, test_segment_ids, test_input_mask, output_all_encoded_layers=False)
        support_sequence_output, _ = self.bert(
            support_token_ids, support_segment_ids, support_input_mask, output_all_encoded_layers=False)
        ''' extract reps '''
        # select pure sent part, remove [SEP] and [CLS], notice: seq_len1 == seq_len2 == max_len.
        test_reps = test_sequence_output.narrow(-2, 1, test_len)  # shape:(batch, test_len, rep_size)
        support_reps = support_sequence_output.narrow(-2, 1, support_len)  # shape:(batch * support_size, support_len, rep_size)
        # select non-word-piece tokens' representation
        test_reps = self.extract_non_word_piece_reps(test_reps, test_nwp_index)#@jinhui 使用index进行选值填充
        support_reps = self.extract_non_word_piece_reps(support_reps, support_nwp_index)


        ''' extract graph reps '''  # jinhui 0802
        test_graph_reps = self.extract_non_word_piece_graph_seq_reps(query_set_sg)#(batch*query_szie, s_len, 100)
        support_graph_reps = self.extract_non_word_piece_graph_seq_reps(support_set_sg)#(batch*support_size, s_len, 100)

        '''cat bert and graph reps'''
        test_reps_shape = list(test_reps.shape)
        test_reps = test_reps.reshape(-1, test_reps_shape[-1])

        test_graph_reps_shape = list(test_graph_reps.shape)
        test_graph_reps = test_graph_reps.reshape(-1, test_graph_reps_shape[-1])

        test_reps = torch.cat([test_reps, test_graph_reps], dim=-1)

        support_reps_shape = list(support_reps.shape)
        support_reps = support_reps.reshape(-1, support_reps_shape[-1])

        support_graph_reps_shape = list(support_graph_reps.shape)
        support_graph_reps = test_graph_reps.reshape(-1, support_graph_reps_shape[-1])
        support_reps = torch.cat([support_reps, support_graph_reps], dim=-1)

        ''' feature maps to hidm = emb_len'''
        test_reps = self.fuse_bert_graph(test_reps)
        support_reps = self.fuse_bert_graph(support_reps)

        # resize to pre-shape
        test_reps = test_reps.reshape(test_reps_shape)
        support_reps = support_reps.reshape(support_reps_shape)
        # resize to shape (batch_size*query_size, support_size, sent_len, emb_len)

        reps_size = test_reps.shape[-1]
        test_reps = self.expand_it(test_reps, support_size)#  #(b*q,s,len,emb)

        #@jinhui # 感觉还未到达[q || s]的目标 猜测意图为讲两个人的表的合并起来
        # 补充，需要把support to (b*q,s,len,emb)
        support_reps_temp = self.expand_support(support_reps, query_size=query_size,support_size=support_size)
        test_reps = self.cat_test_and_support(test_reps, support_reps_temp)#(b*q,s,len,emb*2)

        test_reps = self.feature_maps_fc(test_reps)
        support_reps = support_reps.view(batch_size, support_size, -1, reps_size)
        return test_reps.contiguous(), support_reps.contiguous()

    def feature_maps_fc(self, res):
        shape = list(res.shape)#
        res = res.reshape(-1, shape[-1])
        res = self.feature_maps(res)
        shape[-1] = list(res.shape)[-1]
        return res.reshape(shape)

    def extract_non_word_piece_graph_seq_reps(self, sg_list):
        reps_list = []
        for sg in sg_list:
            reps = self.rcnn(sg.entity, sg.edge_index, sg.edge_type, sg.edge_norm)
            reps_list.append(reps)
        reps = torch.stack(reps_list,dim=0)

        return reps





        return a

