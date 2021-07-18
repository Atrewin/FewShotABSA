#!/usr/bin/env python
import torch
from models.modules.scale_controller import ScaleControllerBase


class SpanClassificationBase(torch.nn.Module):
    def __init__(self, sim_func, emb_log=None, scaler: ScaleControllerBase = None):
        """
        :param similarity_scorer: Module for calculating token similarity
        """
        super(SpanClassificationBase, self).__init__()
        self.scaler = scaler
        self.sim_func = sim_func
        self.emb_log = emb_log
        self.log_content = ''

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_targets: torch.Tensor,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
        :param test_reps: (batch_size, support_size, test_seq_len, dim), notice: reps has been expand to support size
        :param support_reps: (batch_size, support_size, support_seq_len)
        :param test_output_mask: (batch_size, test_seq_len)
        :param support_output_mask: (batch_size, support_size, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :param label_reps: (batch_size, num_tags, dim)
        :return: emission, shape: (batch_size, test_len, no_pad_num_tags)
        """
        raise NotImplementedError()

    def mask_sim(self, sim: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor):
        """
        mask invalid similarity to 0, i.e. sim to pad token is 0 here.
        :param sim: similarity matrix (num sim, test_len, support_len)
        :param mask1: (num sim, test_len, support_len)
        :param mask2: (num sim, test_len, support_len)
        :return:
        """
        mask1 = mask1.unsqueeze(-1).float()
        mask2 = mask2.unsqueeze(-1).float()
        mask = reps_dot(mask1, mask2)
        sim = sim * mask
        return sim

    def expand_it(self, item: torch.Tensor, support_size):
        item = item.unsqueeze(1)
        expand_shape = list(item.shape)
        expand_shape[1] = support_size
        return item.expand(expand_shape)



class PrototypeSpanClassification(SpanClassificationBase):
    def __init__(self, sim_func, emb_log=None,  scaler: ScaleControllerBase = None):
        """
        :param similarity_scorer: Module for calculating token similarity
        """
        super(PrototypeSpanClassification, self).__init__(sim_func, emb_log, scaler)

    def forward(
            self,
            test_reps: torch.Tensor,
            support_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_type_target: torch.Tensor,
            label_reps: torch.Tensor = None, ) -> torch.Tensor:
        """
        :param test_reps: (batch_size, test_seq_len, emb_dim)
        :param support_reps: (batch_size, support_size, support_seq_len, emb_dim)
        :param test_output_mask: (batch_size, test_seq_len)
        :param support_output_mask: (batch_size, support_size, support_seq_len)
        :param support_type_target: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :return: similarity, shape: (batch_size, test_len, no_pad_num_tags)
        """
        # similarity = self.similarity_scorer(
        #     test_reps, support_reps, test_output_mask, support_output_mask, support_targets)
        support_size = support_reps.shape[1]
        test_len = test_reps.shape[-2]  # non-word-piece max test token num, Notice that it's different to input t len
        support_len = support_reps.shape[-2]  # non-word-piece max test token num
        emb_dim = test_reps.shape[-1]
        batch_size = test_reps.shape[0]
        num_tags = support_type_target.shape[-1]

        # flatten dim mention of support size and batch size.
        # shape (batch_size * support_size, sent_len, emb_dim)
        support_reps = support_reps.view(-1, support_len, emb_dim)

        # shape (batch_size * support_size, sent_len, num_tags)
        support_targets = support_type_target.view(batch_size * support_size, support_len, num_tags).float()

        # get prototype reps
        # shape (batch_size*support_size, num_tags, emd_dim)
        sum_reps = torch.bmm(torch.transpose(support_targets, -1, -2), support_reps)

        # sum up tag emb over support set, shape (batch_size, num_tags, emd_dim)
        sum_reps = torch.sum(sum_reps.view(batch_size, support_size, num_tags, emb_dim), dim=1)

        # get num of each tag in support set, shape: (batch_size, num_tags, 1)
        tag_count = torch.sum(support_targets.view(batch_size, -1, num_tags), dim=1).unsqueeze(-1)

        # divide by 0 occurs when the tags, such as "I-x", are not existing in support.
        tag_count = self.remove_0(tag_count)

        prototype_reps = torch.div(sum_reps, tag_count)  # shape (batch_size, num_tags, emd_dim)

        # calculate dot product
        sim_score = self.sim_func(test_reps, prototype_reps)  # shape (batch_size, sent_len, num_tags)
        output = self.get_output(sim_score, support_targets)  # shape(batch_size, test_len, no_pad_num_tag)
        return output

    def get_output(self, similarities: torch.Tensor, support_targets: torch.Tensor):
        """
        :param similarities: (batch_size, support_size, test_seq_len, support_seq_len)
        :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
        :return: emission: shape: (batch_size, test_len, no_pad_num_tags)
        """
        batch_size, test_len, num_tags = similarities.shape
        no_pad_num_tags = num_tags - 2  # block emission on pad
        ''' cut output to block predictions on [PAD] and 'O' label \
        (we use 0 as [PAD] label id and all we need to predict are entity type not 'O')'''
        output = similarities.narrow(-1, 2, no_pad_num_tags)
        if self.scaler:
            output = self.scaler(output, p=1, dim=-1)
        return output

    def remove_0(self, my_tensor):
        """

        """
        return my_tensor + 0.0001

#
# class ProtoWithLabelEmissionScorer(SpanClassificationBase):
#     def __init__(self, similarity_scorer: ProtoWithLabelSimilarityScorer, scaler: ScaleControllerBase = None):
#         """
#         :param similarity_scorer: Module for calculating token similarity
#         """
#         super(PrototypeSpanClassification, self).__init__(similarity_scorer, scaler)
#
#     def forward(
#             self,
#             test_reps: torch.Tensor,
#             support_reps: torch.Tensor,
#             test_output_mask: torch.Tensor,
#             support_output_mask: torch.Tensor,
#             support_targets: torch.Tensor,
#             label_reps: torch.Tensor = None, ) -> torch.Tensor:
#         """
#         :param test_reps: (batch_size, support_size, test_seq_len, dim), notice: reps has been expand to support size
#         :param support_reps: (batch_size, support_size, support_seq_len)
#         :param test_output_mask: (batch_size, test_seq_len)
#         :param support_output_mask: (batch_size, support_size, support_seq_len)
#         :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
#         :param label_reps: (batch_size, num_tags, dim)
#         :return: emission, shape: (batch_size, test_len, no_pad_num_tags)
#         """
#         similarity = self.similarity_scorer(
#             test_reps, support_reps, test_output_mask, support_output_mask, support_targets, label_reps)
#         emission = self.get_emission(similarity, support_targets)  # shape(batch_size, test_len, no_pad_num_tag)
#         return emission
#
#     def get_emission(self, similarities: torch.Tensor, support_targets: torch.Tensor):
#         """
#         :param similarities: (batch_size, support_size, test_seq_len, support_seq_len)
#         :param support_targets: one-hot label targets: (batch_size, support_size, support_seq_len, num_tags)
#         :return: emission: shape: (batch_size, test_len, no_pad_num_tags)
#         """
#         batch_size, test_len, num_tags = similarities.shape
#         no_pad_num_tags = num_tags - 1  # block emission on pad
#         ''' cut emission to block predictions on [PAD] label (we use 0 as [PAD] label id) '''
#         emission = similarities.narrow(-1, 1, no_pad_num_tags)
#         if self.scaler:
#             emission = self.scaler(emission, p=1, dim=-1)
#         return emission

def reps_dot(sent1_reps: torch.Tensor, sent2_reps: torch.Tensor) -> torch.Tensor:
    """
    calculate representation dot production
    :param sent1_reps: (N, sent1_len, reps_dim)
    :param sent2_reps: (N, sent2_len, reps_dim)
    :return: (N, sent1_len, sent2_len)
    """
    return torch.bmm(sent1_reps, torch.transpose(sent2_reps, -1, -2))  # shape: (N, seq_len1, seq_len2)


def reps_l2_sim(sent1_reps: torch.Tensor, sent2_reps: torch.Tensor) -> torch.Tensor:
    """
    calculate representation L2 similarity
    :param sent1_reps: (N, sent1_len, reps_dim)
    :param sent2_reps: (N, sent2_len, reps_dim)
    :return: (N, sent1_len, sent2_len)
    """
    expand_shape = list(sent2_reps.shape).append()
    sim = torch.dist(sent1_reps.unsqueeze(-1), sent2_reps.unsqueeze(-1).expand(), p=2)  # shape: (N, seq_len1, seq_len2)
    return -sim


def reps_cosine_sim(sent1_reps: torch.Tensor, sent2_reps: torch.Tensor) -> torch.Tensor:
    """
    calculate representation cosine similarity, note that this is different from torch version(that compute pairwisely)
    :param sent1_reps: (N, sent1_len, reps_dim)
    :param sent2_reps: (N, sent2_len, reps_dim)
    :return: (N, sent1_len, sent2_len)
    """
    dot_sim = torch.bmm(sent1_reps, torch.transpose(sent2_reps, -1, -2))  # shape: (batch, seq_len1, seq_len2)
    sent1_reps_norm = torch.norm(sent1_reps, dim=-1, keepdim=True)  # shape: (batch, seq_len1, 1)
    sent2_reps_norm = torch.norm(sent2_reps, dim=-1, keepdim=True)  # shape: (batch, seq_len2, 1)
    norm_product = torch.bmm(sent1_reps_norm,
                             torch.transpose(sent2_reps_norm, -1, -2))  # shape: (batch, seq_len1, seq_len2)
    sim_predicts = dot_sim / norm_product  # shape: (batch, seq_len1, seq_len2)
    return sim_predicts
