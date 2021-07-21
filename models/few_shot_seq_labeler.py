#!/usr/bin/env python
import torch
from typing import List
from models.modules.context_embedder_base import ContextEmbedderBase
import torch
import torchcrf
from typing import List

from models.modules.emission_scorer_base import EmissionScorerBase
from models.modules.transition_scorer import TransitionScorerBase
from models.modules.seq_labeler import SequenceLabeler
from models.modules.conditional_random_field import ConditionalRandomField


class FewShotSeqLabeler(torch.nn.Module):
    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 emission_scorer: EmissionScorerBase,
                 decoder: torch.nn.Module,
                 transition_scorer: TransitionScorerBase = None,
                 emb_log: str = None):
        super(FewShotSeqLabeler, self).__init__()
        self.opt = opt
        self.context_embedder = context_embedder
        self.emission_scorer = emission_scorer
        self.transition_scorer = transition_scorer
        self.decoder = decoder
        self.no_embedder_grad = opt.no_embedder_grad
        self.label_mask = None
        self.emb_log = emb_log

    # @pysnooper.snoop()+
    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_input_mask: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_input_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            support_label: torch.Tensor,
            test_label: torch.Tensor,
            epoch_id=0,
    ):
        """
        :param test_token_ids: (batch_size, test_len)  #@ (batch_size, query_size, test_len)
        :param test_segment_ids: (batch_size, test_len) #@ (batch_size, query_size, test_len)
        :param test_nwp_index: (batch_size, test_len)   #@ (batch_size, query_size, test_len)
        :param test_input_mask: (batch_size, test_len)    #@ (batch_size, query_size, test_len)
        :param test_output_mask: (batch_size, test_len)   #@ (batch_size, query_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, test_len)   # (batch_size, query_size, test_len)
        :param support_target: one-hot targets (batch_size, support_size, support_len, num_tags) # (batch_size, support_size, test_len)
        :param support_num: (batch_size, 1)
        :return:
        """

        # reps for tokens: (batch_size, support_size, nwp_sent_len, emb_len)
        test_reps, support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask
        )
        #trans to one hot label (batch_size, support_szie, support_len, num_tag)
        support_label = self.to_one_hot_label(support_label, self.opt.num_tags)
        # calculate emission: shape(batch_size, test_len, no_pad_num_tag)
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_label)

        logits = emission

        # block pad of label_id = 0, so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_label - 1)#len = 2 :

        loss, prediction = torch.FloatTensor(0).to(test_target.device), None

        #calculate transition
        if self.opt.decoder == "crf-specific":
            transitions, start_transitions, end_transitions = self.transition_scorer(test_reps, support_label)

            if self.label_mask is not None:
                transitions = self.mask_transition(transitions, self.label_mask)

            self.decoder: ConditionalRandomField
            if self.training:
                # the CRF staff
                llh = self.decoder.forward(
                    inputs=logits,
                    transitions=transitions,
                    start_transitions=start_transitions,
                    end_transitions=end_transitions,
                    tags=test_target,
                    mask=test_output_mask)
                loss = -1 * llh
            else:
                best_paths = self.decoder.viterbi_tags(logits=logits,
                                                       transitions_without_constrain=transitions,
                                                       start_transitions=start_transitions,
                                                       end_transitions=end_transitions,
                                                       mask=test_output_mask)
                # split path and score
                prediction, path_score = zip(*best_paths)
                # we block pad label(id=0) before by - 1, here, we add 1 back
                prediction = self.add_back_pad_label(prediction)
        elif self.opt.decoder == "crf":
            self.decoder: torchcrf.CRF

            # shape to API
            # emission size is (seq_length, batch_size, num_tags)
            # test_target size is (seq_length, batch_size)
            # mask size is (seq_length, batch_size) as booleans
            # notes: batch_size = batch_size * query_size
            query_shape = list(test_target.shape)
            emission = emission.transpose(0, 1)
            test_target = test_target.reshape(-1, query_shape[-1]).transpose(0, 1)
            test_output_mask = test_output_mask.reshape(-1, query_shape[-1]).transpose(0, 1) > 0

            if self.training:
                # the CRF staff
                llh = self.decoder(emission, test_target, mask=test_output_mask) # emission: (seq_length, batch_size, num_tags); test_target: (seq_length, batch_size)
                loss = -1 * llh
            else:
                # split path and score
                prediction = self.decoder.decode(emission)
                # we block pad label(id=0) before by - 1, here, we add 1 back
                prediction = self.add_back_pad_label(prediction)
            pass
        else:
            # sms
            self.decoder: SequenceLabeler
            if self.training:
                loss = self.decoder.forward(logits=logits,
                                            tags=test_target,
                                            mask=test_output_mask)
            else:
                prediction = self.decoder.decode(logits=logits, masks=test_output_mask)
                # we block pad label(id=0) before by - 1, here, we add 1 back
                prediction = self.add_back_pad_label(prediction)


        if self.training:
            return loss
        else:
            return prediction

        # return boun_predictions, type_predictions
        return logits

    def get_context_reps(
        self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
    ):
        if self.no_embedder_grad:
            self.context_embedder.eval()  # to avoid the dropout effect of reps model
            self.context_embedder.requires_grad = False
        else:
            self.context_embedder.train()  # to avoid the dropout effect of reps model
            self.context_embedder.requires_grad = True
        test_reps, support_reps = self.context_embedder(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask
        )
        if self.no_embedder_grad:
            test_reps = test_reps.detach()  # detach the reps part from graph
            support_reps = support_reps.detach()  # detach the reps part from graph
        return test_reps, support_reps

    def add_back_pad_label(self, predictions: List[List[int]]):
        for pred in predictions:
            for ind, l_id in enumerate(pred):
                pred[ind] += 1  # pad token is in the first place
        return predictions

    def mask_transition(self, transitions, label_mask):
        # transitions = transitions + 100
        trans_mask = label_mask[1:, 1:].float()  # block pad label(at 0) here
        transitions = transitions * trans_mask
        # transitions = transitions - 100
        return transitions

    def to_one_hot_label(self, label, N):
        """
        :param label: (batch_size, support_size, test_len)
        :return one_hot_label: one-hot targets (batch_size, support_size, support_len, num_tags)# https://blog.csdn.net/qq_34914551/article/details/88700334
        """

        size = list(label.size())
        label = label.view(-1)  # reshape 为向量
        one_hot_label = torch.sparse.torch.eye(N).to(self.opt.device)
        one_hot_label = one_hot_label.index_select(0, label)  # 用上面的办法转为换one hot
        size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
        return one_hot_label.view(*size)


class SchemaFewShotSeqLabeler(FewShotSeqLabeler):
    def __init__(
            self,
            opt,
            context_embedder: ContextEmbedderBase,
            emission_scorer: EmissionScorerBase,
            decoder: torch.nn.Module,
            transition_scorer: TransitionScorerBase = None,
            emb_log: str = None
    ):
        super(SchemaFewShotSeqLabeler, self).__init__(
            opt, context_embedder, emission_scorer, decoder, transition_scorer, emb_log)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            support_num: torch.Tensor,
            label_token_ids: torch.Tensor = None,
            label_segment_ids: torch.Tensor = None,
            label_nwp_index: torch.Tensor = None,
            label_input_mask: torch.Tensor = None,
            label_output_mask: torch.Tensor = None,
    ):
        """
        few-shot sequence labeler using schema information
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, test_len)
        :param support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :param label_token_ids:
            if label_reps=cat:
                (batch_size, label_num * label_des_len)
            elif:
                (batch_size, label_num, label_des_len)
        :param label_segment_ids: same to label token ids
        :param label_nwp_index: same to label token ids
        :param label_input_mask: same to label token ids
        :param label_output_mask: same to label token ids
        :return:
        """
        test_reps, support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask,
            support_token_ids, support_segment_ids, support_nwp_index, support_input_mask
        )
        # get label reps, shape (batch_size, max_label_num, emb_dim)
        label_reps = self.get_label_reps(
            label_token_ids, label_segment_ids, label_nwp_index, label_input_mask,
        )

        # calculate emission: shape(batch_size, test_len, no_pad_num_tag)
        # todo: Design new emission here
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_target,
                                        label_reps)
        if not self.training and self.emb_log:
            self.emb_log.write('\n'.join(['test_target\t' + '\t'.join(map(str, one_target))
                                          for one_target in test_target.tolist()]) + '\n')

        logits = emission

        # block pad of label_id = 0, so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_target - 1)

        loss, prediction = torch.FloatTensor([0]).to(test_target.device), None
        # todo: Design new transition here
        if self.transition_scorer:
            transitions, start_transitions, end_transitions = self.transition_scorer(test_reps, support_target, label_reps[0])

            if self.label_mask is not None:
                transitions = self.mask_transition(transitions, self.label_mask)

            self.decoder: ConditionalRandomField
            if self.training:
                # the CRF staff
                llh = self.decoder.forward(
                    inputs=logits,
                    transitions=transitions,
                    start_transitions=start_transitions,
                    end_transitions=end_transitions,
                    tags=test_target,
                    mask=test_output_mask)
                loss = -1 * llh
            else:
                best_paths = self.decoder.viterbi_tags(logits=logits,
                                                       transitions_without_constrain=transitions,
                                                       start_transitions=start_transitions,
                                                       end_transitions=end_transitions,
                                                       mask=test_output_mask)
                # split path and score
                prediction, path_score = zip(*best_paths)
                # we block pad label(id=0) before by - 1, here, we add 1 back
                prediction = self.add_back_pad_label(prediction)
        else:
            self.decoder: SequenceLabeler
            if self.training:
                loss = self.decoder.forward(logits=logits,
                                            tags=test_target,
                                            mask=test_output_mask)
            else:
                prediction = self.decoder.decode(logits=logits, masks=test_output_mask)
                # we block pad label(id=0) before by - 1, here, we add 1 back
                prediction = self.add_back_pad_label(prediction)

        if self.training:
            return loss
        else:
            return prediction

    def get_label_reps(
            self,
            label_token_ids: torch.Tensor,
            label_segment_ids: torch.Tensor,
            label_nwp_index: torch.Tensor,
            label_input_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param label_token_ids:
        :param label_segment_ids:
        :param label_nwp_index:
        :param label_input_mask:
        :return:  shape (batch_size, label_num, label_des_len)
        """
        return self.context_embedder(
            label_token_ids, label_segment_ids, label_nwp_index, label_input_mask,  reps_type='label')



