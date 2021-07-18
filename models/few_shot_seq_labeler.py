#!/usr/bin/env python
import torch
from typing import List
from models.modules.context_embedder_base import ContextEmbedderBase
import torch
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
        :param test_token_ids: (batch_size*4*Q, test_len)
        :param test_nwp_index:  (batch_size*4*Q, test_len)
        :param test_segment_ids:  (batch_size*4*Q, test_len)
        :param test_input_mask:  (batch_size*4*Q, test_len)
        :param test_output_mask:  (batch_size*4*Q, test_len)
        :param support_token_ids:  (batch_size*3*K, test_len)
        :param support_nwp_index:  (batch_size*3*K, test_len)
        :param support_segment_ids:  (batch_size*3*K, test_len)
        :param support_input_mask:  (batch_size*3*K, test_len)
        :param support_output_mask:  (batch_size*3*K, test_len)
        :param support_label:  (batch_size*3*K, test_len)
        :param test_label:  (batch_size*4*K, test_len)
        :param epoch_id: epoch number
        :return:
        """

        # reps for tokens: (batch_size, support_size, nwp_sent_len, emb_len)
        test_reps, support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask
        )

        # calculate emission: shape(batch_size, test_len, no_pad_num_tag)
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_label)

        logits = emission

        # block pad of label_id = 0, so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_label - 1)#len = 2 :

        loss, prediction = torch.FloatTensor(0).to(test_target.device), None

        #calculate transition
        if self.transition_scorer:
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





