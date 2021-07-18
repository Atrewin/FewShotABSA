#!/usr/bin/env python
import torch
from typing import Tuple, Union, List
from allennlp.nn.util import masked_log_softmax


class SpanEntityLabeler(torch.nn.Module):
    def __init__(self):
        super(SpanEntityLabeler, self).__init__()

    def forward(self,
                boun_logits: torch.Tensor,
                type_logits: torch.Tensor,
                boun_mask: torch.Tensor,
                type_mask: torch.Tensor,
                boun_tags: torch.Tensor,
                type_tags: torch.Tensor) -> Tuple[Union[None, torch.Tensor],
                                             Union[None, torch.Tensor]]:
        """

        :param boun_logits: (batch_size, seq_len, n_tags)
        :param type_logits: (batch_size, seq_len, n_tags)
        :param boun_mask: (batch_size, seq_len)
        :param type_mask: (batch_size, seq_len)
        :param boun_tags: (batch_size, seq_len)
        :param type_tags: (batch_size, seq_len)
        :return:
        """
        return self._compute_loss(boun_logits, type_logits, boun_mask,
                                  type_mask, boun_tags, type_tags)

    def _compute_loss(self,
                      boun_logits: torch.Tensor,
                      type_logits: torch.Tensor,
                      boun_mask: torch.Tensor,
                      type_mask: torch.Tensor,
                      boun_tags: torch.Tensor,
                      type_tags: torch.Tensor) -> torch.Tensor:
        """

        :param logits:
        :param mask:
        :param targets:
        :return:
        """

        batch_size, seq_len = boun_mask.shape

        type_mask = [0 if tag in [0, 1] else 1 for tag in type_mask.view(-1).tolist()]
        type_mask = torch.tensor(type_mask).to(boun_logits.device).view(batch_size, -1)

        normalised_boun_logits = masked_log_softmax(boun_logits, boun_mask.unsqueeze(-1), dim=-1)
        normalised_boun_logits = normalised_boun_logits.gather(dim=-1, index=boun_tags.unsqueeze(-1))
        boun_loss = normalised_boun_logits.view(-1).masked_select(boun_mask.view(-1).bool())

        normalised_type_logits = masked_log_softmax(type_logits, type_mask.unsqueeze(-1), dim=-1)
        normalised_type_logits = normalised_type_logits.gather(dim=-1, index=type_tags.unsqueeze(-1))
        type_loss = normalised_type_logits.view(-1).masked_select(type_mask.view(-1).bool())

        return -1 * (boun_loss.mean() + type_loss.mean())

    def decode(self, boun_logits: torch.Tensor, type_logits: torch.Tensor, masks: torch.Tensor,
               spanBoun_label2id: dict, type_target: torch.Tensor=None) -> (List[List[int]], List[List[int]]):

        # boun_preds : shape(batch_size, test_seq_len)
        boun_preds = boun_logits.argmax(dim=-1)
        # boun_preds : shape(batch_size, test_seq_len)
        type_preds = type_logits.argmax(dim=-1)

        batch_size, test_seq_len = boun_preds.shape
        for each_boun_pred, each_type_pred in zip(boun_preds, type_preds):
            for start in range(0, test_seq_len - 1):
                # Only the entity type of first token are true before,
                # Change the label from the second as same as first token type
                # Determine whether the boundary label is B.
                # we need to add 1 because of we narrow the tensor before
                if each_boun_pred[start] == spanBoun_label2id['B'] - 1:
                    for end in range(start+1, test_seq_len):
                        if each_boun_pred[end] == spanBoun_label2id['I'] - 1:
                            each_type_pred[end] = each_type_pred[start]
                        else:
                            break

        if type_target is not None:
            type_preds = type_target

        return self.remove_pad(boun_preds=boun_preds,
                               type_preds=type_preds,
                               masks=masks)

    def remove_pad(self, boun_preds: torch.Tensor, type_preds: torch.Tensor, masks: torch.Tensor) -> (List[List[int]], List[List[int]]):
        # remove predict result for padded token
        ret_boun = []
        ret_type = []
        for boun_pred, type_pred, mask in zip(boun_preds, type_preds, masks):
            boun_temp = []
            type_temp = []
            for boun_l_id, type_l_id, mk in zip(boun_pred, type_pred, mask):
                if mk:
                    boun_temp.append(int(boun_l_id))
                    type_temp.append(int(type_l_id))
            ret_boun.append(boun_temp)
            ret_type.append(type_temp)
        return ret_boun, ret_type


class RuleSequenceLabeler(SpanEntityLabeler):
    def __init__(self, id2label):
        super(RuleSequenceLabeler, self).__init__()
        self.id2label = id2label

    def forward(self,
                logits: torch.Tensor,
                mask: torch.Tensor,
                tags: torch.Tensor) -> Tuple[Union[None, torch.Tensor],
                                             Union[None, torch.Tensor]]:
        """

        :param logits: (batch_size, seq_len, n_tags)
        :param mask: (batch_size, seq_len)
        :param tags: (batch_size, seq_len)
        :return:
        """
        return self._compute_loss(logits, mask, tags)

    def decode(self, logits: torch.Tensor, masks: torch.Tensor) -> List[List[int]]:
        preds = self.get_masked_preds(logits)
        return self.remove_pad(preds=preds, masks=masks)

    def get_masked_preds(self, logits):
        preds = []
        for logit in logits:
            pred_mask = torch.ones(logits.shape[-1]).to(logits.device)  # init mask for a sentence
            pred = []
            for token_logit in logit:
                token_pred = self.get_one_step_pred(token_logit, pred_mask)
                pred_mask = self.get_pred_mask(token_pred).to(logits.device)
                pred.append(token_pred)
            preds.append(pred)
        return preds

    def get_one_step_pred(self, token_logit, pred_mask):
        masked_logit = token_logit * pred_mask
        return masked_logit.argmax(dim=-1)

    def get_pred_mask(self, current_pred):
        mask = [1] * len(self.id2label)  # not here exclude [pad] label
        label_now = self.id2label[current_pred.item() + 1]  # add back [pad]
        if label_now == 'O':
            for ind, label in self.id2label.items():
                if 'I-' in label:
                    mask[ind] = 0
        elif label_now == '[PAD]':
            mask = [0] * len(self.id2label)
        elif 'B-' in label_now:
            for ind, label in self.id2label.items():
                if 'I-' in label and label.replace('I-', '') != label_now.replace('B-', ''):
                    mask[ind] = 0
        elif 'I-' in label_now:
            for ind, label in self.id2label.items():
                if 'I-' in label and label.replace('I-', '') != label_now.replace('I-', ''):
                    mask[ind] = 0
        else:
            raise ValueError('Wrong label {}'.format(label_now))
        mask = torch.FloatTensor(mask[1:])  # exclude [pad] label
        return mask

    def remove_pad(self, preds: torch.Tensor, masks: torch.Tensor) -> List[List[int]]:
        # remove predict result for padded token
        ret = []
        for pred, mask in zip(preds, masks):
            temp = []
            for l_id, mk in zip(pred, mask):
                if mk:
                    temp.append(int(l_id))
            ret.append(temp)
        return ret


def unit_test():
    logits = torch.tensor([[[0.1, 0.2, 0.5, 0.7, 0.3], [1.2, 0.8, 0.5, 0.6, 0.1], [0.4, 0.5, 0.5, 0.9, 1.2]],
                           [[1.9, 0.3, 0.5, 0.2, 0.3], [0.2, 0.1, 0.5, 0.4, 0.1], [0.4, 0.5, 0.1, 0.1, 0.2]]])
    labels = ['[PAD]', 'O', 'B-x', 'I-x', 'B-y', 'I-y']
    id2label = dict(enumerate(labels))
    mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    a = RuleSequenceLabeler(id2label)
    print(a.decode(logits, mask))
    a = SpanEntityLabeler()
    print(a.decode(logits, mask))


if __name__ == '__main__':
    unit_test()
