import torch
import torch.nn as nn
from models.modules.scale_controller import ScaleControllerBase

class SpanDetector(torch.nn.Module):# 判断是否为多词labels【0,O,T】
    def __init__(self, spanBoun_label2id, embedding_dim: int = None, emb_log=None, drop_p=0.1):
        super(SpanDetector, self).__init__()
        self.emb_log = emb_log
        self.log_content = ''
        self.tag_types = 'TO',
        self.spanBoun_label2id = spanBoun_label2id
        self.embedding_dim = embedding_dim
        self.dropout = SpatialDropout(drop_p)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, len(spanBoun_label2id))
        # self.crf = CRF(tagset_size=len(spanBoun_label2id), tag_dictionary=spanBoun_label2id)

    def forward(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_boun_target: torch.Tensor = None,) -> torch.Tensor:
        """
            :param test_reps: (batch_size, support_size, test_seq_len, dim)
            :param test_output_mask: (batch_size, test_seq_len)
            :param support_boun_target: (batch_size, support_size, support_len)

            :return: test_boun_pred
        """
        # average test representation over support set (reps for each support sent can be different)
        test_reps = torch.mean(test_reps, dim=1)  # shape (batch_size, test_seq_len, emb_dim)
        test_reps = self.layer_norm(test_reps)

        test_boun_output = self.classifier(test_reps)# shape (batch_size, test_seq_len, boun_label_size)
        no_pad_num_tags = len(self.spanBoun_label2id) - 1
        test_boun_pred = test_boun_output.narrow(-1, 1, no_pad_num_tags)
        return test_boun_pred # shape (batch_size, test_seq_len, boun_label_size)


class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class SpanRepr(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, NtokenReps):
        # shape of data_list: list(spans_size, *input_len, input_dim)
#        span_regions = [torch.cat([hidden[0], torch.mean(hidden, dim=0), hidden[-1]], dim=-1).view(1, -1)
#                       for hidden in data_list]
        span_regions = torch.mean(NtokenReps, dim=0)
#        span_regions = [torch.cat([hidden[0], hidden[-1]], dim=-1).view(1, -1)
#                       for hidden in data_list]
#         span_out = torch.cat(span_regions, dim=0)
        # regions (spans_size, input_dim)
        return span_regions