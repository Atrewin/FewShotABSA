# coding: utf-8
import os
import logging
import sys
import torch
import time
#!/usr/bin/env python
# coding: utf-8
import os
import logging
import sys
import torch
import copy
# my staff
from models.modules.context_embedder_base import BertContextEmbedder, \
    BertSeparateContextEmbedder, NormalContextEmbedder, BertSchemaSeparateContextEmbedder
from models.modules.span_classification_base import reps_dot, reps_l2_sim, reps_cosine_sim
from models.modules.SpanDetector import SpanDetector
from models.modules.span_classification_base import PrototypeSpanClassification
from models.modules.span_entity_labeler import SpanEntityLabeler, RuleSequenceLabeler
from models.few_shot_seq_labeler import FewShotSeqLabeler
from models.modules.scale_controller import build_scale_controller, ScaleControllerBase
from utils.device_helper import prepare_model




logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def make_scaler_args(name : str, normalizer: ScaleControllerBase, scale_r: float = None):
    ret = None
    if name == 'learn':
        ret = {'normalizer': normalizer}
    elif name == 'fix':
        ret = {'normalizer': normalizer, 'scale_rate': scale_r}
    return ret


def make_model(opt, id2label=None):
    """ Customize and build the few-shot learning model from components """

    ''' Build context_embedder '''
    if opt.context_emb == 'bert':
        context_embedder = BertContextEmbedder(opt=opt)
    elif opt.context_emb == 'sep_bert':
        context_embedder = BertSeparateContextEmbedder(opt=opt)
    elif opt.context_emb == 'elmo':
        raise NotImplementedError
    elif opt.context_emb == 'glove':
        context_embedder = NormalContextEmbedder(opt=opt, num_token=len(opt.word2id))
        context_embedder.load_embedding()
    elif opt.context_emb == 'raw':
        context_embedder = NormalContextEmbedder(opt=opt, num_token=len(opt.word2id))
    else:
        raise TypeError('wrong component type')

    '''Build emission scorer and similarity scorer '''
    # build scaler
    ems_normalizer = build_scale_controller(name=opt.emission_normalizer)
    ems_scaler = build_scale_controller(
        name=opt.emission_scaler, kwargs=make_scaler_args(opt.emission_scaler, ems_normalizer, opt.ems_scale_r))

    if opt.similarity == 'dot':
        sim_func = reps_dot
    elif opt.similarity == 'cosine':
        sim_func = reps_cosine_sim
    elif opt.similarity == 'l2':
        sim_func = reps_l2_sim
    else:
        raise TypeError('wrong component type')

    ''' Create log file to record testing data '''
    if opt.emb_log:
        emb_log = open(os.path.join(opt.output_dir, 'emb.log'), 'w')
        if id2label is not None:
            emb_log.write('id2label\t' + '\t'.join([str(k) + ':' + str(v) for k,v in id2label.items()]) + '\n')
    else:
        emb_log = None
    #@jinhui 用来算translation loss?
    span_detector = SpanDetector(opt.boundary_label2id, opt.emb_dim, opt.emb_log)

    if opt.emission == 'proto':
        span_cls = PrototypeSpanClassification(sim_func=sim_func, scaler=ems_scaler, emb_log=emb_log)
    else:
        raise TypeError('wrong component type')

    # if opt.emission == 'mnet':
    #     similarity_scorer = MatchingSimilarityScorer(sim_func=sim_func, emb_log=emb_log)
    #     emission_scorer = MNetEmissionScorer(similarity_scorer, ems_scaler, opt.div_by_tag_num)
    # elif opt.emission == 'proto':
    #     similarity_scorer = PrototypeSimilarityScorer(sim_func=sim_func, emb_log=emb_log)
    #     emission_scorer = PrototypeEmissionScorer(similarity_scorer, ems_scaler)
    # elif opt.emission == 'proto_with_label':
    #     similarity_scorer = ProtoWithLabelSimilarityScorer(sim_func=sim_func, scaler=opt.ple_scale_r, emb_log=emb_log)
    #     emission_scorer = ProtoWithLabelEmissionScorer(similarity_scorer, ems_scaler)
    # else:
    #     raise TypeError('wrong component type')

    ''' Build decoder '''
    if opt.decoder == 'sms':
        decoder = SpanEntityLabeler()
    elif opt.decoder == 'rule':
        decoder = RuleSequenceLabeler(id2label)
    else:
        raise TypeError('wrong component type')

    ''' Build the whole model '''
    seq_labeler = FewShotSeqLabeler
    model = seq_labeler(
        opt=opt,
        context_embedder=context_embedder,
        span_detector=span_detector,
        span_cls=span_cls,
        decoder=decoder,
        emb_log=emb_log
    )
    return model


def load_model(path):
    try:
        with open(path, 'rb') as reader:
            cpt = torch.load(reader, map_location='cpu')
            model = make_model(opt=cpt['opt'], config=cpt['config'])
            model = prepare_model(args=cpt['opt'], model=model, device=cpt['opt'].device, n_gpu=cpt['opt'].n_gpu)
            model.load_state_dict(cpt['state_dict'])
            return model
    except IOError as e:
        logger.info("Failed to load model from {} \n {}".format(path, e))
        return None


def get_value_from_order_dict(order_dict, key):
    """"""
    for k, v in order_dict.items():
        if key in k:
            return v
    return []

