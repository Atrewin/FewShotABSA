#!/usr/bin/env python
# coding: utf-8
import os
import logging
import sys
import torch
import copy
import torchcrf
import torch.nn as nn
# my staff
from models.modules.context_embedder_base import BertContextEmbedder, \
    BertSeparateContextEmbedder, NormalContextEmbedder, BertGraphSeparateContextEmbedder, BertCatGraphContextEmbedder
from models.modules.similarity_scorer_base import MatchingSimilarityScorer, \
    PrototypeSimilarityScorer, ProtoWithLabelSimilarityScorer, TapNetSimilarityScorer, \
    reps_dot, reps_l2_sim, reps_cosine_sim
from models.modules.emission_scorer_base import MNetEmissionScorer, \
    PrototypeEmissionScorer, ProtoWithLabelEmissionScorer, TapNetEmissionScorer
from models.modules.transition_scorer import FewShotTransitionScorer, FewShotTransitionScorerFromLabel
from models.modules.seq_labeler import SequenceLabeler, RuleSequenceLabeler
from models.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from models.few_shot_seq_labeler import FewShotSeqLabeler, SchemaFewShotSeqLabeler
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


def make_model(opt, num_tags, trans_r, id2label=None, random_num_tags=None):
    """ Customize and build the few-shot learning model from components """



    ''' Build context_embedder '''
    if opt.context_emb == 'bert':
        context_embedder =  BertContextEmbedder(opt=opt)
    elif opt.context_emb == 'sep_bert':
        context_embedder = BertSeparateContextEmbedder(opt=opt)
    elif opt.context_emb == 'elmo':
        raise NotImplementedError
    elif opt.context_emb == 'glove':
        context_embedder = NormalContextEmbedder(opt=opt, num_token=len(opt.word2id))
        context_embedder.load_embedding()
    elif opt.context_emb == 'raw':
        context_embedder = NormalContextEmbedder(opt=opt, num_token=len(opt.word2id))

    elif opt.context_emb == 'bert_graph':
        context_embedder = BertCatGraphContextEmbedder(opt=opt)#@jinhui 0731
        pass
    elif opt.context_emb == 'sep_bert_graph':
        context_embedder = BertGraphSeparateContextEmbedder(opt=opt)#@jinhui 0731
        pass
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

    if opt.emission == 'mnet':
        similarity_scorer = MatchingSimilarityScorer(sim_func=sim_func, emb_log=emb_log)
        emission_scorer = MNetEmissionScorer(similarity_scorer, ems_scaler, opt.div_by_tag_num)
    elif opt.emission == 'proto':
        similarity_scorer = PrototypeSimilarityScorer(sim_func=sim_func, emb_log=emb_log)
        emission_scorer = PrototypeEmissionScorer(similarity_scorer, ems_scaler)
    elif opt.emission == 'proto_with_label':
        similarity_scorer = ProtoWithLabelSimilarityScorer(sim_func=sim_func, scaler=opt.ple_scale_r, emb_log=emb_log)
        emission_scorer = ProtoWithLabelEmissionScorer(similarity_scorer, ems_scaler)
    elif opt.emission == 'tapnet':
        if not random_num_tags:
            random_num_tags = num_tags
        similarity_scorer = TapNetSimilarityScorer(
            sim_func=sim_func, num_tags=random_num_tags, mlp_out_dim=opt.tap_mlp_out_dim,
            random_init=opt.tap_random_init, random_init_r=opt.tap_random_init_r,
            mlp=opt.tap_mlp, emb_log=emb_log, tap_proto=opt.tap_proto,
            tap_proto_r=opt.tap_proto_r, proj_dim=opt.proj_dim)
        emission_scorer = TapNetEmissionScorer(similarity_scorer, ems_scaler)
    else:
        raise TypeError('wrong component type')

    ''' Build decoder '''

    if opt.decoder == 'sms':
        transition_scorer = None
        decoder = SequenceLabeler()
    elif opt.decoder == 'rule':
        transition_scorer = None
        decoder = RuleSequenceLabeler(id2label)

    elif opt.decoder == "crf":
        transition_scorer = 1  # FSTagging 之前的设计，我们用不到，但是会作为后面执行分支的判断， 所以设为1
        decoder = torchcrf.CRF(num_tags-1)# remove pad
    elif opt.decoder == 'crf-specific':
        # Notice: only train back-off now
        if opt.train_trans_mat == None:
            raise ValueError('the tran_mat is not defined')
        trans_normalizer = build_scale_controller(name=opt.trans_normalizer)
        trans_scaler = build_scale_controller(
            name=opt.trans_scaler, kwargs=make_scaler_args(opt.trans_scaler, trans_normalizer, opt.trans_scale_r))
        if opt.transition == 'learn':
            transition_scorer = FewShotTransitionScorer(
                num_tags=num_tags, normalizer=trans_normalizer, scaler=trans_scaler,
                r=trans_r, backoff_init=opt.backoff_init, trans_mat=opt.train_trans_mat)
        elif opt.transition == 'learn_with_label':
            label_trans_normalizer = build_scale_controller(name=opt.label_trans_normalizer)
            label_trans_scaler = build_scale_controller(
                name=opt.label_trans_scaler, kwargs=make_scaler_args(opt.label_trans_scaler, label_trans_normalizer, opt.label_trans_scale_r))
            transition_scorer = FewShotTransitionScorerFromLabel(
                num_tags=num_tags, normalizer=trans_normalizer, scaler=trans_scaler,
                r=trans_r, backoff_init=opt.backoff_init, label_scaler=label_trans_scaler)
        if opt.add_transition_rules and id2label:
            non_pad_id2label = copy.deepcopy(id2label).__delitem__(0)  # we 0 as [PAD] label id, here remove it.
            for k, v in non_pad_id2label.items():
                non_pad_id2label[k] = v - 1  # we 0 as [PAD] label id, here remove it.
            constraints = allowed_transitions(constraint_type='BIO', labels=non_pad_id2label)
        else:
            constraints = None
        decoder = ConditionalRandomField(num_tags=transition_scorer.num_tags, constraints=constraints)  # accurate tags
    else:
        raise TypeError('wrong component type')

    ''' Build the whole model '''
    seq_labeler = SchemaFewShotSeqLabeler if opt.use_schema else FewShotSeqLabeler
    model = seq_labeler(
        opt=opt,
        context_embedder=context_embedder,
        emission_scorer=emission_scorer,
        decoder=decoder,
        transition_scorer=transition_scorer,
        emb_log=emb_log
    )
    return model


def load_model(path):
    try:
        with open(path, 'rb') as reader:
            cpt = torch.load(reader, map_location='cpu')
            old_num_tags = len(get_value_from_order_dict(cpt['state_dict'], 'label_reps'))
            model = make_model(opt=cpt['opt'], num_tags=cpt['num_tags'], trans_r=cpt['trans_r'],
                               random_num_tags=old_num_tags // 3)
            model = prepare_model(args=cpt['opt'], model=model, device=cpt['opt'].device, n_gpu=cpt['opt'].n_gpu)
            model.load_state_dict(cpt['state_dict'])
            return model
    except IOError:
        logger.info("Failed to load model, sleeping ...")
        return None


def get_value_from_order_dict(order_dict, key):
    for k, v in order_dict.items():
        if key in k:
            return v
    return []


def get_linear_layer(layer_size_list):
    #jinhui 未实现 可复用， 直接针对任务配置
    layer = nn.Linear(layer_size_list[0], layer_size_list[-1])

    return layer