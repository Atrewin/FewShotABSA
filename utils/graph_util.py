import  time

import os,sys
from os.path import normpath,join,dirname
# print(__file__)#获取的是相对路径
# print(os.path.abspath(__file__))#获得的是绝对路径
# print(os.path.dirname(os.path.abspath(__file__)))#获得目录的绝对路径
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#获得的是Test的绝对路径
import numpy as np, pickle, argparse
from os.path import normpath,join,dirname
import torch
Base_DIR=normpath(join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0,Base_DIR)#添加环境变量，因为append是从列表最后开始添加路径，可能前面路径有重复，最好用sys.path.insert(Base_DIR)从列表最前面开始添加

from os.path import normpath,join
import os
import numpy as np, pickle, argparse
import torch
import torch.nn.functional as F
from utils.path_util import from_project_root

from torch_scatter import scatter_add
from torch_geometric.data import Data
import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
Base_DIR=normpath(join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0,Base_DIR)



def getGraphMaps(maps_root): #@jinhui 0731
    # 返回的是word2id的映射， 这些映射是之前在预训练的时候已经确定的，这里直接load就来

    maps_root = from_project_root(maps_root)
    relation_map = pickle.load(open(maps_root + '/relation_map.pkl', 'rb'))  # 文件目录 @jinhui 目前是硬绑定
    unique_nodes_mapping = pickle.load(open(maps_root + '/unique_nodes_mapping.pkl', 'rb'))  # 被reviewconcept过滤过的
    concept_map = pickle.load(open(maps_root + '/concept_map.pkl', 'rb'))  # 来自总的大图

    return relation_map, concept_map, unique_nodes_mapping

def getReviewConceptTriples(reviewJson):#@jinhui 未实现
    # 获取到所有的三元组
    # reviewConceptNetTriples = reviewJson["conceptNetTriples"] #rawdata 中没有拿到对应的conceptNetTriples @jinhui 未实现

    reviewConceptNetTriples = [
                [
                    "full",
                    "nsubj",
                    "it"
                ],
                [
                    "interested",
                    "advcl",
                    "reproduce"
                ],
                [
                    "interested",
                    "mark",
                    "if"
                ],
                [
                    "guide",
                    "amod",
                    "better"
                ],
                [
                    "anything",
                    "amod",
                    "close"
                ],
                [
                    "find",
                    "advcl",
                    "interested"
                ],
                [
                    "find",
                    "conj",
                    "full"
                ],
                [
                    "weather",
                    "amod",
                    "inclement"
                ],
                [
                    "close",
                    "obl",
                    "it"
                ],
                [
                    "close",
                    "advcl",
                    "intend"
                ],
                [
                    "full",
                    "obl",
                    "photograph"
                ],
                [
                    "addition",
                    "amod",
                    "valuable"
                ],
                [
                    "costume",
                    "amod",
                    "authentic"
                ],
                [
                    "interested",
                    "cop",
                    "be"
                ],
                [
                    "full",
                    "cop",
                    "be"
                ],
                [
                    "interested",
                    "nsubj",
                    "you"
                ]
            ]

    return reviewConceptNetTriples

def rawTriples2index(rawConceptTriples, maps):

    relation_map = maps[0]  # 文件目录 @jinhui 目前是硬绑定
    concept_map = maps[1]  # 来自总的大图
    unique_nodes_mapping = maps[2]  # 被reviewconcept过滤过的

    reviewTriples = []
    for triple in rawConceptTriples:  # 这个步骤也很慢，推荐之后转换好直接读取
        try:
            # 到word2int
            srcMap = concept_map[triple[0]]
            relMap = relation_map[triple[1]]
            distMap = concept_map[triple[2]]
            #  到int2node_index # 存在数组越界的情况unique_nodes_mapping：8090 concept_map：118651# 实际上是数据不一致的问题

            srcMap, distMap = unique_nodes_mapping[srcMap], unique_nodes_mapping[distMap]
        except:
            a = 0  # 实际上是数据不一致的问题 主要是前后数据没有连起来，导致字典为空的查询
            continue #continue 的原因是并不能够保证所有的world都可以在图上有id(节点)，尤其是在test的时候
        triple = [srcMap, relMap, distMap]
        reviewTriples.append(triple)

    return reviewTriples  # 返回该review




    return reviewTriples  # 返回该review


def unique_rows(a):
    """
    Drops duplicate rows from a numpy 2d array
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def generate_graph(triplets, num_rels):
    """
        Get feature extraction graph without negative sampling.
    """
    edges = triplets
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    # relabeled_edges = np.stack((src, rel, dst)).transpose()

    src = torch.tensor(src, dtype=torch.long).contiguous()
    dst = torch.tensor(dst, dtype=torch.long).contiguous()
    rel = torch.tensor(rel, dtype=torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)

    return data

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    """
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    """
    one_hot = F.one_hot(edge_type, num_classes=2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm



def tokens2unique_nodes(reviewJson, maps):
    # 需要讲 token 转化为 token_graph_ids 以便于获取到和整个图等长的 graph embedding
    # maps:  word2graph id
    # item: a sample for raw review json

    entity = []

    # word2id
    relation_map = maps[0]  # 文件目录 @jinhui 目前是硬绑定
    concept_map = maps[1]  # 来自总的大图
    unique_nodes_mapping = maps[2]  # 被reviewconcept过滤过的

    for token in reviewJson["text"]:  # 这个步骤也很慢，推荐之后转换好直接读取
        try:
            # 到word2domain_graph_int
            graph_big_index_map = concept_map[token]
            graph_domain_index_map = unique_nodes_mapping[graph_big_index_map]
        except:
            graph_domain_index_map = 0

        entity.append(graph_domain_index_map)

    return entity


