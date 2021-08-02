from tqdm import tqdm
from graphDataLoader import getDataSet
from utils_data import get_domain_dataset, spacy_seed_concepts_list  # utils的引用产生了包冲突问题
from utils_graph import unique_rows  # 不明白包引入的顺序问题
from os.path import normpath,join
import os
import numpy as np, pickle, argparse
import torch
import torch.nn.functional as F
from rgcn import RGCN
from torch_scatter import scatter_add
from torch_geometric.data import Data
import sys
import codecs
import argparse
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
Base_DIR=normpath(join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0,Base_DIR)
from data.data_process_utils.concept_util import getDomainDataURL, getReviewConceptTriples, rawTriples2index, getGraphMaps
from utils import json_util

def sample_edge_uniform(n_triples, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)


def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels


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


def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate):
    """
        Get training graph and labels with negative sampling.
    """
    edges = triplets  # 已经id化了的
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))  # 去重后的index
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype=torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype=torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype=torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data


def generate_graph(triplets, num_rels):
    """
        Get feature extraction graph without negative sampling.
    """
    edges = triplets
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

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


def getDomainJsonDataset(domain, exp_type="labeled"):
    # 获得路径
    data_file = getDomainDataURL([domain])
    if exp_type == "pos":
        reviewsJson = json_util.load(data_file[0])
        reviewsJson = reviewsJson[exp_type]
    elif exp_type == "pos":
        reviewsJson = json_util.load(data_file[0])
        reviewsJson = reviewsJson[exp_type]
    elif exp_type == "unlabeled":
        reviewsJson = json_util.load(data_file[1])
    else:
        a = "没有这个数据集"
        a = 0/0
    return reviewsJson

def addGraphFeature2ReviwsJson(model, reviewsJsons, all_seeds, maps, relation_map, unique_nodes_mapping, kg="wordNet"):
    """
        add Graph features to each sentence (document) instance json data in reviewsJsons.
    """
    x = reviewsJsons

    sent_features = np.zeros((len(x), 100))# 每篇文章的向量表示 另存方案可能用到
    maxTriple = 5000

    for j in tqdm(range(len(x)), position=0, leave=False):
        c = x[j]["concepts"]# 一篇文章中所有的单词
        n = list(set(c).intersection(set(all_seeds)))# 共有的部分 为啥c中的没有完全在all_seeds中 getAllConcept出了问题？

        try:# concept_graphs[item] 这里不应该找从大图找到的子图，应该要自己构造，应该是opinioncept + conceptNet
            rawConceptTriples = getReviewConceptTriples(x[j], kg=kg)#上面的去重操作失效了, 但是内部所如果找不到，就过滤掉的操作
            xg = rawTriples2index(rawConceptTriples, maps)

            xg = np.array(xg)# 这是在大图上的triple
            # 我们想要获得这篇review的所有tripe 包括conceptNet 和opinion
            xg = xg[~np.all(xg == 0, axis=1)]# 只要有人包含index  = 0 要过滤掉，为什么？
            # 是以大图的id为依据的
            absent1 = set(xg[:, 0]) - set(unique_nodes_mapping.values())# 没有的index
            absent2 = set(xg[:, 2]) - set(unique_nodes_mapping.values())
            absent = absent1.union(absent2)

            for item in absent:
                xg = xg[~np.any(xg == item, axis=1)]

            # xg[:, 0] = np.vectorize(unique_nodes_mapping.get)(xg[:, 0])# 拿到在小图上面的index
            # xg[:, 2] = np.vectorize(unique_nodes_mapping.get)(xg[:, 2])
            xg = unique_rows(xg).astype('int64')
            if len(xg) > maxTriple:
                xg = xg[:maxTriple, :]

            sg = generate_graph(xg, len(relation_map)).to(torch.device('cuda'))
            features = model(sg.entity, sg.edge_index, sg.edge_type, sg.edge_norm)
            x[j]["graphFeature"] = features.cpu().detach().numpy().mean(axis=0).tolist()# 作为narray存入json回有问题吗？
            torch.cuda.empty_cache()# 这里可能会是空，导致某一篇没有对应的属性， 有会报错的文档

        except ValueError:
            # 考虑是否删掉这个review
            # x.pop(j)
            pass

    # 我们考虑写回json data里面
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=50000, help='graph batch size')
    parser.add_argument('--split-size', type=float, default=0.5, help='what fraction of graph edges used in training')
    parser.add_argument('--ns', type=int, default=1, help='negative sampling ratio')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--save', type=int, default=50, help='save after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.25, help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-2, help='regularization coefficient')
    parser.add_argument('--grad-norm', type=float, default=1.0, help='grad norm')
    parser.add_argument("--data_root", default="data/domain_data/processed_data", help="data_json_dir")
    parser.add_argument("--domain", default="books",
                        help="domain name")
    parser.add_argument("--kg", default="wordNet",
                        help="knowledge graph type name")
    args = parser.parse_args()
    print(args)
    n_bases = 4
    dropout = args.dropout
    grad_norm = args.grad_norm
    domainName = args.domain
    domainList = [domainName]  # dvd
    data_root = args.data_root
    filePath = ""
    for domain in domainList:
        filePath = filePath + domain + "_"
        pass
    filePath = filePath + args.kg
    fileName = join("preprocess_data", filePath)
    all_seeds = pickle.load(open(fileName + '/all_seeds.pkl', 'rb'))
    relation_map = pickle.load(open(fileName + '/relation_map.pkl', 'rb'))
    unique_nodes_mapping = pickle.load(open(fileName + '/unique_nodes_mapping.pkl', 'rb'))  # node_index2int_id


    model = RGCN(len(unique_nodes_mapping), len(relation_map), num_bases=n_bases, dropout=dropout).cuda()
    epoch = 950
    PATH = 'weights/model_epoch' + str(epoch) + "_" + filePath +'.pt'# "_" +
    model.load_state_dict(torch.load(PATH))
    model.eval()

    maps = getGraphMaps(domainList, kg=args.kg)

    for domain in domainList: #, 'dvd', 'electronics', 'kitchen'
        print ('add graph features for', domain)
        urlList = getDomainDataURL([domain])
        for url in urlList:
            json_data = json_util.load(url)
            if "unlabeled" not in url:
                # labled 的情况
                for mode in ["pos", "neg"]:
                    reivewJsonList = json_data[mode]
                    reivewJsonList = addGraphFeature2ReviwsJson(model, reivewJsonList, all_seeds, maps, relation_map, unique_nodes_mapping,kg=args.kg)
                    json_data[mode] = reivewJsonList
                    pass
                # 保存数据
                json_util.dump(json_data, url)

            else:
                json_data = addGraphFeature2ReviwsJson(model, json_data, all_seeds, maps, relation_map, unique_nodes_mapping,kg=args.kg)
                json_util.dump(json_data, url)

    print('Done.')


def get_json_data_graph(domain="books",
                        data_root="data/domain_data/processed_data",
                        kg="wordNet",
                        epoch=50):

    n_bases = 4
    dropout = 0.25
    domainName = domain
    domainList = [domainName]  # dvd
    data_root = data_root
    filePath = ""
    for domain in domainList:
        filePath = filePath + domain + "_"
        pass
    filePath = filePath + kg
    fileName = join("preprocess_data", filePath)
    all_seeds = pickle.load(open(fileName + '/all_seeds.pkl', 'rb'))
    relation_map = pickle.load(open(fileName + '/relation_map.pkl', 'rb'))
    unique_nodes_mapping = pickle.load(open(fileName + '/unique_nodes_mapping.pkl', 'rb'))  # node_index2int_id

    model = RGCN(len(unique_nodes_mapping), len(relation_map), num_bases=n_bases, dropout=dropout).cuda()

    PATH = 'weights/model_epoch' + str(epoch) + filePath + '.pt'  # "_" +
    model.load_state_dict(torch.load(PATH))
    model.eval()

    maps = getGraphMaps(domainList)

    for domain in domainList:  # , 'dvd', 'electronics', 'kitchen'
        print('add graph features for', domain)
        urlList = getDomainDataURL([domain])
        for url in urlList:
            json_data = json_util.load(url)
            if "unlabeled" not in url:
                # labled 的情况
                for mode in ["pos", "neg"]:
                    reivewJsonList = json_data[mode]
                    reivewJsonList = addGraphFeature2ReviwsJson(model, reivewJsonList, all_seeds, maps, relation_map,
                                                                unique_nodes_mapping)
                    json_data[mode] = reivewJsonList
                    pass

                return json_data

            else:
                json_data = addGraphFeature2ReviwsJson(model, json_data, all_seeds, maps, relation_map,
                                                       unique_nodes_mapping)
                return json_data


    print('error.')
    a = 1/0
