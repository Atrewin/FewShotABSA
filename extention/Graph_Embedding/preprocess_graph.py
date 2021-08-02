from tqdm import tqdm
import numpy as np
import os.path, pickle
import os,sys
from os.path import normpath,join,dirname
from utils_graph import conceptnet_graph, domain_aggregated_graph, subgraph_for_concept
Base_DIR=normpath(join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0,Base_DIR)
from data.data_process_utils.concept_util import getDomainDataURL, getAllConcepts
import argparse

import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
# 没有起到切换encoding的效果，变成在open（）上改


if __name__ == '__main__':
    # 目标是拿到各种map
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/domain_data/processed_data",
                        help="data_json_dir")
    parser.add_argument("--domain", default="books",
                        help="domain name")
    parser.add_argument("--kg", default="wordNet",
                        help="knowledge graph type name")
    opt = parser.parse_args()

    print ('Extracting seed concepts from ' + opt.domain)

    domainList = [opt.domain]# , "books", "dvd"，"electronics", "kitchen" 用于控制从大图中抽取什么样的节点作为

    urlList = getDomainDataURL(domainList,opt.data_path)
    all_seeds = getAllConcepts(urlList)  # 有接近两万个，而concept每小时最多拿到3600个

    if opt.kg == "conceptNet":
        print ('Creating conceptNet graph.')#是独立的超参数大图图，应该来源于ConceptNet + opinionconceptTriple：118651
        G, G_reverse, concept_map, relation_map = conceptnet_graph('conceptnet_english_ours.txt')# 上面应该加入我们的节点
    elif opt.kg == "wordNet":
        print('Creating wordNet graph.')  # 是独立的超参数大图图，应该来源于ConceptNet + opinionconceptTriple：118651
        G, G_reverse, concept_map, relation_map = conceptnet_graph('wordnet_aspect2opinion_english_ours.txt')  # 上面应该加入我们的节点
    else:
        print('no such a graph triples .txt.')
        exit()
    # concept_map单纯的word2int
    print ('Num seed concepts:', len(all_seeds))
    print ('Populating domain aggregated sub-graph with seed concept sub-graphs.')
    triplets, unique_nodes_mapping = domain_aggregated_graph(all_seeds, G, G_reverse, concept_map, relation_map)# @jinhui 这边是全部dataset的吗？这个只是做为一个快速找到邻居的工具
    # unique_nodes_mapping 80908 其实是过滤掉一些all_seeds无关的，将index缩短 triplets中的是压缩过的 {conceptMap：coutureindex}
    print ('Creating sub-graph for seed concepts.')
    concept_graphs = {}# 每个seed 一个graph是为什么？在总图中把相关的东西都筛选出来

    # for node in tqdm(all_seeds, desc='Instance', position=0):
    #     concept_graphs[node] = subgraph_for_concept(node, G, G_reverse, concept_map, relation_map)# concept_map relation_map来表示的

    # Create mappings
    inv_concept_map = {v: k for k, v in concept_map.items()}
    inv_unique_nodes_mapping = {v: k for k, v in unique_nodes_mapping.items()}
    inv_word_index = {}
    for item in inv_unique_nodes_mapping:
        inv_word_index[item] = inv_concept_map[inv_unique_nodes_mapping[item]]# index 是 小图的
    word_index = {v: k for k, v in inv_word_index.items()}
    inv_relation_map = {v: k for k, v in relation_map.items()}
    print ('Saving files.')
    fileName = ""
    for domain in domainList:
        fileName = fileName + domain + "_"
        pass
    fileName = join("preprocess_data", fileName) + opt.kg
    if not os.path.exists(fileName):
        os.makedirs(fileName)
    pickle.dump(all_seeds, open(fileName + '/all_seeds.pkl', 'wb'))
    pickle.dump(concept_map, open(fileName + '/concept_map.pkl', 'wb'))# 在大图上的index
    pickle.dump(relation_map, open(fileName + '/relation_map.pkl', 'wb'))
    pickle.dump(unique_nodes_mapping, open(fileName + '/unique_nodes_mapping.pkl', 'wb'))# 大图index到小图index
    pickle.dump(word_index, open(fileName + '/word_index.pkl', 'wb'))
    # pickle.dump(concept_graphs, open(fileName + '/concept_graphs.pkl', 'wb'))#每个concept的独立子图，就是多了成索引

    np.ndarray.dump(triplets, open(fileName + '/triplets.np', 'wb'))        #过滤后的concept的子图index
    print ('Completed.')


