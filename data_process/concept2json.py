# coding:utf-8
# @Time: 2021/7/16 3:22 下午
# @File: data_concept_link.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
import argparse
import sys
import codecs
from tqdm import tqdm
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import os
from os.path import normpath,join
Base_DIR= normpath(join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, Base_DIR)
from utils.path_util import from_project_root
from utils import json_util


def getDomainDataURL(data_root, domainList):
    urlList= []
    for domain in domainList:
        domain_urls = []
        if domain in ["device", "laptop", "rest", "service"]:
            domain_urls.append(from_project_root(join(data_root, domain+'_train.json')))
            domain_urls.append(from_project_root(join(data_root, domain+'_test.json')))
        else:
            error = 0/0
        urlList.extend(domain_urls)
    return urlList


def getReviewConceptNetTriples(conceptList):
    # return [len(concepts), len(triples)]
    dataRoot = "run_out/processed_data/"
    conceptGraphDict_keep_url = join(dataRoot, "ConceptsGraphDict.json")
    conceptGraphDict_keep_url = from_project_root(conceptGraphDict_keep_url)

    if os.path.exists(conceptGraphDict_keep_url):
        concept2ConceptGraphDict = json_util.load(conceptGraphDict_keep_url)["concept2ConceptGraphDict"]
    else:
        concept2ConceptGraphDict = {}
    reviewConceptNetTriples = []
    for concept in conceptList:
        # 因为前面连接的字典可能早不到这个图所以需要先检查concept2ConceptGraphDict[concept]是否存在
        if concept in concept2ConceptGraphDict.keys():
            concept2ConceptGraphDict[concept]
            reviewConceptNetTriples.extend(concept2ConceptGraphDict[concept])
    return reviewConceptNetTriples


def addConceptNetTripe2reviewJson(reviewJsonList):

    index_arr = tqdm(range(len(reviewJsonList)))
    for index in index_arr:
        reviewJson = reviewJsonList[index]
        reviewConcepts = reviewJson["concepts"]
        if "conceptNetTriples" not in reviewJson.keys() or len(reviewJson["conceptNetTriples"]) == 0:
            conceptNetTriples = getReviewConceptNetTriples(reviewConcepts)
            reviewJson["conceptNetTriples"] = conceptNetTriples
            reviewJsonList[index] = reviewJson
        pass
    return reviewJsonList


def addConceptNetTriple2JsonData(data_file):
    """
    :param data_file: domain_data_file
    :return:
    """
    url = from_project_root(data_file)
    json_data = json_util.load(url)
    json_data = addConceptNetTripe2reviewJson(json_data)
    json_util.dump(json_data, url)
    print("addConceptNetTriple2JsonData to " + data_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="run_out/processed_data", help="data_json_dir")
    parser.add_argument("--start", default=0, help="beginIndex")

    opt = parser.parse_args()

    # 统一获取后分配的方案
    domainList = ["device", "laptop", "rest", "service"]
    dataRoot = opt.data_path
    concept_keep_url = join(dataRoot, "ConceptsSetDict.json")
    concept_keep_url = from_project_root(concept_keep_url)

    conceptGraphDict_keep_url = join(dataRoot, "ConceptsGraphDict.json")
    conceptGraphDict_keep_url = from_project_root(conceptGraphDict_keep_url)

    # TODO 回写到reviewJosnData
    urlList = getDomainDataURL(dataRoot, domainList)

    for url in urlList:
        print("Add concepts into the " + url)
        addConceptNetTriple2JsonData(url)

    print("完成获取conceptTriple写入JsonData")
