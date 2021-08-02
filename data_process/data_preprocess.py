# coding:utf-8
# @Time: 2021/7/16 8:56 上午
# @File: data_process.py
# @Software: PyCharm
# -*- coding:utf-8 -*
import sys, io, os
from os.path import normpath, join, dirname
# hhhh

# sys.getdefaultencoding()    # 查看设置前系统默认编码
# sys.setdefaultencoding('utf-8')
# sys.getdefaultencoding()    # 查看设置后系统默认编码
# print("---"*15)
# print(__file__)
# print(normpath(join(dirname(__file__), '../..')), flush=True)# 指向的是你文件运行的路径，如果在命令行跑那么它是根据你启动的路径来确认的
sys.path.append(normpath(join(dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
# 命令行中带的坑： 要加个PYTHONPATH=. 指向python工程的根目录，就可以省去很多麻烦（这是pycharm帮我们集成了的）
# 运行的环境路径和工程链接路径的差异性   #核心冲突，命令行认为的项目根目录和实际的项目根目录

# 使用统一基于工程根目录的方式组织文件目录
from utils.path_util import from_project_root
from utils import json_util

import argparse

# 获取doc的entities
#import stanza
# auto download resource files, if the resource has been download, you can delete this line of code.
#stanza.download("en")
#nlp = stanza.Pipeline('en', use_gpu=True)  # 斯坦福的解析工具


def getReviewConcepts(tokens, postags):
    posTagKeepType = ["JJ", "JJR", "JJS", "NNS", "NN", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    concepts = []
    for index, tag in enumerate(postags):
        if tag in posTagKeepType:
            concepts.append(tokens[index])

    concepts = set(concepts)
    return list(concepts)


def reviews2json(train_url, test_url):
    train_json_data = []
    test_json_data = []
    urls = [train_url, test_url]
    for url in urls:
        with open(url, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for l in lines:
                if l.isspace():
                    continue
                l = l.strip("\n")  # strip来去除掉换行符
                sentence = l.split("***")
                # print(sentence)  # split把字符串拆分为列表
                tokens = sentence[0].strip().split(" ")
                labels = sentence[1].strip().split(" ")
                postags = sentence[2].strip().split(" ")
                concepts = getReviewConcepts(tokens, postags)
                reviewJson = {"tokens": tokens,
                              "labels": labels,
                              "postags": postags,
                              "concepts": concepts
                              }
                if url == train_url:
                    train_json_data.append(reviewJson)
                else:
                    test_json_data.append(reviewJson)
    return train_json_data, test_json_data


def main():
    # 参数获取
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="datasets/", type=str, help="domain train.txt url")
    parser.add_argument("--save_root", default="run_out/processed_data", type=str, help="save json url")
    opt = parser.parse_args()

    domains = ["device", "laptop", "rest", "service"]
    for domain in domains:
        train_url = join(opt.data_root, domain+'.train.txt')
        test_url = join(opt.data_root, domain+'.test.txt')
        train_save_url = join(opt.save_root, domain+'_train.json')
        test_save_url = join(opt.save_root, domain+'_test.json')

        train_url = from_project_root(train_url)
        test_url = from_project_root(test_url)
        train_save_url = from_project_root(train_save_url)
        test_save_url = from_project_root(test_save_url)

        # 判断如果代码预处理完毕，则不在进行预处理
        if os.path.isfile(train_save_url):
            print(train_save_url + " has been already processed!")
            continue

        if os.path.isfile(test_save_url):
            print(test_save_url + " has been already processed!")
            continue

        # 将reviews转换成json结构 // 包含tokens化 get entities, 过滤词语，保留形容词、动词、名词作为句子的concepts
        # keys = {"tokens", "labels", "postags", "concepts"}
        train_json_data, test_json_data = reviews2json(train_url, test_url)
        # 保存数据
        json_util.dump(train_json_data, train_save_url)
        json_util.dump(test_json_data, test_save_url)


if __name__ == '__main__':
    main()
