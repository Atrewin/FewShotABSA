import joblib
import os
import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '..')))
from utils.path_util import from_project_root

"""
    本文件将数据过滤成
"""


def load_raw_data(data_url, update=False):
    """ load data into sentences and records

    Args:
        data_url: url to data file
        update: whether force to update
    Returns:
        sentences(raw), records
    """

    # load from pickle
    save_url = data_url.replace('.bio', '.nested.raw.pkl').replace('.iob2', '.nested.raw.pkl')
    if not update and os.path.exists(save_url):
        return joblib.load(save_url)

    sentences = list()
    records = list()
    with open(data_url, 'r', encoding='utf-8') as iob_file:
        first_line = iob_file.readline()
        n_columns = first_line.count('\t')
        # JNLPBA dataset don't contains the extra 'O' column
        if 'jnlpba' in data_url:
            n_columns += 1
        columns = [[x] for x in first_line.split()]
        for line in iob_file:
            if line != '\n':
                line_values = line.split()
                for i in range(n_columns):
                    columns[i].append(line_values[i])

            else:  # end of a sentence
                sentence = columns[0]
                record = infer_records(columns[1:])

                if is_nested(record):
                    sentences.append(sentence)
                    records.append(record)
                columns = [list() for i in range(n_columns)]

    joblib.dump((sentences, records), save_url)
    return sentences, records


def is_nested(record):
    """
    判断该实例是否出现嵌套实体，是则返回True, 否则返回False
    :param record:
    :return:
    """
    start = 0
    end = 0
    for i, (key, value) in enumerate(record.items()):
        (s, e) = key
        if i == 0:
            start = s
            end = e
            continue
        if s < end: # nested
            return True
        else:
            start = s
            end = e

    return False


def infer_records(columns):
    """ inferring all entity records of a sentence

    Args:
        columns: columns of a sentence in iob2 format

    Returns:
        entity record in gave sentence

    """
    records = dict()
    for col in columns:
        start = 0
        while start < len(col):
            end = start + 1
            if col[start][0] == 'B':
                while end < len(col) and col[end][0] == 'I':
                    end += 1
                records[(start, end)] = col[start][2:]
            start = end
    return records


if __name__ == '__main__':
    test_url = from_project_root("data/genia/genia.test.iob2")
    load_raw_data(test_url)
