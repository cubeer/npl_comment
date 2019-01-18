# coding: utf-8

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import os
# import json


def pre_review(comment_str):
    resdict = {1: 'neg', 0: 'pos'}
    token = Tokenizer(num_words=500)
    comment_str_dic = [comment_str]
    token.fit_on_texts(comment_str_dic)
    input_seq = token.texts_to_sequences(comment_str_dic)
    pad = sequence.pad_sequences(input_seq, maxlen=200)
    model = load_model('npl_comment.h5')
    predict = model.predict_classes(pad)
    # print(resdict[predict[0][0]])
    return (predict)


def read_comment_test(dir_path, label, value):
    correct = 0
    comment_count = 0
    for i in os.listdir(dir_path):
        with open(dir_path + i, mode='r', encoding='UTF-8') as r:
            res = r.readline()
            comment_count += 1

            predict = pre_review(res)
            if predict == value:
                correct += 1

    print(label, ' rate:', correct / comment_count)


read_comment_test('./imdb_dataset/test/neg/', 'neg', 1)
read_comment_test('./imdb_dataset/test/pos/', 'pos', 0)
