#coding: utf-8

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import SimpleRNN


def read_comment(dir_path):
    comments = []
    for i in os.listdir(dir_path):
        with open(dir_path + i, mode='r', encoding='UTF-8') as r:
            res = r.readline()
            comments.append(res)
    return comments


comment_dic_neg = read_comment('./imdb_dataset/train/neg/')
comment_dic_pos = read_comment('./imdb_dataset/train/pos/')

comment_dic = comment_dic_neg + comment_dic_pos

token = Tokenizer(num_words=2000)
token.fit_on_texts(comment_dic)
# print token.document_count
# print token.word_index

x_train_seq = token.texts_to_sequences(comment_dic)
x_train = sequence.pad_sequences(x_train_seq, maxlen=200, padding='post')

all_labels = [1] * len(comment_dic_neg) + [0] * len(comment_dic_pos)

print(len(comment_dic_neg), len(comment_dic_pos), len(comment_dic))

# 构建网络
model = Sequential()
model.add(Embedding(output_dim=20, input_dim=2000, input_length=200))
model.add(Dropout(0.1))
model.add(SimpleRNN(units=16))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    x=x_train, y=all_labels, batch_size=500, validation_split=0.2, epochs=5)

model.save('npl_comment.h5')