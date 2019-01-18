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


def read_comment_test(dir_path):
    comments = []
    for i in os.listdir(dir_path):
        with open(dir_path + i, mode='r', encoding='UTF-8') as r:
            res = r.readline()
            comments.append(res)
    return comments


def run_test(dict, str, value):
    correct_neg = 0
    for str in dict:
        predict = pre_review(str)
        if predict == value:
            correct_neg += 1

    print(str, ' rate:', correct_neg / len(dict))


comment_dic_neg_test = read_comment_test('./imdb_dataset/test/neg/')
comment_dic_pos_test = read_comment_test('./imdb_dataset/test/pos/')

run_test(comment_dic_neg_test, 'neg', 1)
run_test(comment_dic_pos_test, 'pos', 0)

#neg review
#pre_review('This movie was obviously made with a very low budget, but did they have to make it so obvious? It looked like they made no effort to make the "future" look in the least futuristic. For example, the first scene takes place in an 80\'s office building and all the cars that get blown up are from the late 70\'s (I assume they didn\'t want to blow up cars that cost more than $500). Additionally, its pretty obvious that Don "the Dragon" is driving his personal car during the movie (after all, he did partially fund the film). Finally, they point out at the beginning of the film that all kinds of drugs are now legal in this new "cyberpunk" society. Not only does this never become important in the film, but later when don needs surgery without anesthesia, why doesn\'t he just go out and get some legal heroin or morphine? The whole movie is sloppy like this and completely anticlimactic since Don easily blows up an "unstoppable" Cybertracker about 25 minutes into the movie. However, if you find this movie cheap or free I\'d watch it, the last scene is almost worth putting up with this whole film.')
#pos review
#pre_review('If you had asked me how the movie was throughout the film, I would have told you it was great! However, I left the theatre feeling unsatisfied. After thinking a little about it, I believe the problem was the pace of the ending. I feel that the majority of the movie moved kind of slow, and then the ending developed very fast. So, I would say the ending left me disappointed.<br /><br />I thought that the characters were well developed. Costner and Kutcher both portrayed their roles very well. Yes! Ashton Kutcher can act! Also, the different relationships between the characters seemed very real. Furthermore,I thought that the different plot lines were well developed. Overall, it was a good movie and I would recommend seeing it.<br /><br />In conclusion: Good Characters, Great Plot, Poorly Written/Edited Ending. Still, Go See It!!!')
