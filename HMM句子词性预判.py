#!/usr/bin/env python
# coding: utf-8

from io import open
import numpy as np
import pandas as pd

from nltk import FreqDist
from nltk import WittenBellProbDist
from nltk.util import ngrams

from conllu import parse_incr

corpora = {}
corpora['en'] = 'UD_English-EWT/en_ewt'
corpora['es'] = 'UD_Spanish-GSD/es_gsd'
corpora['nl'] = 'UD_Dutch-Alpino/nl_alpino'


def train_corpus(lang):
    return 'D:/CS5012/P1 HMM/corpora/' + corpora[lang] + '-ud-train.conllu'  # adjust to local path. debug only


# return corpora[lang] + '-ud-train.conllu'

def test_corpus(lang):
    return 'D:/CS5012/P1 HMM/corpora/' + corpora[lang] + '-ud-test.conllu'  # adjust to local path, debug only


# return corpora[lang] + '-ud-test.conllu'

# Remove contractions such as "isn't".
def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]


def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]


# Choose language.
lang = 'en'

# Limit length of sentences to avoid underflow.
max_len = 100

train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))
test_sents = [sent for sent in test_sents if len(sent) <= max_len]
print(len(train_sents), 'training sentences')
print(len(test_sents), 'test sentences')

# counting occurrences of one part of speech following another in a training corpus
words = []
tags = []
# debug
# train_sents = [train_sents[0]]
#
for sent in train_sents:
    for token in sent:
        words.append(token.get('form'))
        tags.append(token.get('upos'))
print(len(words), " words")
print(len(tags), " tags")
sorted_tags_set = sorted(set(tags))
temp_wrd = words
sorted_words_set = sorted(set(temp_wrd))


def follow_tag_occurance(past, tag):
    count = 0
    for i in range(len(tags)):
        if ((i + 1) < len(tags)) and (tags[i] == past) and (tags[i + 1] == tag):
            count += 1
    return count



smoothed = {}
transitions_probability=[]
def get_pos_transition_probability():
    transitions = []
    # transitions_probability = np.zeros( ( len(sorted_tags_set), len(sorted_tags_set) ))
    tag_next_list = []
    for tag in sorted_tags_set:
        for next in sorted_tags_set:
            transitions.append(((tag, next), follow_tag_occurance(tag, next)))
    transitions = sorted(transitions)
    cnt = []
    for i in range(len(transitions)):
        tag_next_list.append(transitions[i][0])
        cnt.append((transitions[i][1]))
        smoothed[transitions[i][0]] = WittenBellProbDist(FreqDist(cnt), bins=1e7)
    for tag in sorted_tags_set:
        tran = []
        for i in range(len(transitions)):
            if transitions[i][0][0] == tag:  # instance of transitions[i]: {tuple2} (('VERB', 'PUNCT'), 0)
                tran.append(smoothed[transitions[i][0]].prob(transitions[i][0][1]))
        transitions_probability.append(tran)
        # transitions_probability = np.row_stack( ( transitions_probability, np.array(tran) ) )
    transitions_probability_arr = np.array(transitions_probability)
    # transition_df = pd.DataFrame(transitions_probability, columns=sorted_tags_set, index=sorted_tags_set)
    # return transitions_probability, transition_df
    return transitions_probability_arr


emission_smoothed={}
def get_emissions_probability():
    emissions = []
    # emissions_probability = np.zeros( (len(sorted_tags_set), len(sorted_words_set) ))
    emissions_probability = []
    for sent in train_sents:
        for token in sent:
            emissions.append((token.get('upos'), token.get('form')))
    #   frequency estimation with smoothing
    tags_set = set([t for (t, _) in emissions])
    for tag in sorted(tags_set):
        words = [w for (t, w) in emissions if t == tag]
        words = sorted(words) # debug michelle
        emission_smoothed[tag] = WittenBellProbDist(FreqDist(words), bins=1e7)
    #debug
    # emission_df = pd.DataFrame(smoothed, columns=sorted_words_set, index=semission_smoothedorted_tags_set)
    # print(emission_df)
    #debug
    for tag in sorted_tags_set:
        emi = []
        for word in sorted_words_set:
            emi.append((emission_smoothed[tag].prob(word)))
        emissions_probability.append(emi)
        # emissions_probability = np.row_stack( ( emissions_probability, np.array(emi) ) )
    emissions_probability = np.array(emissions_probability)
    # emission_df = pd.DataFrame(emissions_probability, columns=sorted_words_set, index=sorted_tags_set)
    # return emissions_probability, emission_df
    return emissions_probability


# def update_transition_probability(token, transitions_probability):
#     tran_row=[]
#     tran_col = []
#     sorted_tags_set.append(token.get('upos'))
#     for tag in sorted_tags_set:
#         tran_col.append(smoothed[tag].prob(token.get('upos') ) )
#     # transitions_probability.append(tran_row)
#     transitions_probability.append(tran_col)
#     # transitions_probability = np.row_stack( ( transitions_probability, np.array(tran_row) ) )
#     # transitions_probability = np.column_stack( ( transitions_probability, np.array(tran_col).T ) )
#     return transitions_probability


def update_emission_probability(token, emissions_probability_arr):
    emi = []
    for tag in sorted_tags_set:
        emi.append(emission_smoothed[tag].prob(token.get('form')))
    # emissions_probability.append(emi)
    emissions_probability_arr = np.column_stack( (emissions_probability_arr, np.array(emi).T) )
    return emissions_probability_arr

# In[95]:




# finding the best sequence of parts of speech for a list of words
# in the test corpus,
# according to a HMM model with smoothed probabilities.
def viterbi(transitions_probability_arr, emissions_probability_arr, pi, obs_seq):
    #  transit to matrix
    # transitions_probability = np.array(transitions_probability)
    # emission_probability = np.array(emission_probability)
    pi = np.array(pi)
    # words sequence
    # return a Roq * Col matrix
    # ROW: noun, verb, adj, pron, ...
    Row = transitions_probability_arr.shape[0]
    # Col: words in test sentence
    Col = len(obs_seq)
    F = np.zeros((Row, Col))
    F[:, 0] = pi * np.transpose(emissions_probability_arr[:, obs_seq[0]])
    # print("F[:,0]: ", F[:, 0])
    for t in range(1, Col):
        list_max = []
        for n in range(Row):
            list_x = list(np.array(F[:, t - 1]) * np.transpose(transitions_probability_arr[:, n]))
            # print("list_x: ", list_x)
            # get the max probability
            list_max.append(max(list_x))
        F[:, t] = np.array(list_max) * np.transpose(emissions_probability_arr[:, obs_seq[t]])
    return F


#判断句子词性序列
# initial pos probability distribution
pi = []
# invisible sequence: pos
invisible = []

for tag in sorted_tags_set:
    invisible.append(tag)
    pi.append((tags.count(tag) / len(tags)))
transitions_probability_arr = get_pos_transition_probability()
emissions_probability_arr = get_emissions_probability()

count = 0
total_count = 0
# for sent in test_sents:
for i in range(len(test_sents)):
    expected_output = []
    print("test case: ", i+1)
    sent = test_sents[i]
    for token in sent:
        expected_output.append( (token.get('form'), token.get('upos')) )
    print("expected output: ", expected_output)
    total_count += 1
    sent_words= []
    test_output = []
    # observed sequence: words
    obs_seq = []
    for token in sent:
        # if token.get('upos') not in sorted_tags_set:
        #     transitions_probability = update_transition_probability(token, transitions_probability)
        # add a column token['form'] to emission_probability array
        if token.get('form') not in sorted_words_set:
            emissions_probability_arr = update_emission_probability(token, emissions_probability_arr)
            sorted_words_set.append(token.get('form'))

        #  get observe sequence by getting each token's index in updated sorted_tags_set
        words_array = np.array(sorted_words_set)
        obs_seq.append( list( np.where( token.get('form') == words_array ) )[0][0] )
        sent_words.append(token.get('form'))

    # transitions_probability = np.array(transitions_probability)
    transition_df = pd.DataFrame(transitions_probability_arr, columns=sorted_tags_set, index=sorted_tags_set)
    # emissions_probability = np.array(emissions_probability)
    emission_df = pd.DataFrame(emissions_probability_arr, columns=sorted_words_set, index=sorted_tags_set)
    if obs_seq:
        F = viterbi(transitions_probability_arr, emissions_probability_arr, pi, obs_seq)
        F_df = pd.DataFrame(F, index=sorted_tags_set,  columns=sent_words)
        F_df.loc['max_idx'] = F_df.idxmax(axis=0) #求每一列的最大值对应的行索引,并新增一行到最后
        # print(F_df)
        for j in range(len(F_df.columns.values)):
            # F_df.columns.values[i]: word index, F_df.values[-1][i]: corresponded tag
            test_output.append( ( F_df.columns.values[j], F_df.values[-1][j] ) )
        print("test output: ", test_output)
    if test_output is expected_output:
        print("pass")
        count += 1
        print("so far", count, "cases are passed")
    else:
        print("fail")
    print("\n")
accuracy = count / total_count
print("accuracy: ", accuracy)