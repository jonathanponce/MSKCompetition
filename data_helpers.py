# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import re
import itertools
from collections import Counter
import os
# from gensim.models import word2vec

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[<>\+\.:;]", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    text2 = np.loadtxt("train_variants_extra", delimiter="|",dtype=np.str)
    text = np.loadtxt("train_txt_new", delimiter="||",skiprows=1,dtype=np.str)
    labels = np.loadtxt("training_variants/training_variants", delimiter=",",skiprows=1,dtype=np.str)
    # Split by words
    x_text = [clean_str(sent[1]) for sent in text]
    x_text2 = [clean_str(sent[1]) for sent in text2]
    x_text = [x_text2[x]+" "+x_text[x] for x in range(len(x_text))]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    y = labels[:,-1].astype(np.int8)
    y= [x-1 for x in y]
    return [x_text, y]

def load_data_and_labels_eval():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    x_text2 = list(open("test_variants_extra").readlines())
    x_text2 = [clean_str(x_text2[x]) for x in range(len(x_text2))]
    text = np.loadtxt("test_txt_new", delimiter="||",skiprows=1,dtype=np.str)
    # Split by words
    x_text = [clean_str(sent[1]) for sent in text]
    x_text = [x_text2[x]+" "+x_text[x] for x in range(len(x_text))]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    y = []
    return [x_text, y]

    
def pad_sentences_eval(sentences,sentences_train, padding_word="</s>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(sentences_train,max(len(x) for x in sentences))
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences
def pad_sentences(sentences,sentences_test, padding_word="</s>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(sentences_test,max(len(x) for x in sentences))
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    #add words that only occur in the test corpus to the vocab
    labels = list(open("test_variants_extra").readlines())
    x_text = [clean_str(labels[x]) for x in range(len(labels))]
    x_text = [s.split(" ") for s in x_text]
    sentences_padded_train = pad_sentences_eval(x_text,len(sentences[0]))
    sentences=np.concatenate((sentences, sentences_padded_train), axis=0)
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def build_input_data_with_word2vec(sentences, labels, word2vec):
    """Map sentences and labels to vectors based on a pretrained word2vec"""
    x_vec = []
    for sent in sentences:
        vec = []
        for word in sent:
            if word in word2vec:
                vec.append(word2vec[word])
            else:
                vec.append(word2vec['</s>'])
        x_vec.append(vec)
    x_vec = np.array(x_vec)
    y_vec = np.array(labels)
    return [x_vec, y_vec]


def load_data_with_word2vec(word2vec):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    return build_input_data_with_word2vec(sentences_padded, labels, word2vec)

def load_data_eval():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data 
    sentences, labels = load_data_and_labels_eval()
    sentences_train, labels_train = load_data_and_labels()
    # must add longest sentences of both sets
    sentences_padded_train = pad_sentences(sentences_train,max(len(x) for x in sentences))
    sentences_padded = pad_sentences_eval(sentences,max(len(x) for x in sentences_train))
    vocabulary, vocabulary_inv = build_vocab(sentences_padded_train)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]
    
def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_test, _ = load_data_and_labels_eval()
    # must add longest sentences of both sets
    sentences_padded = pad_sentences(sentences,max(len(x) for x in sentences_test))
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_pretrained_word2vec(infile):
    if isinstance(infile, str):
        infile = open(infile)

    word2vec = {}
    for idx, line in enumerate(infile):
        if idx == 0:
            vocab_size, dim = line.strip().split()
        else:
            tks = line.strip().split()
            word2vec[tks[0]] = map(float, tks[1:])

    return word2vec


def load_google_word2vec(path):
    model = word2vec.Word2Vec.load_word2vec_format(path, binary=True)
    return model