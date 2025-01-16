import os
import math
from tqdm import tqdm
import re
import random
import abc
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse
import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor


def doc_preprocess(dataset,node_name,dataset_path,output_path):



    node_path = os.path.abspath(os.path.join(dataset_path, 'nodes'))
    edge_path = os.path.abspath(os.path.join(dataset_path, 'edges'))
    if dataset=='NIH_reporter':
        output_path_node = os.path.join(output_path, 'NIH_reporter/nodes')
        output_path_edge = os.path.join(output_path, 'NIH_reporter/edges')
        if not os.path.exists(output_path_node):
            os.makedirs(output_path_node)
        if not os.path.exists(output_path_edge):
            os.makedirs(output_path_edge)
        corpus = get_node_feature(node_name, node_path)

        feature_size = 10
        data_feature_vector = word2vec_process(corpus,feature_size)
        data_feature_vector=np.array(data_feature_vector)
        print(data_feature_vector)
        # data_feature_vector = corpus
        if node_name=='publications':
            corpus=pd.DataFrame(corpus)
            corpus.to_csv(os.path.join(output_path_node, 'publications_corpus.csv'), index=False)
            # data_feature_vector.to_csv(os.path.join(output_path_node, 'publications.csv'), index=False)
        if node_name=='PI':
            corpus=pd.DataFrame(corpus)
            corpus.to_csv(os.path.join(output_path_node, 'PI_corpus.csv'), index=False)
        # if node_name == 'authors':
        #     data_feature_vector.to_csv(os.path.join(output_path_node, 'authors.csv'), index=False)
        if node_name == 'affiliation':
            np.save(os.path.join(output_path_node, 'affiliation_feature.npy'), data_feature_vector)

        if node_name == 'venue':
            np.save(os.path.join(output_path_node, 'venue_feature.npy'), data_feature_vector)
    # return data_feature_vector

def get_node_feature(node_name, node_path):
    assert node_name in ['affiliation', 'authors', 'PI', 'publications', 'venue']
    corpus = []
    if node_name == 'publications':
        node_file = pd.read_csv(os.path.join(node_path, 'publications.csv'))
        node_file = node_file.fillna('')

        for index, row in tqdm(node_file.iterrows(), total=node_file.shape[0]):
            text_feature = row['title'] + ' ' + row['keywords'] + ' ' + row['country']

            corpus.append(text_feature)

    if node_name == 'PI':
        node_file = pd.read_csv(os.path.join(node_path, 'PI.csv'))
        node_file = node_file.fillna('')

        for index, row in tqdm(node_file.iterrows(), total=node_file.shape[0]):
            text_feature = (row['terms'] + ' ' + row['pref_terms'] + ' ' + row['abstract_text']
                            + ' ' + row['project_title'] + ' ' + row['organization_org_name'])

            corpus.append(text_feature)

    if node_name == 'authors':
        node_file = pd.read_csv(os.path.join(node_path, 'authors.csv'))
        node_file = node_file.fillna('')

        for index, row in tqdm(node_file.iterrows(), total=node_file.shape[0]):
            text_feature = (str(row['research_time_span']) + ' ' + str(row['cited_paper_count']) + ' '+
                            row['race'] + ' ' + row['gender'])

            corpus.append(text_feature)
    if node_name == 'affiliation':
        node_file = pd.read_csv(os.path.join(node_path, 'affiliation.csv'))
        node_file = node_file.fillna('')

        for index, row in tqdm(node_file.iterrows(), total=node_file.shape[0]):
            text_feature = row['organization']
            corpus.append(text_feature)
    if node_name == 'venue':
        node_file = pd.read_csv(os.path.join(node_path, 'venue.csv'))
        node_file = node_file.fillna('')

        for index, row in tqdm(node_file.iterrows(), total=node_file.shape[0]):
            text_feature = row['simplified_journal']
            corpus.append(text_feature)

    return corpus
def word2vec_process(corpus,feature_size):
    feature_size = feature_size
    window_context = 5
    min_word_count = 1

    corpus_norm = [document_tokenize(text) for text in corpus]
    corpus_tokens = [document_tokenize(text).split(' ') for text in corpus]
    # tfidf = TfidfVectorizer()
    # X = tfidf.fit_transform(corpus).toarray()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    w2v_model = word2vec.Word2Vec(
        corpus_tokens,
        vector_size=feature_size,  # Word embeddings dimensionality
        window=window_context,  # Context window size
        min_count=min_word_count,  # Minimum word count
        sg=1,  # `1` for skip-gram; otherwise CBOW.
        seed=123,  # random seed
        workers=1,  # number of cores to use
        negative=5,  # how many negative samples should be drawn
        cbow_mean=1,  # whether to use the average of context word embeddings or sum(concat)
        epochs=100,  # number of epochs for the entire corpus
        batch_words=10000,  # batch size
    )
    words = w2v_model.wv.index_to_key
    wvs = w2v_model.wv[words]  ## get embeddings of all word forms

    tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=2)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(wvs)
    labels = words

    plt.figure(figsize=(12, 6))
    plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        plt.annotate(label,
                     xy=(x + 1, y + 1),
                     xytext=(0, 0),
                     textcoords='offset points')

    w2v_feature_array = averaged_word_vectorizer(corpus=corpus_tokens,
                                                 model=w2v_model,
                                                 num_features=feature_size)
    data_feature_vector = pd.DataFrame(w2v_feature_array, index=corpus_norm)
    return data_feature_vector

def document_tokenize(doc):

    wpt = nltk.WordPunctTokenizer()
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    doc = ' '.join(tokens)
    return doc

def average_word_vectors(words, model, vocabulary, num_features):

    feature_vector = np.zeros((num_features, ), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [
        average_word_vectors(tokenized_sentence, model, vocabulary,
                             num_features) for tokenized_sentence in corpus
    ]
    return np.array(features)

def main():
    output_path = os.path.join(os.getcwd(), 'data/processed_data')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dataset_path = os.path.join(os.getcwd(), 'data/raw_data/NIH_reporter')
    doc_preprocess(dataset='NIH_reporter', node_name='venue', dataset_path=dataset_path, output_path = output_path)



if __name__ == '__main__':

    main()

