#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 09:12:10 2023

@author: dfox
"""

USE_BERT = False

import datetime as dt
start = dt.datetime.now()
def print_time(text):
    print(f"{text}: elapsed time {dt.datetime.now() - start}")
    
import enchant
import pandas as pd
import nltk
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile as zf
import os
import numpy as np
import scipy
import json
import random
import pickle
nltk.download('punkt')
nltk.download('words')
import nltk.data
from gensim.parsing.preprocessing import (
    remove_stopwords,
    strip_punctuation,
    strip_numeric,
    strip_non_alphanum,
    strip_multiple_whitespaces,
    strip_short,
    preprocess_string,
)
# from gensim.summarization.summarizer import textcleaner, summarize

# from sentence_transformers import SentenceTransformer
    
from nltk.stem import WordNetLemmatizer

use_zip = False

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

def remove_non_english_words(
    words: list, eng_words: list = list(set(nltk.corpus.words.words()))
) -> list:
    """It is better to not use this func as 'eng_words' does not consider
    different format of a word. Or we need to do lemmatization."""
    return [x for x in words if x in eng_words]


## The following funcs can be changed to static method for the class
def strip_short2(s, minsize=1):
    return strip_short(s, minsize)

PUNC = list(string.punctuation)
MORE_PUNC = ['..', '...', '. . .','“','’','…','”','–','️','``',"''",
             '``',"''",'•','••••','©', '••']
PUNC.extend(MORE_PUNC)
    

def norm_sentence(
    sent: str,
    terminate: bool = True,
    do_lemma: bool = False,
    only_english: bool = True,
    eng_dict: enchant.Dict = enchant.Dict("en_US"),
    lemmatizer: WordNetLemmatizer = WordNetLemmatizer(),
) -> str:
    """_summary_
    Normalize a sentence.
    Args:
        sent (str): _description_
        do_lemma (bool, optional): Do lemmatization or not. Defaults to True.
            Note that if we want to do lemmatization. Need to do POS tagging first;
            otherwise, NOUN would be considered in the lemmatization so some words
            would not be transformed to the expected form.
        only_english (bool, optional): _description_. Defaults to False.
        eng_dict (enchant.Dict, optional): Dictionary of English words.
            Defaults to enchant.Dict("en_US"). It is used to check the spelling
            of English words.

    Returns:
        str: _description_
    """
    norm_sent_steps = [
        lambda x: x.lower(),
        strip_non_alphanum,
        strip_numeric,
        strip_punctuation,
        strip_multiple_whitespaces,
        strip_short2,
        remove_stopwords
    ] # remove_stopwords,
    normed_words = preprocess_string(sent, norm_sent_steps)
    if only_english:
         normed_words = [x for x in normed_words if eng_dict.check(x)]
    if do_lemma:
        # If do lemmatization, need to do pos tagging first.
        # Here the POS tagging part is not added. Can use pos_tag from nltk.tag
        normed_words = [lemmatizer.lemmatize(x) for x in normed_words]

    new_sent = " ".join(normed_words)
    if terminate:
        new_sent += "."

    return(new_sent)

def text_to_sentences(text: str) -> list:
    """Split text or document to sentences"""
    return tokenizer.tokenize(text)


## Some sentences generated after using 'tokenizer' and need to be removed
TO_REMOVE_SENT = [
    "",
    ".",
    "pp.",
    "p.",
    "o o.",
    "lj.",
    "c y.",
    "r p.",
    "l l.",
    "v.",
    "j.",
]


def get_norm_sentences_from_text(
    text: str, terminate: bool = True, 
    to_remove_sentence: list = TO_REMOVE_SENT
) -> list:
    """_summary_
    Get the normalized sentences from a document.
    Args:
        text (str): A document of text
        to_remove_sentence (list, optional): Some sentences that
            we want to remove from the returned sentences.

    Returns:
        list: normalized sentences.
    """
    sents = text_to_sentences(text)
    normed_sents = [norm_sentence(x, terminate, True) for x in sents]
    normed_sents = [x for x in normed_sents if not x in to_remove_sentence]
    return normed_sents

def raw_text_to_words(text):
    # Used by TD and TFIDF
    sent_list = get_norm_sentences_from_text(text, False)
    return ' '.join(sent_list)

def raw_text_to_word_list(text):
    # Used by Doc2Vec
    doc = []
    sent_list = get_norm_sentences_from_text(text, False)
    for sent in sent_list:
        doc.extend(sent.split(' '))
    return doc

    
# def get_params_list(params_dict):
#     keys = params_dict.keys()
#     vals = params_dict.values()
    
#     combinations = list(itertools.product(*vals))
#     return [dict(zip(keys, combination)) for combination in combinations]

class CorpusVectorizer:
    def __init__(self, vectorizer, cache_id, transformed_cache_id, ops, 
                 **kwargs):
        self.vectorizer = vectorizer
        self.ops = ops
        self.method = cache_id.split('_')[0]
        self.cache_id = cache_id
        self.transformed_cache_id = transformed_cache_id
        self.vectorizer_filename = self.ops.obj_filename.format(cache_id)
        self.mtx_filename = self.ops.mtx_filename.format(cache_id)
        self.matrix_read_from_file = 0
        self.matrix_obj = None
        
    def get_matrix(self):
        matrix = None
        msize = 0
        cache_id = self.cache_id
        method = self.method
        
        self.ops.v_meth[method]['load']()
        
        mtx_filename = self.mtx_filename
        vectorizer_filename = self.vectorizer_filename

        mtx_readcache_func = self.ops.v_meth[method]['mrfunc']
        if os.path.exists(mtx_filename):
            msize, matrix = mtx_readcache_func(mtx_filename)
            
        if self.transformed_cache_id != cache_id:
            # we haven't fitted and transformed the corpus yet
            # ignore anything in memory or disk cache
            matrix = self.ops.v_meth[method]['mcfunc'](self)
            msize = self.ops.corpus_size
            self.transformed_cache_id = cache_id
    
            self.ops.v_meth[method]['mwfunc'](mtx_filename, matrix)
            with open(vectorizer_filename, "wb") as fp:
                 pickle.dump(self, fp)
            if use_zip:
                with zf.ZipFile(self.ops.zipfilename, 'a') as myzip:
                    myzip.write(mtx_filename)
                    myzip.write(vectorizer_filename)
          
        if cache_id in self.ops.v_meth[method]['mtx']:
            matrix = self.ops.v_meth[method]['mffunc']\
                (self.ops.v_meth[method]['mtx'][cache_id])
            msize = self.ops.corpus_size
            self.matrix_obj = CorpusMatrix(matrix, cache_id, self)
            return self.matrix_obj
        
        if matrix is None:
            mtx_readcache_func = self.ops.v_meth[method]['mrfunc']
            if os.path.exists(self.ops.zipfilename):
                if use_zip:
                    with zf.ZipFile(self.ops.zipfilename, "r") as myzip:
                        if mtx_filename in myzip.namelist():
                            myzip.extract(mtx_filename)
                            msize, matrix = mtx_readcache_func(mtx_filename)
                            os.remove(mtx_filename)
            elif os.path.exists(mtx_filename):
                msize, matrix = mtx_readcache_func(mtx_filename)

        if matrix is None or msize < self.ops.corpus_size:
            self.matrix = self.ops.v_meth[method]['mcfunc'](self)
            msize = self.ops.corpus_size
            self.transformed_cache_id = cache_id
    
            self.ops.v_meth[method]['mwfunc'](mtx_filename, matrix)
            with open(vectorizer_filename, "wb") as fp:
                pickle.dump(self, fp)
            if use_zip:
                with zf.ZipFile(self.ops.zipfilename, 'a') as myzip:
                    myzip.write(mtx_filename)
                    myzip.write(vectorizer_filename)
                    os.remove(mtx_filename)
                    os.remove(vectorizer_filename)

        if self.ops.max_memory_matrix_cache > 0:
            while (self.ops.max_memory_matrix_cache 
                   < len(self.ops.v_meth[method]['mtx'])):
                self.ops.v_meth[method]['mtx'].\
                    pop(next(iter(self.v_meth[method]['mtx'])))
            self.ops.v_meth[method]['mtx'][cache_id] =\
                self.ops.v_meth[method]['mtfunc'](matrix)
                
        self.matrix_obj = CorpusMatrix(matrix, cache_id, self)
        return self.matrix_obj
    
class CorpusMatrix:
    def __init__(self, matrix, cache_id, vectorizer):
        self.matrix = matrix
        self.cache_id = cache_id
        self.method = cache_id.split('_')[0]
        print(type(vectorizer))
        self.vectorizer = vectorizer
        self.ops = vectorizer.ops
        
    def get_similarity(self, vector_text=None):
        
        matrix = self.matrix
        cache_id = self.cache_id
        method = self.method
        ops = self.ops
        if vector_text is not None:
            cache_id = cache_id + "_" + str(hash(vector_text))
        
        sim_matrix = None
        tvec = None

        ops.v_meth[method]['load']()
         
        vector_filename = ops.vec_filename.format(cache_id)
        if vector_text is not None:
            if cache_id in ops.v_meth[method]['mtx']:
                tvec = ops.v_meth[method]['mtx'][cache_id]
            elif use_zip and os.path.exists(ops.zipfilename):        
                with zf.ZipFile(ops.zipfilename, 'r') as myzip:
                    if vector_filename in myzip.namelist():
                        myzip.extract(vector_filename)
                        _, tvec = ops.v_meth[method]['mrfunc'](vector_filename)
                        os.remove(vector_filename)
            elif os.path.exists(vector_filename):
                _, tvec = ops.v_meth[method]['mrfunc'](vector_filename)
                
            if tvec is None:
                print("calculating document similarity")
                tvec = ops.v_meth[method]['vtfunc'](self.vectorizer,
                                                    vector_text)
            if ops.corpus_disk_cache and not os.path.exists(vector_filename):
                ops.v_meth[method]['mwfunc'](vector_filename, tvec)
                if use_zip:
                    with zf.ZipFile(ops.zipfilename, 'a') as myzip:
                        myzip.write(vector_filename)
                        os.remove(vector_filename)
        
        sim_filename = ops.sim_filename.format(cache_id)
        
        if cache_id in ops.v_meth[method]['simmtx']:
            sim_matrix = ops.v_meth[method]['simmtx'][cache_id]
        
        if sim_matrix is not None:
            return CorpusSimMatrix(sim_matrix, matrix)
    
        if use_zip:
            if os.path.exists(ops.zipfilename):        
                with zf.ZipFile(ops.zipfilename, 'r') as myzip:
                    if sim_filename in myzip.namelist():
                        myzip.extract(sim_filename)
                        sim_matrix =\
                            ops.v_meth[method]['srfunc'](sim_filename)
                        os.remove(sim_filename)
        elif os.path.exists(sim_filename):
            sim_matrix =\
                ops.v_meth[method]['srfunc'](sim_filename)
        
        if sim_matrix is None:
        
            if tvec is not None:
                sim_matrix = ops.v_meth[method]['scfunc'](matrix, tvec)
            else:
                sim_matrix = ops.v_meth[method]['scfunc'](matrix)
    
        if ops.corpus_disk_cache and not os.path.exists(sim_filename):
            ops.v_meth[method]['swfunc'](sim_filename, sim_matrix)
            if use_zip:
                with zf.ZipFile(ops.zipfilename, 'a') as myzip:
                    myzip.write(sim_filename)
                    os.remove(sim_filename)

        if ops.max_memory_sim_cache > 0:
            while (ops.max_memory_sim_cache < 
                   len(ops.v_meth[method]['simmtx'])):
                ops.v_meth[method]['simmtx'].\
                    pop(next(iter(ops.v_meth[method]['simmtx'])))
            ops.v_meth[method]['simmtx'] = sim_matrix
        
        return CorpusSimMatrix(sim_matrix, matrix)

class CorpusSimMatrix:
    def __init__(self, sim_matrix, corpus_matrix):
        self.sim_matrix = sim_matrix
        self.corpus_matrix = corpus_matrix

class CorpusOps:
    def __init__(self, name, corpus, **kwargs):
        # corpus is a pre-loaded dataframe

        # currently we only support a .csv with a specific format for the
        # corpus
        
        self.max_memory_matrix_cache = 0
        self.max_memory_sim_cache = 0
        self.max_memory_vectorizer_cache = 0
        self.max_disk_matrix_cache = 3
        self.max_disk_sim_cache = 3
        self.corpus_disk_cache = True
        for key, val in kwargs.items():
            if key == 'max_memory_matrix_cache':
                self.max_memory_matrix_cache = val
            if key == 'max_memory_sim_cache':
                self.max_memory_sim_cache = val
            if key == 'max_memory_vectorizer_cache':
                self.max_memory_vectorizer_cache = val
            if key == 'mask_disk_matrix_cache':
                self.mask_disk_matrix_cache = val
            if key == 'max_disk_sim_cache':
                self.max_disk_sim_cache = val
            if key == 'corpus_disk_cache':
                self.corpus_disk_cache = val

        self.name = name
        # all state files will be stored in the working directory
        self.base_filename = f'{self.name}.csv'
    
        self.base_proc_name = f'{self.name}_proc'
       
        self.zipfilename = f'{self.base_proc_name}.zip'
        self.dl_filename = f'{self.base_proc_name}_dl.json'
        self.vec_filename = f'vec_{self.base_proc_name}' + '_{}.mtx'
        self.mtx_filename = f'mtx_{self.base_proc_name}' + '_{}.mtx'
        self.sim_filename = f'sim_{self.base_proc_name}' + '_{}.csv'
        self.obj_filename = f'obj_{self.base_proc_name}' + '_{}.pkl'
        
        # each supported vectorize method goes here
        self.v_meth = {'td'     : {'mtx': {}, 'simmtx': {}, 'vec' : {},
                                   'corpus': None,
                                   'corpus_filename': ('corp_' +
                                                       f'{self.base_proc_name}'
                                                       + '_td.json'),
                                   'corpfunc': self.doc_list_to_corpus_td,
                                   'docfunc': raw_text_to_words,
                                   'load': self.load_corpus_td,
                                   'mrfunc': self.read_diskcache_td_mtx,
                                   'mwfunc': self.write_diskcache_td_mtx,
                                   'ivfunc': self.init_td_vectorizer,
                                   'mcfunc': self.calc_td_mtx,
                                   'vtfunc': self.transform_td_vec,
                                   'mtfunc': self.to_cache_td_mtx,
                                   'mffunc': self.from_cache_td_mtx,
                                   'srfunc': self.read_diskcache_td_sim,
                                   'swfunc': self.write_diskcache_td_sim,
                                   'scfunc': self.calc_td_sim},
                       'tfidf'  : {'mtx': {}, 'simmtx': {}, 'vec' : {},
                                   'corpus': None,
                                   'corpus_filename': ('corp_' +
                                                       f'{self.base_proc_name}'
                                                       + '_td.json'),
                                   'corpfunc': self.doc_list_to_corpus_tfidf,
                                   'docfunc': raw_text_to_words,
                                   'load': self.load_corpus_tfidf,
                                   'mrfunc': self.read_diskcache_tfidf_mtx,
                                   'mwfunc': self.write_diskcache_tfidf_mtx,
                                   'ivfunc': self.init_tfidf_vectorizer,
                                   'mcfunc': self.calc_tfidf_mtx,
                                   'vtfunc': self.transform_tfidf_vec,
                                   'mtfunc': self.to_cache_tfidf_mtx,
                                   'mffunc': self.from_cache_tfidf_mtx,
                                   'srfunc': self.read_diskcache_tfidf_sim,
                                   'swfunc': self.write_diskcache_tfidf_sim,
                                   'scfunc': self.calc_tfidf_sim},
                       'doc2vec': {'mtx': {}, 'simmtx': {}, 'vec' : {},
                                   'corpus': None,
                                   'corpus_filename': ('corp_' +
                                                       f'{self.base_proc_name}'
                                                       + '_d2v.json'),
                                   'corpfunc': self.doc_list_to_corpus_d2v,
                                   'docfunc': raw_text_to_word_list,
                                   'load': self.load_corpus_d2v,
                                   'mrfunc': self.read_diskcache_d2v_mtx,
                                   'mwfunc': self.write_diskcache_d2v_mtx,
                                   'ivfunc': self.init_d2v_vectorizer,
                                   'mcfunc': self.calc_d2v_mtx,
                                   'vtfunc': self.transform_d2v_vec,
                                   'mtfunc': self.to_cache_d2v_mtx,
                                   'mffunc': self.from_cache_d2v_mtx,
                                   'srfunc': self.read_diskcache_d2v_sim,
                                   'swfunc': self.write_diskcache_d2v_sim,
                                   'scfunc': self.calc_d2v_sim},
                       # 'txtrnk'  : {'mtx': {}, 'simmtx': {}, 'vec' : {},
                       #             'corpus': None,
                       #             'corpus_filename': ('corp_' +
                       #                                 f'{self.base_proc_name}'
                       #                                 + '_td.json'),
                       #             'corpfunc': self.doc_list_to_corpus_txtrnk,
                       #             'docfunc': textcleaner.clean_text,
                       #             'load': self.load_corpus_txtrnk,
                       #             'mrfunc': self.read_diskcache_txtrnk_mtx,
                       #             'mwfunc': self.write_diskcache_txtrnk_mtx,
                       #             'ivfunc': self.init_txtrnk_vectorizer,
                       #             'mcfunc': self.calc_txtrnk_mtx,
                       #             'vtfunc': self.transform_txtrnk_vec,
                       #             'mtfunc': self.to_cache_txtrnk_mtx,
                       #             'mffunc': self.from_cache_txtrnk_mtx,
                       #             'srfunc': self.read_diskcache_txtrnk_sim,
                       #             'swfunc': self.write_diskcache_txtrnk_sim,
                       #             'scfunc': self.calc_txtrnk_sim},
                       # 'bert'   : {'mtx': {}, 'simmtx': {}, 'vec' : {},
                       #             'corpus': None,
                       #             'corpus_filename': ('corp_' +
                       #                                 f'{self.base_proc_name}'
                       #                                 + '_bert.json'),
                       #             'corpfunc': self.doc_list_to_corpus_bert,
                       #             'docfunc': get_norm_sentences_from_text,
                       #             'load': self.load_corpus_bert,
                       #             'mrfunc': self.read_diskcache_bert_mtx,
                       #             'mwfunc': self.write_diskcache_bert_mtx,
                       #             'ivfunc': self.init_bert_vectorizer,
                       #             'mcfunc': self.calc_bert_mtx,
                       #             'vtfunc': self.transform_bert_vec,
                       #             'mtfunc': self.to_cache_bert_mtx,
                       #             'mffunc': self.from_cache_bert_mtx,
                       #             'srfunc': self.read_diskcache_bert_sim,
                       #             'swfunc': self.write_diskcache_bert_sim,
                       #             'scfunc': self.calc_bert_sim}
                       }
        
        self.tokens_dict = None
        self.corpus = None
        self.tagged_corpus = None
        self.cdf = corpus.copy()
        self.corpus_size = corpus.shape[0]
      
    def load_corpus(self, method):
        # special case: 'td' and 'tfidf' share the same corpus
        corpus = self.v_meth[method]['corpus']
        corpus_filename = self.v_meth[method]['corpus_filename']
        if corpus is None:
            if self.corpus_disk_cache:
                if use_zip and os.path.exists(self.zipfilename):
                    with zf.ZipFile(self.zipfilename, "r") as myzip:
                        if corpus_filename in myzip.namelist():
                            myzip.extract(corpus_filename)
                            with open(corpus_filename, "r") as fp:
                                doc_list = json.loads(fp.read())
                                corpus = self.v_meth[method]['corpfunc'](doc_list)
                            os.remove(corpus_filename)
                elif os.path.exists(corpus_filename):
                    with open(corpus_filename, "r") as fp:
                        doc_list = json.loads(fp.read())
                        corpus = self.v_meth[method]['corpfunc'](doc_list)
                    
        if corpus is None:
            doc_list = self.df_to_doc_list(self.cdf, method)
            if self.corpus_disk_cache:
                with open(corpus_filename, "w") as fp:
                    json.dump(doc_list, fp)
                if use_zip:
                    with zf.ZipFile(self.zipfilename, 'a') as myzip:
                        myzip.write(corpus_filename)
                        os.remove(corpus_filename)
            corpus = self.v_meth[method]['corpfunc'](doc_list)
            
        self.v_meth[method]['corpus'] = corpus
        
    def df_to_doc_list(self, df, method):
        doc_list = []
        for index, row in df.iterrows():
            if index != 0 and index % 100 == 0:
                print(index)
            doc_list.append(self.v_meth[method]['docfunc'](row['full_text']))
        return doc_list

    def load_corpus_td(self):
        if self.v_meth['td']['corpus'] is not None:
            return
        if self.v_meth['tfidf']['corpus'] is not None:
            self.v_meth['td']['corpus'] = self.v_meth['tfidf']['corpus']
            return
        self.load_corpus('td')
    def doc_list_to_corpus_td(self, doc_list):
        return doc_list
    def read_diskcache_td_mtx(self, filename):
        sm = scipy.io.mmread(filename)
        return sm.shape[0], sm
    def init_td_vectorizer(self, **kwargs):
        return CountVectorizer(**kwargs)
    def calc_td_mtx(self, vectorizer):
        return vectorizer.vectorizer.fit_transform(self.v_meth['td']['corpus'])
    def transform_td_vec(self, vectorizer, vector_text):
        vector = raw_text_to_words(vector_text)
        return vectorizer.vectorizer.transform([vector])
    def write_diskcache_td_mtx(self, filename, matrix):
        # matrix is return from calc_td_mtx()
        scipy.io.mmwrite(filename, matrix)
    def to_cache_td_mtx(self, matrix):
        return matrix # already sparse
    def from_cache_td_mtx(self, matrix):
        return matrix # already sparse
    def read_diskcache_td_sim(self, filename):
        return pd.read_csv(filename, index_col=0).to_numpy()
    def calc_td_sim(self, X, Y=None):
        # matrix is return from calc_td_mtx()
        return cosine_similarity(X, Y)
    def write_diskcache_td_sim(self, filename, sim_matrix):
        # matrix is return from calc_td_sim()
        return pd.DataFrame(sim_matrix).to_csv(filename)

            
    def load_corpus_tfidf(self):
        if self.v_meth['tfidf']['corpus'] is not None:
            return
        if self.v_meth['td']['corpus'] is not None:
            self.v_meth['tfidf']['corpus'] = self.v_meth['tfidf']['corpus']
        self.load_corpus('tfidf')
    def doc_list_to_corpus_tfidf(self, doc_list):
        return doc_list
    def read_diskcache_tfidf_mtx(self, filename):
        sm = scipy.io.mmread(filename)
        return sm.shape[0], sm
    def init_tfidf_vectorizer(self, **kwargs):
        return TfidfVectorizer(**kwargs)
    def calc_tfidf_mtx(self, vectorizer):
        return vectorizer.vectorizer.fit_transform(self.v_meth['tfidf']['corpus'])
    def transform_tfidf_vec(self, vectorizer, vector_text):
        vector = raw_text_to_words(vector_text)
        return vectorizer.vectorizer.transform([vector])
    def write_diskcache_tfidf_mtx(self, filename, matrix):
        scipy.io.mmwrite(filename, matrix)
    def to_cache_tfidf_mtx(self, matrix):
        return matrix # already sparse
    def from_cache_tfidf_mtx(self, matrix):
        return matrix # already sparse
    def read_diskcache_tfidf_sim(self, filename):
        return pd.read_csv(filename, index_col=0).to_numpy()
    def calc_tfidf_sim(self, X, Y=None):
        return cosine_similarity(X,Y)
    def write_diskcache_tfidf_sim(self, filename, sim_matrix):
        return pd.DataFrame(sim_matrix).to_csv(filename)
    
    def load_corpus_d2v(self):
        self.load_corpus('doc2vec')
    def doc_list_to_corpus_d2v(self, doc_list):
        start = dt.datetime.now()
        print("starting to build Doc2Vec corpus from doclist file")
        corpus = []
        for index, doc in enumerate(doc_list):
            # doc needs to be a word list
            corpus.append(TaggedDocument(words=doc, tags=[index]))
        print(f"build Doc2Vec corpus took {dt.datetime.now() - start}")
        return corpus
    def read_diskcache_d2v_mtx(self, filename):
        vec_list = self.from_cache_d2v_mtx(scipy.io.mmread(filename))
        return len(vec_list), vec_list
    def init_d2v_vectorizer(self, **kwargs):
        return Doc2Vec(self.v_meth['doc2vec']['corpus'], **kwargs)
    def calc_d2v_mtx(self, vectorizer):
        vec_list = []
        for index, doc in enumerate(self.v_meth['doc2vec']['corpus']):
            if index % 100 == 0:
                print(index)
            vec_list.append(vectorizer.vectorizer.infer_vector(doc.words))
        return vec_list
    def transform_d2v_vec(self, vectorizer, vector_text):
        vector = raw_text_to_word_list(vector_text)
        return vectorizer.vectorizer.infer_vector(vector)
    def write_diskcache_d2v_mtx(self, filename, matrix):
        sparse_matrix = self.to_cache_d2v_mtx(matrix)
        scipy.io.mmwrite(filename, sparse_matrix)
    def to_cache_d2v_mtx(self, matrix):
        return scipy.sparse.csr_matrix(np.array(matrix))
    def from_cache_d2v_mtx(self, matrix):
        df = pd.DataFrame(matrix.toarray())
        vector_list = []
        for index, row in df.iterrows():
            vector_list.append(row.to_numpy())
        return vector_list
    def read_diskcache_d2v_sim(self, filename):
        return pd.read_csv(filename, index_col=0).to_numpy()
    def calc_d2v_sim(self, X, Y=None):
        return cosine_similarity(np.array(X, dtype='float32'), [Y])
    def write_diskcache_d2v_sim(self, filename, sim_matrix):
        return pd.DataFrame(sim_matrix).to_csv(filename)

 #    def load_corpus_txtrnk(self):
 #        self.load_corpus('txtrnk')
 #    def doc_list_to_corpus_txtrnk(self, doc_list):
 #        return doc_list
 #    def read_diskcache_txtrnk_mtx(self, filename):
 #        return self.corpus_size, None
 #    def init_txtrnk_vectorizer(self, **kwargs):
 #        return None
 #    def calc_txtrnk_mtx(self, vectorizer):
 #        return self.v_meth['txtrnk']['corpus']
 #    def transform_txtrnk_vec(self, vectorizer, vector_text):
 #        return vector_text
 #    def write_diskcache_txtrnk_mtx(self, filename, matrix):
 #        return None
 #    def to_cache_txtrnk_mtx(self, matrix):
 #        return matrix
 #    def from_cache_txtrnk_mtx(self, matrix):
 #        return matrix
 #    def read_diskcache_txtrnk_sim(self, filename):
 #        return pd.read_csv(filename, index_col=0).to_numpy()
 #    def calc_txtrnk_sim(self, X, Y=None):
 #        if Y is not None:
 #            X.append(Y)
 #            return summarize(X, ratio=1.0, split=True)[-1][0:X.shape[0]]
 #        else:
 #            return summarize(X, ratio=1.0, split=True)
 #    def write_diskcache_txtrnk_sim(self, filename, sim_matrix):
 #        return pd.DataFrame(sim_matrix).to_csv(filename)
    
            
 #    def load_corpus_bert(self):
 #        self.load_corpus('bert')
 #    def doc_list_to_corpus_bert(self, doc_list):
 #        # each doc in doc_list is a sentence list
 #        print_time(f"doc_list size {len(doc_list)}")
 #        if len(doc_list) > 0:
 #            print(f"first doc has size {len(doc_list[0])}")
 #            if len(doc_list[0]) > 5:
 #                print(doc_list[0][:5])
 #        return doc_list
 #    def read_diskcache_bert_mtx(self, filename):
 #        vec_list = self.from_cache_bert_mtx(scipy.io.mmread(filename))
 #        return len(vec_list), vec_list
 #    def init_bert_vectorizer(self, **kwargs):
 # #       return ContextDocEmbed()
 #        pass
 #    def calc_bert_mtx(self, vectorizer):
 #        msize = 0
 #        if not os.path.exists(vectorizer.mtx_filename):
 #            vec_list = []
 #        else:
 #            msize, vec_list = self.read_diskcache_bert_mtx(vectorizer.mtx_filename)
 #            print_time(f"msize {msize}")
 #        for index, doc in enumerate(self.v_meth['bert']['corpus']):
 #            if index < msize:
 #                continue
 #            if index == 20 or index % 100 == 0:
 #                print_time(index)
 #                print(len(doc))
 #                if len(doc) > 0 and len(doc[0]) > 0:
 #                    print(doc[0])
 #                self.write_diskcache_bert_mtx(vectorizer.mtx_filename, vec_list)
 #                self.v_meth['bert']['mtx'] = vec_list
 #            vec_list.append(vectorizer.vectorizer.get_doc_embed_sbert_from_sent_list(doc))
 #        return vec_list
 #    def transform_bert_vec(self, vectorizer, vector):
 #        return vectorizer.vectorizer.get_doc_embed_sbert(vector)
 #    def write_diskcache_bert_mtx(self, filename, matrix):
 #        sparse_matrix = self.to_cache_bert_mtx(matrix)
 #        scipy.io.mmwrite(filename, sparse_matrix)
 #    def to_cache_bert_mtx(self, matrix):
 #        if len(matrix) > 0:
 #            print_time(len(matrix))
 #        return scipy.sparse.csr_matrix(np.array(matrix))
 #    def from_cache_bert_mtx(self, matrix):
 #        df = pd.DataFrame(matrix.toarray())
 #        vector_list = []
 #        for index, row in df.iterrows():
 #            vector_list.append(row.to_numpy())
 #        return vector_list
 #    def read_diskcache_bert_sim(self, filename):
 #        return pd.read_csv(filename, index_col=0).to_numpy()
 #    def calc_bert_sim(self, X, Y=None):
 #        return cosine_similarity(np.array(X, dtype='float32'), [Y])
 #    def write_diskcache_bert_sim(self, filename, sim_matrix):
 #        return pd.DataFrame(sim_matrix).to_csv(filename)
        
    def get_vectorizer(self, method, **kwargs):
        corpus_vectorizer = None
        new_corpus_vectorizer = None
        
        self.v_meth[method]['load']()
        
        x = '_'.join([f"{key}={str(val)}"
                      for key, val in sorted(kwargs.items())])
        
        if len(kwargs) > 0:
            cache_id = method + '_'+ x
        else:
            cache_id = method

        vectorizer_filename = self.obj_filename.format(cache_id)

        if cache_id in self.v_meth[method]['vec']:
            corpus_vectorizer = self.v_meth[method]['vec'][cache_id]

        if corpus_vectorizer is None:
            if use_zip and os.path.exists(self.zipfilename):
                with zf.ZipFile(self.zipfilename, "r") as myzip:
                    if vectorizer_filename in myzip.namelist():
                        myzip.extract(vectorizer_filename)
                        with open(vectorizer_filename, "rb") as fp:
                            corpus_vectorizer = pickle.load(fp)
                        os.remove(vectorizer_filename)
            elif os.path.exists(vectorizer_filename):
                with open(vectorizer_filename, "rb") as fp:
                    corpus_vectorizer = pickle.load(fp)
                os.remove(vectorizer_filename)
                            
        if corpus_vectorizer is None:
            new_corpus_vectorizer = CorpusVectorizer(
                self.v_meth[method]['ivfunc'](**kwargs), cache_id, "", self, 
                **kwargs)
        else:
            new_corpus_vectorizer = CorpusVectorizer(
                corpus_vectorizer.vectorizer, corpus_vectorizer.cache_id, 
                corpus_vectorizer.transformed_cache_id, self, **kwargs) 
            
        if self.max_memory_vectorizer_cache > 0:
            while self.max_memory_vectorizer_cache < len(self.v_meth[method]['vec']):
                self.v_meth[method]['vec'].\
                    pop(next(iter(self.v_meth[method]['vec'])))
            self.v_meth[method]['vec'][cache_id] = new_corpus_vectorizer
            
        return new_corpus_vectorizer

def read_preprocess_corpus(corpus_filename):

    corpus_filename = corpus_filename
    
    
    corpus_df = pd.read_csv(corpus_filename)
    corpus_df = corpus_df[corpus_df['full_text'].notna()] # must be done before reset_index
    if corpus_df.shape[0] == len(corpus_df.title.unique()):
        print("No dupliate titles in the data.")
        corpus_df['id'] = range(1, len(corpus_df) + 1)
    else:
        print("Dupliate titles in the data!")
    
    print(corpus_df.source_id.nunique(), corpus_df.title.nunique())
    corpus_df.dropna(subset=['full_text'], inplace=True)
    print("Shape after removing null text ones: ", corpus_df.shape)
    corpus_df.head()
    return corpus_df

def test_split(corpus_df, target_paper_num, random_seed):
    
    random.seed(random_seed)
    target_ids = random.sample(corpus_df['id'].tolist(), target_paper_num)
    target_ids # [1825, 410, 4508, 4013, 3658]
    
    tgt_df = pd.DataFrame(columns=corpus_df.columns)
    for target in target_ids:
        s_df = corpus_df[corpus_df['id'] == target]
        tgt_df.loc[len(tgt_df.index)] = s_df.iloc[0]
        corpus_df.drop(s_df.index[0], inplace=True)
    tgt_df
    
    corpus_df.reset_index(drop=True, inplace=True)
    print(corpus_df.shape)
    tgt_df.reset_index(drop=True, inplace=True)
    
    return corpus_df, tgt_df

def split_preprocess_corpus(corpus_filename, random_seed):
    return test_split(read_preprocess_corpus(corpus_filename), 5, 42)
