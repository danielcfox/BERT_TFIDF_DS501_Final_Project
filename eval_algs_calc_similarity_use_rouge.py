#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 09:12:10 2023

@author: dfox
"""

# USE_BERT = False

import datetime as dt
start = dt.datetime.now()
def print_time(text):
    print(f"{text}: elapsed time {dt.datetime.now() - start}")
    
import corpus_vectorizer as cv
# from datasets import load_metric
import pandas as pd
import time
from rouge import Rouge

rouge = Rouge()
# rouge = load_metric("rouge")

random_seed = 42

corpus_df, tgt_df = cv.split_preprocess_corpus("papers.csv", random_seed)

# test_df = corpus_df.sample(1, random_state=random_seed)
# print(test_df.index[0])
# test_df.reset_index(drop=True, inplace=True)

print("test documents:")
print(tgt_df[['id', 'title']])

this_corpus = cv.CorpusOps(f"NIPS_papers2_{random_seed}_{corpus_df.shape[0]}",
                           corpus_df)

for index, row in tgt_df.iterrows():
    print("")
    print("")
    print_time("")
    print(f"Running similarity for document id {row['id']}")
    print(f"Title: {row['title']}")
    for method in ['tfidf', 'doc2vec']:
        print("")
        print_time("")
        print(f"Running similarity with algorithm {method}:")
        res_df = pd.DataFrame(columns=['rank', 'id', 'title', 'sim', 'rouge1',
                                       'rouge2', 'rougeL'
                                       # 'rougeLsum'
                                       ])
        
        td_params = {'min_df' : 20, 'max_features': 100000}
        
        if method == 'doc2vec':
            vectorizer = this_corpus.get_vectorizer(method)
        else:
            vectorizer = this_corpus.get_vectorizer(method, **td_params)
        matrix = vectorizer.get_matrix()
        sim_matrix = matrix.get_similarity(row['full_text'])        
        sim_list = list(sim_matrix.sim_matrix)
        print(max(sim_list))
        max_indexes = sorted(
            range(len(sim_list)), key=lambda i: sim_list[i], reverse=True)[:10]
        res_df.loc[0] = [0, row['id'], row['title'], 1.0, 1.0, 1.0, 1.0]
        accum_f1_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,
                           # 'rougeLsum': 0.0,
                           }
        for i in max_indexes:
            f1_scores = {}
            # rouge_scores = rouge.compute(
            #     references=[row['full_text']], 
            #     predictions=[this_corpus.cdf.at[i, 'full_text']])
            # f1_scores['rouge1'] = rouge_scores['rouge1'].mid.fmeasure
            # f1_scores['rouge2'] = rouge_scores['rouge2'].mid.fmeasure
            # f1_scores['rougeL'] = rouge_scores['rougeL'].mid.fmeasure
            # f1_scores['rougeLsum'] = rouge_scores['rougeLsum'].mid.fmeasure
            rouge_scores = rouge.get_scores(
                refs=[row['full_text']], 
                hyps=[this_corpus.cdf.at[i, 'full_text']])
            f1_scores['rouge1'] = rouge_scores[0]['rouge-1']['f']
            f1_scores['rouge2'] = rouge_scores[0]['rouge-2']['f']
            f1_scores['rougeL'] = rouge_scores[0]['rouge-l']['f']
            print_time("")
            print(f"sim={sim_list[i]}", 
                  f"rouge1={f1_scores['rouge1']}",
                  f"rouge2={f1_scores['rouge2']}", 
                  f"rougeL={f1_scores['rougeL']}", 
                  # f"rougeLsum={f1_scores['rougeLsum']}", 
                  this_corpus.cdf.at[i, 'id'], this_corpus.cdf.at[i, 'title'])
            res_df.loc[len(res_df.index)] = [i+1, this_corpus.cdf.at[i, 'id'],
                                             this_corpus.cdf.at[i, 'title'], 
                                             sim_list[i], f1_scores['rouge1'],
                                             f1_scores['rouge2'],
                                             f1_scores['rougeL']
                                             # f1_scores['rougeLsum']
                                             ]
            accum_f1_scores['rouge1'] += f1_scores['rouge1']
            accum_f1_scores['rouge2'] += f1_scores['rouge2']
            accum_f1_scores['rougeL'] += f1_scores['rougeL']
            # accum_f1_scores['rougeLsum'] += f1_scores['rougeLsum']
        # test_id = str(hash(test_df.at[0, 'full_text']))
        res_df.loc[len(res_df.index)] = [-1, -1, 'AVERAGE', 0.00, 
                                         accum_f1_scores['rouge1']/10.0,
                                         accum_f1_scores['rouge2']/10.0,
                                         accum_f1_scores['rougeL']/10.0
                                         # accum_f1_scores['rougeLsum']/10.0
                                         ]
        if method != 'doc2vec':
            x = '_'.join([f"{key}={str(val)}"
                          for key, val in sorted(td_params.items())])
        else:
            x = ""
        res_df.to_csv(f"{method}_{x}_similarity_{row['id']}_{str(time.time())}"
                      + "_results.csv")

print_time("finished")
        # res_df.to_csv(f"{method}_{x}_similarity_{test_hash}_{str(time.time())}"
        #               + "_results.csv")
    
    # full_text_list = this_corpus.cdf['full_text'].tolist()
    # f1_scores = []
    # for ref_text in full_text_list:
    #     rouge_scores = rouge.compute(predictions=[test_df.at[0, 'full_text']], 
    #                                  references=[ref_text])
    #     f1_scores.append(rouge_scores['rougeLsum'].mid.fmeasure)
    
    # print(max(f1_scores))
    
    # max_rouge_indexes = sorted(range(len(sim_list)), 
    #                             key=lambda i: sim_list[i], reverse=True)[:11]
        
    # for i in max_rouge_indexes:
    #     print(f"rouge={f1_score}", f"sim={sim_list[i]}", i, 
    #           this_corpus.cdf.at[i, 'id'], this_corpus.cdf.at[i, 'title'])
    