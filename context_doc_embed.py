from nltk.tokenize import sent_tokenize
import nltk.data
from gensim.parsing.preprocessing import (
    remove_stopwords,
    strip_tags,
    strip_punctuation,
    strip_numeric,
    strip_non_alphanum,
    strip_multiple_whitespaces,
    strip_short,
    preprocess_string,
)
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from numpy.linalg import norm
import pickle
import enchant
from math import log
import spacy
import pytextrank
from rouge import Rouge


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


def norm_sentence(
    sent: str,
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
        strip_multiple_whitespaces,
        strip_short2,
    ]  # remove_stopwords,
    normed_words = preprocess_string(sent, norm_sent_steps)
    if only_english:
        normed_words = [x for x in normed_words if eng_dict.check(x)]
    if do_lemma:
        # If do lemmatization, need to do pos tagging first.
        # Here the POS tagging part is not added. Can use pos_tag from nltk.tag
        normed_words = [lemmatizer.lemmatize(x) for x in normed_words]

    return " ".join(normed_words) + "."


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
    text: str, to_remove_sentence: list = TO_REMOVE_SENT
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
    normed_sents = [norm_sentence(x) for x in sents]
    normed_sents = [x for x in normed_sents if not x in to_remove_sentence]
    return normed_sents


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def merge_pickles(file_path_list: list, new_file_path: str):
    """_summary_
    This func is not general enough, but this is not needed at runtime.
    Args:
        file_path_list (list): _description_
        new_file_path (str): _description_
    """
    file_list = []
    for a_file_path in file_path_list:
        with open(a_file_path, "rb") as pkl:
            a_file = pickle.load(pkl)
            file_list.extend(a_file["paper"])
    with open(new_file_path, "wb") as pkl:
        pickle.dump(file_list, pkl)


def get_most_similar_docs_from_list(
    target_id: int,
    embed_list: list,
    target_embed: np.ndarray = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """_summary_
    This func is used for NIPS data set. It can also apply to other data
    set after changing the data format as required.
    The expect
    Args:
        target_id (int): The target document id
        embed_list (pd.DataFrame): A list of dictionaries where each
            dictionary contains doc 'id' and document embedding 'doc_embed'
            keys.
        target_embed (np.ndarray, optional): Target doc embedding. This
            is optional. However if the target document is not from the
            provided 'embed_list', this parameter is needed so that
            similarity can be measure.
        top_n (int, optional): Num of most similar docs to return.
            Defaults to 10.

    Raises:
        Exception: _description_

    Returns:
        pd.DataFrame: A data frame that contains the top K most similar
            docs.
    """
    embed_df = pd.DataFrame(embed_list)
    embed_df.drop(columns="doc_embed", inplace=True)
    if (target_id not in embed_df["id"]) & (target_embed == None):
        raise Exception(
            "Either 'id' needs to be in the list or provide target embedding!"
        )

    most_sim_df = embed_df.loc[embed_df["id"] != target_id].copy()
    most_sim_df["cosine_similarity"] = np.NaN
    if target_embed is None:
        target_index = embed_df.loc[embed_df["id"] == target_id].index.item()
        target_embed = embed_list[target_index]["doc_embed"]
    for i, a_row in most_sim_df.iterrows():
        row_index = embed_df.loc[embed_df["id"] == a_row["id"]].index.item()
        a_doc_embed = embed_list[row_index]["doc_embed"]
        if len(a_doc_embed) > 0:  # non-empty embedding
            most_sim_df.at[i, "cosine_similarity"] = cosine_similarity(
                target_embed, a_doc_embed
            )
    return most_sim_df.sort_values(by="cosine_similarity", ascending=False).iloc[
        0:top_n
    ]


## Neet to VM, cluster or Google Colab for using the following code
# class ContextDocEmbed:
#     def __init__(
#         self, sbert_model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
#     ) -> None:
#         self.sbert_model = sbert_model

#     def get_doc_embed_sbert(
#         self,
#         text: str,
#     ) -> np.array:
#         """_summary_
#         Get document embedding from mean of sentence embeddings through SBERT
#         May need to virtual machine or google Colab.
#         Args:
#             text (str): _description_
#             sbert_model (SentenceTransformer): SBert model

#         Returns:
#             np.array: doc embedding
#         """
#         sentences = get_norm_sentences_from_text(text)
#         sentence_embed = [self.sbert_model.encode(x) for x in sentences]
#         doc_embed = pd.DataFrame(sentence_embed).apply(np.mean, axis=0).to_numpy()
#         return doc_embed


nlp = spacy.load("en_core_web_sm")
# add PyTextRank to the spaCy pipeline
nlp.max_length = 1500000
nlp.add_pipe("textrank")


def get_top_keywords(text: str, nlp: spacy.lang.en.English = nlp, top_n: int = 20):
    doc = nlp(text)
    key_words = list()
    rank_scores = list()
    for phrase in doc._.phrases[0:top_n]:
        key_words.append(phrase.text)
        rank_scores.append(phrase.rank)
    return {"key_word": key_words, "rank_score": rank_scores}


def get_word_count(text):
    return len(text.split())


def get_similarity_textrank(text1: str, text2: str, top_n: int = 20):
    text1_keywords = get_top_keywords(text=text1, top_n=top_n)["key_word"]
    text2_keywords = get_top_keywords(text=text2, top_n=top_n)["key_word"]
    text1_len = get_word_count(text1)
    text2_len = get_word_count(text2)
    common_words = list(set(text1_keywords) & set(text2_keywords))
    return {
        "similarity": len(common_words) / (log(text1_len, 10) + log(text2_len, 10)),
        "common_key": common_words,
        "text1_keywords": text1_keywords,
        "text2_keywords": text2_keywords,
    }


def get_similarity_using_kwords(
    text1_keywords: list,
    text2_keywords: list,
    text1_len: int,
    text2_len: int,
):
    common_words = list(set(text1_keywords) & set(text2_keywords))
    return {
        "similarity": len(common_words) / (log(text1_len, 10) + log(text2_len, 10)),
        "common_key": common_words,
        "text1_keywords": text1_keywords,
        "text2_keywords": text2_keywords,
    }


def get_most_similar_docs_from_list_textrank(
    target_id: int,
    keyword_info_list: list,
    target_text: str = None,
    top_n: int = 10,
    top_n_keywords: int = 30,
) -> pd.DataFrame:
    """_summary_
    This func is used for NIPS data set. It can also apply to other data
    set after changing the data format as required.
    This func uses TextRank method
    Args:
        target_id (int): The target document id
        keyword_info_list (list): A list of dictionaries where each
            dictionary contains doc 'id', document text, key word information,
            etc.
        target_text (str, optional): Target doc text. This
            is optional. However if the target document is not from the
            provided 'keyword_info_list', this parameter is needed so that
            similarity can be measure.
        top_n (int, optional): Num of most similar docs to return.
            Defaults to 10.
        top_n_keywords (int, optional): Num of keywords considered during
            similarity calculation.

    Raises:
        Exception: _description_

    Returns:
        pd.DataFrame: A data frame that contains the top K most similar
            docs.
    """
    keyword_df = pd.DataFrame(keyword_info_list)
    if (target_id not in keyword_df["id"]) & (target_text == None):
        raise Exception("Either 'id' needs to be in the list or provide target text!")

    most_sim_df = keyword_df.loc[keyword_df["id"] != target_id].copy()
    most_sim_df["similarity"] = np.NaN
    if target_text is None:
        target_index = keyword_df.loc[keyword_df["id"] == target_id].index.item()
        target_keywords = keyword_info_list[target_index]["key_words"][0:top_n_keywords]
        target_len = keyword_info_list[target_index]["doc_len"]
        print(keyword_info_list[target_index]["title"])
    for i, a_row in most_sim_df.iterrows():
        row_index = keyword_df.loc[keyword_df["id"] == a_row["id"]].index.item()
        a_doc_keywords = keyword_info_list[row_index]["key_words"][0:top_n_keywords]
        a_doc_len = keyword_info_list[row_index]["doc_len"]
        if len(a_doc_keywords) > 0:  # non-empty embedding
            most_sim_df.at[i, "similarity"] = get_similarity_using_kwords(
                target_keywords, a_doc_keywords, target_len, a_doc_len
            )["similarity"]
    return most_sim_df.sort_values(by="similarity", ascending=False).iloc[0:top_n]


def get_rouge_metrics(
    candi_text: str, reference: str, rouge: Rouge = Rouge()
) -> list:
    return rouge.get_scores(candi_text, reference)


def get_mean_rouge(
    candidate_df: pd.DataFrame,
    reference: str,
):
    rouge1_list = list()
    rougel_list = list()
    for a_text in candidate_df["full_text"]:
        rouge_info = get_rouge_metrics(a_text, reference)
        rouge1_list.append(rouge_info[0]["rouge-1"]["f"])
        rougel_list.append(rouge_info[0]["rouge-l"]["f"])
    return {
        "mean_rouge1_f": np.mean(rouge1_list),
        "mean_rougel_f": np.mean(rougel_list),
    }
