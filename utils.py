# -*- coding: utf-8 -*-
"""
Created on 2020-01-16 10:17 PM

author  : michelle
"""

import numpy as np
import os
import takahe
import nltk.data
import spacy

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
spacynlp=spacy.load("en_core_web_sm")

def rank_sent_Lexrank(lxr,sentences_list,cutoff=1.5):
    scores = list(lxr.rank_sentences(sentences_list,threshold=None,fast_power_method=True))
    # idx_to_slice=[sscores.index(s) for s in scores if s> cutoff]
    idx_to_slice=[ i for i, score in enumerate(scores) if score>cutoff]
    sentences_list=[sentences_list[idx] for idx in idx_to_slice]
    # print("ranked sentence:", sentences_list)
    return sentences_list

def rank_sent_centroid(centroid_rank,sentences_list):
    sentences_list=centroid_rank.centroidRank(sentences_list)
    return sentences_list

# get word embeddings
def get_w2v_embeddings(filename):
    word_embeddings = {}
    f = open(filename, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

def get_sentence_embedding(sent, word_embeddings):
    sent=sent.lower()
    eps=1e-10
    #print(sent)
    if len(sent) != 0:
        vectors = [word_embeddings.get(w, np.zeros((100,))) for w in sent.split()]
        v=np.mean(vectors, axis=0)
    else:
        v = np.zeros((100,))
    v = v + eps
    return v


def get_SentNode_embedding(sentences_list, word_embeddings):
    emb_sentence_vectors=np.zeros([len(sentences_list),100])
    for count, sent in enumerate(sentences_list):
        emb_sen=get_sentence_embedding(sent, word_embeddings)
        emb_sentence_vectors[count,]=emb_sen
    return emb_sentence_vectors

def build_similarity_matrix(emb_sentence_vectors):
    from sklearn.metrics.pairwise import cosine_similarity
    sim_mat = np.zeros([len(emb_sentence_vectors), len(emb_sentence_vectors)])
    for i in range(len(emb_sentence_vectors)):
        for j in range(len(emb_sentence_vectors)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(emb_sentence_vectors[i].reshape(1,100), emb_sentence_vectors[j].reshape(1,100))[0,0]
    return sim_mat

def rank_sent_Pagerank(sim_mat,sentences_list,n=3,alpha=0.85,tol=1.0e-6):
    import networkx as nx
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph,alpha=alpha,tol=tol)
    ranked_sent = sorted(((scores[i],s) for i,s in enumerate(sentences_list)), reverse=True)
    ranked_sentences_list = [ tuple_[1] for i, tuple_ in enumerate(ranked_sent)]
    return ranked_sentences_list

def get_first_doc(line, tag="|||||"):
    first_doc = line.split(tag)[0]
    return first_doc

def truncate_doc(doc):
    sent_list = sent_detector.tokenize(doc.strip())
    seg=6
    if len(sent_list)>seg:
        return sent_list[:seg]
    else:
        return sent_list

def read_lead_sentences(doc,tag="story_separator_special_tag"):
    doc_list = doc.split(tag)
    # print(doc_list)
    sent_list=[]
    # only keep the first three sources
    if len(doc_list) == 2:
        for count, doc in enumerate(doc_list):
            doc_sent_list = sent_detector.tokenize(doc.strip())
            try:
                sent_list += doc_sent_list[:4]
            except:
                sent_list += doc_sent_list
    elif len(doc_list) > 2:
        doc_list = doc_list[:3]
        for count, doc in enumerate(doc_list):
            doc_sent_list = sent_detector.tokenize(doc.strip())
            try:
                sent_list += doc_sent_list[:3]
            except:
                sent_list += doc_sent_list
    print(f"number of sources in this instance: {len(doc_list)}")    
    print(f"number of lead sentences in this instance: {len(sent_list)}")
    print("*"*80)
    return sent_list



# read source file into a list of list:
def read_file(path, file_name, read_lead_only=False, read_first_doc=False):
    f = open(os.path.join(path, file_name),"r")
    lines = f.readlines()
    src_list = []
    tag="story_separator_special_tag"
    for line in lines:
        if read_first_doc:
            line = get_first_doc(line)
            sent_list = truncate_doc(line)
        elif read_lead_only:
            sent_list = read_lead_sentences(line,tag=tag)
        else:
            # remove tag; uncomment below for baseline
            line = line.replace(tag, "")
            # tokenzie line to sentences
            sent_list = sent_detector.tokenize(line.strip())
        src_list.append(sent_list)
    return src_list

def tag_pos(str_text):
    doc=spacynlp(str_text)
    textlist=[]
    # compare the words between two strings
    for item in doc:
        source_token = item.text
        source_pos = item.tag_
        textlist.append(source_token+'/'+source_pos)
    return ' '.join(textlist)


def convert_sents_to_tagged_sents(sent_list):
    tagged_list = []
    if(len(sent_list)>0):
        for s in sent_list:
            s = s.replace("/", "")
            # print("original sent -------- \n",s)
            temp_tagged = tag_pos(s)
            tagged_list.append(temp_tagged)
    else:
        tagged_list.append(tag_pos("."))
    return tagged_list

def get_compressed_sen(sentences, nb_words):
    compresser = takahe.word_graph(sentences, nb_words = nb_words, lang = 'en', punct_tag = "." )
    candidates = compresser.get_compression(3)
    # print("--------------------Top 3 candicate---------------", candidates)
    reranker = takahe.keyphrase_reranker(sentences,
                                      candidates,
                                      lang = 'en')
    # print("reranker: ", reranker)
    # print("finish initialising reranker------------")

    reranked_candidates = reranker.rerank_nbest_compressions()
    # print(reranked_candidates)
    if len(reranked_candidates)>0:
        score, path = reranked_candidates[0]
        result = ' '.join([u[0] for u in path])
    else:
        result=' '
    # print("----------------selected candicate as final output-------------- ", result)
    return result
