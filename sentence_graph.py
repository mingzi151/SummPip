# -*- coding: utf-8 -*-
"""
Created on 2020-01-16 10:17 PM

author  : michelle
"""

import numpy as np
import gensim.downloader as api
import spacy
from nltk.corpus import wordnet as wn
from orderedset import OrderedSet
import scipy
from scipy import *


glove_word_vectors = api.load("glove-wiki-gigaword-100")
spacynlp=spacy.load("en_core_web_sm")

# Step 1: Deverbal Noun Reference
# step 1.1: get nouns for verbs in the current sentence
verbs_to_escape = ["be", "is","am","are","was", "were", "being","been","do","did",
               "done","have","had","get","got","gotten"]

# Step 3: Discourse Markers => only for two adjacent sentences
markers=["for","so","because","since","therefore","consequently","additionally","furthermore","moreover",
         "but","however","although","despite","similarly","otherwise","whereas","while","unlike","thus",
        "instead","nevertheless","afterward","finally","subsequently","conversely","later","next","then",
         "likewise","compared","besides","further","as","also","equally","hence","accordingly","stil",
        "simultaneously"]
# 39 markers

class SentenceGraph:
    def __init__(self, sentences_list, w2v, ita=0.9, threshold=0.65):
        self.sentences_list = sentences_list

        self.length = len(sentences_list)

        self.w2v = w2v

        # threshold for step1
        self.threshold = threshold

        # threshold for step4
        self.ita = ita

    def get_nouns_for_verbs(self, string):
        doc = spacynlp(string)
        nouns_list = []
        if len(doc)>0:
            for token in doc:
                # find noun reference for verbs, escaping verbs that are too ambiguous
                if token.pos_ == "VERB" and token.text not in verbs_to_escape:
                    # print("token.text ", token.text)
                    noun_forms = self._nounify(token.text)
                    nouns_list.extend(noun_forms)
        return nouns_list

    def _nounify(self, verb):
        # get the lemmas of base verbs;
        base = wn.morphy(verb, wn.VERB)
        if base:
            lemmas = wn.lemmas(base, pos="v")
            noun_forms = []
            # derive noun forms for each lemma
            for lemma in lemmas:
                nouns = [forms.name() for forms in lemma.derivationally_related_forms()]
                noun_forms.extend(nouns)
            # remove repetition
            nouns_set = OrderedSet(noun_forms)
            return nouns_set
        else:
            return []

    # step 1.2: find most similar word from word2vec
    # get most similar words for nouns, including itself
    def find_most_similar_words(self, word_vectors,nouns_list,threshold=0.65):
        similar_nouns_list=[]
        nouns_list=list(set(nouns_list))
        for noun in nouns_list:
            try:
                nn = word_vectors.most_similar(positive=[noun])
                # keep nn whose have high similary score
                nn = [ tuple_[0] for tuple_ in nn if tuple_[1] > threshold]
                similar_nouns_list.extend(nn)
            # pass on uncommon words
            except KeyError:
                pass
        similar_nouns_list.extend(nouns_list)
        return list(set(similar_nouns_list))

    # check if deverbal noun reference exits in the subsequent sentence
    def check_noun_reference(self, similar_nouns_list, subsequent_sen):
        flag=False
        doc = spacynlp(subsequent_sen)
        if len(doc)>0:
            for token in doc:
                if token.pos_ == "NOUN":
                    if token.text in similar_nouns_list:
                        flag=True
                        break
        return flag

    # step 2: Event/Entity Continuation
    # Str needs to be raw, i.e., use str before normalisation and stemming
    def compare_name_entity(self, str1, str2):
        flag = False
        doc1 = spacynlp(str1)
        doc2 = spacynlp(str2)
        if len(doc1)>0 and len(doc2)>0:
            ent_list1=[(ent.text, ent.label_) for ent in doc1.ents]
            ent_list2=[(ent.text, ent.label_) for ent in doc2.ents]
            for (text, label) in ent_list1:
                if (text, label) in ent_list2:
                    flag=True
                    break
        return flag

    def check_discourse_markers(self, str1,str2):
        flag = False
        doc2 = spacynlp(str2)
        if len(doc2)>0:
            first_token = doc2[0].text
            if first_token.lower() in markers:
                flag = True
        return flag

    # compute the cos similarity between a and b. a, b are numpy arrays
    def cos_sim(self, a, b):
        return 1 - scipy.spatial.distance.cosine(a,b)


    def make_graph_undirected(self, source, target, weight):
        source.extend(target)
        target.extend(source)
        weight.extend(weight)
        triplet_list=[ (source[i],target[i],weight[i]) for i in range(len(source))]
        sorted_by_src = sorted(triplet_list, key=lambda x: (x[0],x[1]))

        sorted_source = []
        sorted_target = []
        sorted_weight = []
        for triplet in sorted_by_src:
            sorted_source.append(triplet[0])
            sorted_target.append(triplet[1])
            sorted_weight.append(triplet[2])
        return sorted_source, sorted_target, sorted_weight

    # Step4: calculate sentence embeddings
    def get_sentence_embeddings(self,string):
        v = self.get_wv_embedding(string)
        return v

    # get sentence embeddings with w2v
    def get_wv_embedding(self, string):
        word_embeddings = self.w2v
        sent = string.lower()
        eps = 1e-10
        if len(sent) != 0:
            vectors = [word_embeddings.get(w, np.zeros((100,))) for w in sent.split()]
            v = np.mean(vectors, axis=0)
        else:
            v = np.zeros((100,))
        v = v + eps
        return v

    # step 4: compare sentence similarity
    def check_if_similar_sentences(self,sentence_emb1,sentence_emb2):
        flag = False
        similarity = cos_sim(sentence_emb1,sentence_emb2)
        if similarity > self.ita:
            flag = True
        return flag


    def build_sentence_graph(self,):
        # spectral clustering  
        X = np.zeros([self.length, self.length])
        # iterate all sentence nodes to check if they should be connected
        counter=0
        for i in range(self.length):
            flag = False
            sen_i = self.sentences_list[i]
            # check above steps
            for j in range(i+1,self.length):
                sen_j = self.sentences_list[j]
                if (j-i) == 1:
                    # perform step1 and step3,which are only for adjacent sentences
                    nouns_list = self.get_nouns_for_verbs(sen_i)
                    # get most similar words for above nouns, including itself
                    similar_nouns_list = self.find_most_similar_words(glove_word_vectors, nouns_list,self.threshold)
                    # check for devebal noun 
                    flag = self.check_noun_reference(similar_nouns_list, sen_j)
                    if not flag:
                        # check for disourse markers
                        flag=self.check_discourse_markers(sen_i,sen_j)
                else:
                    # check for name entities
                    flag=self.compare_name_entity(sen_i,sen_j)

               # => step4 check for similar sentences
                if not flag:
                    continue
                    i_sen_emb = emb_sentence_vectors[i,]
                    j_sen_emb = emb_sentence_vectors[j,]
                    flag = self.check_if_similar_sentences(i_sen_emb,j_sen_emb)

                if flag:  
                    counter += 1                                      
                    X[i,j] = 1
                    X[j,i] = 1
        return X

