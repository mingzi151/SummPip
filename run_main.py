# -*- coding: utf-8 -*-
"""
Created on 2020-01-17 11:17 AM

author  : michelle
"""


from utils import *
from SentenceGraph import *
import timeit
from datetime import date
import argparse
import math
import pprint
import copy
import math

def read_arguments():
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="dataset/multi_news/")
    parser.add_argument("--source_file", type=str, default="test.truncate.fix.pun.src.txt")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--w2v_file", type=str, default="word_vec/multi_news/news_w2v.txt")
    parser.add_argument("--mode", type=str, default="SummPip")
    parser.add_argument("--cluster", type=str, default="spc")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--output_file", type=str, default="Summary.txt")
    parser.add_argument("--nb_words", type=int, default=5)
    parser.add_argument("--use_lm", action="store_true")
    parser.add_argument("--nb_clusters", type=int, default=9,help="for spectral clustering")
    parser.add_argument("--ita",type=float,default=0.98)

    return parser.parse_args()

def main():
    
    # read arguments
    args=read_arguments() 
    print("arguments",args)
    nb_words = args.nb_words
    nb_clusters = args.nb_clusters
    seed = args.seed
    path = args.input_path
    clus_alg = args.cluster
    ita = args.ita

    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    #read source file
    src_file = args.source_file
    src_list = read_file(path, src_file)
    summary_list = []
    start = timeit.default_timer()

    # run SummPip
    if args.mode == "SummPip":
        #load the LM or word vectors
        if args.use_lm:
            #load the LM, tokenizer
            model_path = args.lm_path
            lm_tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            lm_model = GPT2Model.from_pretrained(model_path,
                                          output_hidden_states=True,
                                          output_attentions=False)
            w2v = ''
        else:
            #load word vectors
            lm_model = ''
            lm_tokenizer = ''
            # get word vectors
            w2v_file = args.w2v_file
            if w2v_file != "":
                w2v = get_w2v_embeddings(w2v_file)

        # iterate over all docs
        for idx, sentences_list in enumerate(src_list[:12]):
            num_sents = len(sentences_list)
            print("processing {}th item---------".format(idx))
            print("number of sentences:", num_sents)
            # handle short doc
            if num_sents <= nb_clusters:
                summary_list.append(" ".join(sentences_list))
                print("continue----")
                continue


            # if num_sents < nb_clusters:
            #     nb_clusters = num_sents

            summary = []
            # build sentence graph
            senGraph=SentenceGraph(sentences_list, w2v, lm_model, lm_tokenizer, cluster_algo=clus_alg,ita=ita)
            # perform graph clustering
            if clus_alg == "spc":
                # method2: spectral clustering
                X = senGraph.build_sentence_graph()
                clustering = SpectralClustering(n_clusters=nb_clusters, random_state=seed).fit(X)
                clusterIDs = clustering.labels_

            num_clusters = max(clusterIDs)+1
            text_dict={new_list:[] for new_list in range(num_clusters)}
            # group sentences by cluster ID
            for i, clusterID in enumerate(clusterIDs):
                text_dict[clusterID].append(sentences_list[i])

            # perform cluster compression for each cluster IDs
            for k,v in text_dict.items():
                tagged_sens = convert_sents_to_tagged_sents(v)
                compressed_sent = get_compressed_sen(tagged_sens, nb_words)
                summary.append(compressed_sent)
            summary_list.append(' '.join(summary))

    # write output to file
    stop = timeit.default_timer()
    out_path = args.output_path
    outfile = args.output_file
    f = open(os.path.join(out_path, outfile), "w")
    print("summary list length",len(summary_list))
    summary_list = [line.replace("\n","") +"\n" for line in summary_list]
    f.writelines(summary_list)
    f.close()
    print('Done')

if __name__ == "__main__":
    main()
