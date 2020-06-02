from summarizer import SummPip
import timeit
from datetime import date
import argparse
from utils import read_file 
import os

def read_arguments():
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="dataset/multi_news/")
    parser.add_argument("--source_file", type=str, default="test.truncate.fix.pun.src.txt")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--w2v_file", type=str, default="word_vec/multi_news/news_w2v.txt")
    parser.add_argument("--cluster", type=str, default="spc")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--output_file", type=str, default="testSummary.txt")
    parser.add_argument("--nb_words", type=int, default=5)
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
    w2v_file = args.w2v_file
    ita = args.ita

    #read source file
    src_file = args.source_file
    src_list = read_file(path, src_file)[:2]
    print("Number of instances: ", len(src_list))
    start = timeit.default_timer()

    pipe = SummPip(nb_clusters=nb_clusters, nb_words=nb_words, ita=ita, seed=seed, w2v_file=w2v_file)
    summary_list = pipe.summarize(src_list)

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