
from sentence_graph import SentenceGraph
from utils import *
from sklearn.cluster import SpectralClustering
import torch
import nltk.data

class SummPip():

    def __init__(
        self, 
        nb_clusters: int = 14,
        nb_words: int = 5, 
        ita: float = 0.98,
        seed: int = 88,
        w2v_file: str = "word_vec/multi_news/news_w2v.txt",
        ):
        """
        This is the SummPip class

        :param nb_clusters: this determines the number of sentences in the output summary
        :param nb_words: this controls the length of each sentence in the output summary
        :param ita: threshold for determining whether two sentences are similar by vector similarity
        :param seed: the random state to reproduce summarization
        :param w2v_file: file for storing w2v matrix
        """

        self.nb_clusters = nb_clusters
        self.nb_words = nb_words
        self.ita = ita
        self.seed = seed
        self.w2v = self._get_w2v_embeddings(w2v_file)
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

        # set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _get_w2v_embeddings(self, w2v_file):
        """
		Get w2v word embedding matrix
		
		:return: w2v matrix
        """
        word_embeddings = {}
        f = open(w2v_file, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
        f.close()
        return word_embeddings

    def construct_sentence_graph(self, sentences_list):
        """
		Construct a sentence graph

		:return: adjacency matrix 
        """

        graph = SentenceGraph(sentences_list, self.w2v, self.ita)
        X = graph.build_sentence_graph()
        return X

    def cluster_graph(self, X, sentences_list):
        """
		Perform graph clustering

		:return: a dictionary with key, value pairs of cluster Id and sentences
        """
         # ???? n
        clustering = SpectralClustering(n_clusters = self.nb_clusters, random_state = self.seed).fit(X)
        clusterIDs = clustering.labels_

        num_clusters = max(clusterIDs)+1
        cluster_dict={new_list:[] for new_list in range(num_clusters)}
		# group sentences by cluster ID
        for i, clusterID in enumerate(clusterIDs):
            cluster_dict[clusterID].append(sentences_list[i])
        return cluster_dict

    def convert_sents_to_tagged_sents(self, sent_list):
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

    def get_compressed_sen(self, sentences):
        compresser = takahe.word_graph(sentence_list = sentences, nb_words = self.nb_words, lang = 'en', punct_tag = "." )
        candidates = compresser.get_compression(3)
        reranker = takahe.keyphrase_reranker(sentences, candidates, lang = 'en')

        reranked_candidates = reranker.rerank_nbest_compressions()
	    # print(reranked_candidates)
        if len(reranked_candidates)>0:
            score, path = reranked_candidates[0]
            result = ' '.join([u[0] for u in path])
        else:
            result=' '
        return result

    def compress_cluster(self, cluster_dict):
        """
		Perform cluster compression

		:return: a string of concatenated sentences from all clusters
        """

        summary = []
        for k,v in cluster_dict.items():
            tagged_sens = self.convert_sents_to_tagged_sents(v)
            compressed_sent = self.get_compressed_sen(tagged_sens)
            summary.append(compressed_sent)
        return " ".join(summary)

    def split_sentences(self, docs):
        tag="story_separator_special_tag"
        src_list = []
        for doc in docs:
            doc = doc.replace(tag,"")
            sent_list = self.sent_detector.tokenize(doc.strip()) 
            src_list.append(sent_list)
        return src_list



    def summarize(self, src_list):
        """
		Construct a graph, run graph clustering, compress each cluster, then concatenate sentences

		:param src_list: a list of input documents each of whose elements is a list of multiple documents
		:return: a list of summaries
        """
        #TODO: split sentences
        summary_list = []
        # iterate over all docs
        for idx, sentences_list in enumerate(src_list):
            num_sents = len(sentences_list)
			# handle short doc
            if num_sents <= self.nb_clusters:
                summary_list.append(" ".join(sentences_list))
                print("continue----")
                continue

            X = self.construct_sentence_graph(sentences_list)
            cluster_dict = self.cluster_graph(X, sentences_list)
            summary = self.compress_cluster(cluster_dict)
            summary_list.append(summary)
        return summary_list










	