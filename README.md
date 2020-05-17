# SummPip

**This code is for Sigir 2020 paper Unsupervised Multi-Document Summarizationwith Sentence Graph Compression**

**Python version**: this node requires is in Python3.6

## Dataset

[source data](https://drive.google.com/file/d/1_iDBecWsEkzuEou5-xi0z2ek3oJJ8CPB/view?usp=sharing) which has minimal text pre-processing

[target data](https://drive.google.com/file/d/1T9uE2sF3bN3a1T2KLp7mR4xK9MqqpkH1/view?usp=sharing) (for evaluation)

## Test SummPip

**Step1**: place downloaded dataset in the folder `dataset/multi_news/`.

**Step2**: download the pre-trained [word2vec model](https://drive.google.com/file/d/1DVaktsGKbH8oPy28rrHuVgL_QVDsbfSA/view?usp=sharing) and place it in the folder `word_vec/multi_news`. 

- If you want to run SummPip on your own dataset, you need to pre-train a W2V model yourself first with [gensim](https://radimrehurek.com/gensim/index.html).

**Step3**: Unsupervised Extractive Summarisation

> python run_main.py

- You may want to change `-nb_clusters` and `-nb_words` to control the length of the output summary when applying SummPip on your own dataset. 


