# Transformer-Encoder-GRU (T-E-GRU) for Chinese Sentiment Analysis on Chinese Comment Text
## Environments
- Python 3.6.0
- Pytorch 1.11.0
 ## Prerequisites
 The code is built with following libraries:
- tensorflow == 2.3.0
- pandas == 1.2.1
- numpy == 1.19.4
- jieba == 0.42.1
- torch == 1.7.1
- gensim == 3.8.3
- torchtext ==0.12.0
- nltk==3.7

**Pretrain word embedding model**

> - Corpus: Zhihu_QA 知乎问答.
> - Context Features: Word + Ngram, 300Dim .
> - https://github.com/Embedding/Chinese-Word-Vectors.
> - Shen Li, Zhe Zhao, Renfen Hu, Wensi Li, Tao Liu, Xiaoyong Du, Analogical Reasoning on Chinese Morphological and Semantic Relations, ACL 2018.

---

> - https://nlp.stanford.edu/projects/glove/
> - Pennington J, Socher R, Manning C D. Glove: Global vectors for word representation[C]//Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014: 1532-1543.

```shell script
├─models                // models
│   └─embeddings        // pretrain word embedding model
│       └─sgns.zhihu.bigram.bz2
│       └─sgns.zhihu.bigram
|		└─glove-wiki-gigaword-300.gz
```

## Dataset

```shell script
├─data                // Dataset
│   └─IMDB          // IMDB
│   |   └─train.csv
│   │   └─val.csv
│   │   └─test.csv
│   └─yelp_review          // yelp_review
│   |   └─train.csv
│   │   └─val.csv
│   │   └─test.csv
│   └─douban          // dmsc_v2
│   |   └─train.csv
│   │   └─val.csv
│   │   └─test.csv
│   └─ dianping       // yf_dianping
│   │    └─train.csv
│   │    └─val.csv
│   │    └─test.csv
│   └─shopping        // yf_amazon
│        └─train.csv
│        └─val.csv
│        └─test.csv
```
- dmsc_v2:
- https://www.kaggle.com/utmhikari/doubanmovieshortcomments
- yf_dianping , yf_amazon:
- http://yongfeng.me/dataset/
- IMDB
  - http://ai.stanford.edu/~amaas/data/sentiment/
  - https://pytorch.org/text/stable/datasets.html#imdb
- yelp_review
  - https://arxiv.org/abs/1509.01626
  - https://pytorch.org/text/stable/datasets.html#yelpreviewpolarity

Specifically, about how to convert the source data to the data required by the project,
 please refer to [dataSetShow.ipynb](notebooks/dataSetShow.ipynb), [dealwithData.ipynb](notebooks/dealwithData.ipynb), [EDA_IMDB.ipynb](notebooks/EDA_IMDB.ipynb), [EDA_YELP.ipynb](notebooks/EDA_YELP.ipynb)

# Train
```shell script
├─models
│  ├─embeddings
│  │  
│  │  Eng_T_E_GRU.py
│  │  T_E_GRU.py
```
***models/~.py***  are two version of T-E-GRU

1. *Eng_T_E_GRU.py* is designed for English
2. *T_E_GRU.py* is designed for Chinese
3. To be continue  (Other comparison algorithms）

To train these models, you need :
- Modify hyper-parameters or dataName;
```python
if __name__ == '__main__':
    # You can modify the value of each variable before the net is instantiated
```
- Just run following:
```shell script
python xxxx.py 
```
**Note**:
- If your GPU is available, it will run to accelerate training, otherwise it will only be CPU by default;
- During training, it will provide rough estimations of accuracies and loss;
- After the training, the training log and the final model will be generated in the ***log*** folder, for example
```shell script
├─log
│  ├─douban
│  │  ├─T_E_GRU 
│  │  │  │  100.pkl
│  │  │  │  92.pkl
│  │  │  │  94.pkl
│  │  │  │  96.pkl
│  │  │  │  98.pkl
│  │  │  │  测试结果.txt  # result on testSet
│  │  │  ├─Acc_test
│  │  │  ├─Acc_train
│  │  │  ├─Loss_test
│  │  │  └─Loss_train

# if you want to know more training details, you can use the following command:
tensorboard --logdir=log
```

## Notebook
- ***./notebook*** has some detail about data processing and model training (To be continue).
- If you want to run them, **Jupyter lab** is the best.;
