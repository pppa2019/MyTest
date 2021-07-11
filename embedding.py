'''
@Author: xiaoyao jiang
@Date: 2020-04-08 17:22:54
LastEditTime: 2021-01-03 21:39:08
LastEditors: Peixin Lin
@Description: train embedding & tfidf & autoencoder
FilePath: /JD_NLP1-text_classfication/embedding.py
'''
import pandas as pd
import numpy as np
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import jieba
from gensim.models import LdaMulticore
from features import label2idx
import gensim
import config


class SingletonMetaclass(type):
    '''
    @description: singleton
    '''
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,
                                    self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        '''
        @description: This is embedding class. Maybe call so many times. we need use singleton model.
        In this class, we can use tfidf, word2vec, fasttext, autoencoder word embedding
        @param {type} None
        @return: None
        '''

        #######################################################################
        #          TODO:  读取停用词 #
        #######################################################################
        # 停止词
        #
        self.stopword = []
        with open('./data/stopwords.txt', 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            self.stopword.extend(lines)
        print(self.stopword)



    def load_data(self, path):
        '''
        @description:Load all data, then do word segment
        @param {type} None
        @return:None
        '''
        data = pd.read_csv(path, sep='\t', encoding='utf-8')
        data = data.fillna("")
        #######################################################################
        #          TODO:  去除停用词 #
        #######################################################################
        # 对data['text']中的词进行分割，并去除停用词 参考格式： data['text'] = data["text"].apply(lambda x: " ".join(x))
        #
        data['text'] = data['text'].apply(lambda x: " ".join([w for w in x.split() if w not in self.stopword and w!='']))
        self.labelToIndex = label2idx(data)
        data['label'] = data['label'].map(self.labelToIndex)
        data['label'] = data.apply(lambda row: float(row['label']), axis=1)
        data = data[['text', 'label']]
#         self.train, _, _ = np.split(data[['text', 'label']].sample(frac=1), [int(data.shape[0] * 0.7), int(data.shape[0] * 0.9)])
        self.train = data['text'].tolist()


    def trainer(self):
        '''
        @description: Train tfidf,  word2vec, fasttext and autoencoder
        @param {type} None
        @return: None
        '''
        #######################################################################
        #          TODO:  模型训练 #
        #######################################################################
        #count_vect 对 tfidfVectorizer 初始化
        #
        count_vect = TfidfVectorizer(stop_words=self.stopword, max_df=0.4, min_df=0.001,ngram_range=(1,2))
        print(self.train[:5])
        self.tfidf = count_vect.fit(self.train)



        #######################################################################
        #          TODO:  模型训练 #
        #######################################################################
        #对 w2v 初始化 并建立词表，训练
        #
        #
        #
        self.train = [sample.split() for sample in self.train]
        self.w2v = models.Word2Vec(min_count=2, size=300, window=5, negative=10, max_vocab_size=20000)
        self.w2v.build_vocab(self.train)
        self.w2v.train(self.train, total_examples=self.w2v.corpus_count, epochs=10, report_delay=1)

        self.id2word = gensim.corpora.Dictionary(self.train)
        corpus = [self.id2word.doc2bow(text) for text in self.train]

        #######################################################################
        #          TODO:  模型训练 #
        #######################################################################
        # 建立LDA模型
        #
        self.LDAmodel = models.ldamodel.LdaModel(corpus=corpus, id2word=self.id2word, num_topics=11)


    def saver(self):
        '''
        @description: save all model
        @param {type} None
        @return: None
        '''
        joblib.dump(self.tfidf, './model/tfidf')

        self.w2v.wv.save_word2vec_format('./model/w2v.bin',
                                         binary=False)

        self.LDAmodel.save('./model/lda')

    def load(self):
        '''
        @description: Load all embedding model
        @param {type} None
        @return: None
        '''
        self.tfidf = joblib.load('./model/tfidf')
        self.w2v = models.KeyedVectors.load_word2vec_format('./model/w2v.bin', binary=False)
        self.lda = models.ldamodel.LdaModel.load('./model/lda')


if __name__ == "__main__":
    em = Embedding()
    em.load_data(config.train_data_file)
    em.trainer()
    em.saver()
