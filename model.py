'''
Author: xiaoyao jiang
LastEditors: Peixin Lin
Date: 2020-08-31 14:19:30
LastEditTime: 2021-01-03 21:36:09
FilePath: /JD_NLP1-text_classfication/model.py
Desciption:
'''
import json
import jieba
import joblib
import numpy as np
import lightgbm as lgb
import pandas as pd
import sklearn.metrics as metrics
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from TextCNN import textCNN, train, evaluate
import torch
from torchsummary import summary

from embedding import Embedding
from features import (get_basic_feature, get_embedding_feature,
                      get_lda_features, get_tfidf)


class Classifier:
    # initialize, load the train/validation/test set
    def __init__(self, train_mode=False) -> None:
        self.stopWords = [
            x.strip() for x in open('./data/stopwords.txt', encoding='utf-8').readlines()
        ]
        self.embedding = Embedding()
        self.embedding.load()
        self.labelToIndex = json.load(
            open('./data/label2id.json', encoding='utf-8'))
        self.ix2label = {v: k for k, v in self.labelToIndex.items()}
        if train_mode:
            self.train = pd.read_csv('./data/train.csv',
                                     sep='\t').dropna().reset_index(drop=True)
            self.dev = pd.read_csv('./data/eval.csv',
                                   sep='\t').dropna().reset_index(drop=True)
            self.test = pd.read_csv('./data/test.csv',
                                    sep='\t').dropna().reset_index(drop=True)
        self.exclusive_col = ['text', 'lda', 'bow', 'label']

    # transfer strings to word vectors using TF-idf, word2vec, lda
    def feature_engineer(self, data):
        print(data.values.shape)
        data = get_tfidf(self.embedding.tfidf, data)
        print(data.values.shape)
        #data = get_embedding_feature(data, self.embedding.w2v)
        #print(data.values.shape)
        data = get_lda_features(data, self.embedding.lda)
        print(data.values.shape)
        data = get_basic_feature(data)
        print(data.values.shape)
        return data

    # preprocess the label and train model LGB
    def trainer(self, algo):
        #######################################################################
        #          TODO:  标签转换和数量统计 #
        #######################################################################
        # 初始化多标签训练
        dev = self.feature_engineer(self.dev)
        y_train = self.train['label'].values
        y_test = dev['label'].values
        y_train = np.array([self.labelToIndex[i] for i in y_train])
        y_test = np.array([self.labelToIndex[i] for i in y_test])
        counter = Counter(y_train)
        print('label distribution before SMOTE', counter)

        self.train = self.feature_engineer(self.train)
        cols = [x for x in self.train.columns if x not in self.exclusive_col]

        # self.train_select = self.train[cols].where(self.train[cols], 0)
        X_train = self.train[cols].values
        X_test = dev[cols].values
        print(X_train.shape, X_test.shape)
        test_df = self.train[cols]
        print(test_df.columns[test_df.isnull().any(axis=0)])
        # exit(0)
        '''
        # 用SMOTE来增强训练数据
        # TODO:写一个函数来确定sampling_strategy
        sample_num = 100
        oversample = SMOTE(sampling_strategy={0: sample_num, 1: sample_num, 2: sample_num,
                                              4: sample_num, 5: sample_num, 6: sample_num,
                                              7: sample_num, 8: sample_num, 9: sample_num, 10: sample_num})
        undersample = RandomUnderSampler(sampling_strategy={3: sample_num})
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        counter = Counter(y_train)
        print('label distribution after SMOTE', counter)
        X_train, y_train = undersample.fit_resample(X_train, y_train)
        counter = Counter(y_train)
        print('label distribution after random under sample', counter)
        '''
        if algo == 'XGBoost':
            #######################################################################
            #          TODO:  XGBoost模型最优参数GridSearchCV #
            #######################################################################
            # grid_param = [{'eta': [0.3, 0.1, 1], 'n_estimators': [70, 100, 120],'max_depth':[5, 7]}]
            # grid_model = GridSearchCV(xgb.XGBClassifier(), grid_param, cv=3, scoring='neg_log_loss', verbose=1)
            # grid_model.fit(X_train, y_train)
            # self.clf_BR = grid_model.best_estimator_
            # self.best_param = grid_model.best_params_
            # print(self.best_param)
            # f = open('best_param.txt', 'w')
            # f.write(str(self.best_param))
            # f.close()

            #######################################################################
            #          TODO:  XGBoost模型训练 #
            #######################################################################
            # 初始化训练参数，并进行fit
            param = {'eta': 0.1, 'max_depth': 7, 'n_estimators': 70}

            self.clf_XGBoost = xgb.XGBClassifier(param)
            print(X_train.shape, y_train.shape)
            self.clf_XGBoost.fit(X_train, y_train)
            prediction = self.clf_XGBoost.predict(X_test)
            print(prediction)
            print('Kappa score', metrics.cohen_kappa_score(y_test, [i.item() for i in prediction]))
            

        if algo == 'TextCNN':
            args = {'embedding_num': 6000, 'embed_dim': X_train.shape[1], 'class_num': 11, 'kernel_num': 32, 'kernel_size_list':(3,3,3), 'dropout':0.1}
            model = textCNN(args)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
            loss_function = torch.nn.CrossEntropyLoss()
            train(model=model, opt=optimizer, loss_function=loss_function, X_train=X_train, y_train=y_train)
            acc, _ = evaluate(model=model, X_train=X_test, y_train=y_test)
            print('result', acc)
            self.textCNN = model
        

    def save(self, algo):
        if algo == 'XGBoost':
            joblib.dump(self.clf_XGBoost, './model/clf_XGBoost')
        if algo == 'TextCNN':
            joblib.dump(self.textCNN, './model/clf_textCNN')

    def load(self, algo):
        if algo == 'XGBoost':
            self.clf_XGBoost = joblib.load('./model/clf_XGBoost')
        if algo == 'TextCNN':
            self.textCNN = joblib.load('./model/clf_textCNN')


    def predict(self, text, algo='XGBoost'):
        df = pd.DataFrame([[text]], columns=['text'])
        df['text'] = df['text'].apply(lambda x: " ".join(
            [w for w in jieba.cut(x) if w not in self.stopWords and w != '']))
        df = get_tfidf(self.embedding.tfidf, df)
        #df = get_embedding_feature(df, self.embedding.w2v)
        df = get_lda_features(df, self.embedding.lda)
        df = get_basic_feature(df)
        cols = [x for x in df.columns if x not in self.exclusive_col]
        #######################################################################
        #          TODO:  XGBoost模型预测 #
        #######################################################################
        # 利用模型获得预测结果
        if algo == 'XGBoost':
            print(df[cols].values.shape)
            pred = self.clf_XGBoost.predict(df[cols].values)
            return [self.ix2label[i] for i in range(len(pred)) if pred[i] > 0]
        if algo == 'TextCNN':
            pred = self.textCNN(torch.Tensor(df[cols].values))
            pred_int = torch.max(pred, dim=1)[1]
            print(self.ix2label)
            print(self.labelToIndex)
            print(int(pred_int[0]))
            return self.ix2label[int(pred_int[0])]


if __name__ == "__main__":
    bc = Classifier(train_mode=True)
    bc.trainer(algo='XGBoost')
    bc.save(algo='XGBoost')
