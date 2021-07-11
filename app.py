'''
Author: xiaoyao jiang
LastEditors: Peixin Lin
Date: 2020-08-31 14:18:26
LastEditTime: 2021-01-03 21:41:09
FilePath: /JD_NLP1-text_classfication/app.py
Desciption: Application.
'''
from flask import Flask, request,render_template,url_for,redirect
import json
from model import Classifier


#######################################################################
#          TODO:  Initialize and load classifier model      #
#######################################################################
# 初始化模型， 避免在函数内部初始化，耗时过长
#
#
model = Classifier(train_mode=False)
model.load(algo='TextCNN')

#######################################################################
#          TODO:  Initialize flask     #
#######################################################################
# 初始化 flask
#
app = Flask(__name__)



#设定端口
@app.route('/', methods=["POST",'GET'])
def gen_ans():
    '''
    @description: 以RESTful的方式获取模型结果, 传入参数为title: 图书标题， desc: 图书描述
    @param {type}
    @return: json格式， 其中包含标签和对应概率
    '''
    if request.method=='GET':
        return render_template('index.html')


    #######################################################################
    #          TODO:  预测结果并返回 #
    #######################################################################
    #
    if request.method=="POST":
        text = request.form['text']
        label = model.predict(text, algo='TextCNN')
        result = {
            "label": label
        }

        return json.dumps(result, ensure_ascii=False)


# python3 -m flask run
if __name__ == '__main__':
    app.run(port=12000, debug=True)
