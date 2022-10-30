#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

# Step1 读取数据
data = pd.read_excel("./data.xlsx")
data = np.array(data)
data_x = data[1:50001,0:-1]
y = data[1:50001,-1:]


# 在选择的数据中，选择70%作为训练集，30%作为测试集
X_train, X_test, y_train, y_test = train_test_split(data_x, y, test_size=0.3, random_state=1036, shuffle = True)

# 归一化
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)




# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues):
    """
    Normalization can be applied by setting `normalize = True`.
    """
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                horizontalalignment="center",
                color="white" if cm[i,j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("matrix.jpg")


def create_model(train_x):
    model = Sequential()
    #输入数据的shape为(n_samples, timestamps, features)
    # model.add(LSTM(units=50,input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(LSTM(units=60,return_sequences=True,input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(LSTM(units=40))
    model.add(Dense(units=1040))
    model.add(Dense(units=520))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))#选用线性激活函数
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr = 0.0001),metrics=['acc'])#损失函数为平均均方误差，优化器为Adam，学习率为0.01
    return model

# %% 训练率调整函数
def scheduler_cosine_decay(epoch,learning_rate=0.0001, decay_steps=100):
    if epoch==0:
      global global_step
      global new_lr
      global sess
      global_step = tf.Variable(tf.constant(0))
      new_lr = tf.train.polynomial_decay(learning_rate,global_step, decay_steps, cycle=True, end_learning_rate=0.00001)
      sess = tf.Session()
    lr = sess.run(new_lr,feed_dict={global_step: epoch})
    return lr

# %% Lstm训练
def Lstm_train(train_x,train_y):
    x_train_lstm = np.array(train_x)[:,:,np.newaxis]
    # x_train_lstm = np.array(trian_x).reshape(365,3,49)
    y_train = np.array(train_y)
    model = create_model(x_train_lstm)
    reduce_lr = LearningRateScheduler(scheduler_cosine_decay)
    history = model.fit(x_train_lstm, y_train, epochs=50, batch_size=100,callbacks = [reduce_lr])
    return model, history


def lossfig(history):
    epochs=range(len(history.history['acc']))
    plt.figure(2)
    plt.plot(epochs,history.history['acc'],'b',label='acc')
    plt.plot(epochs,history.history['loss'],'r',label='loss')
    plt.title('Training loss and accuracy')
    plt.legend()
    plt.savefig('loss_acc.jpg')


#训练LSTM网络
start_time = time.time()
lstm_model,lstm_history = Lstm_train(X_train, y_train)
X_test1 = np.expand_dims(X_test, axis=2)
endingtime = time.time()
print("-----lstm的训练时间为：{}".format(endingtime-start_time))
lstm_predict = lstm_model.predict(X_test1)
lossfig(lstm_history)


# 使用固定参数的随机森林分类器
# from sklearn.ensemble import RandomForestClassifier
# start_time = time.time()
# rf_clf = RandomForestClassifier(max_depth=146,n_estimators=100, max_leaf_nodes=2500, oob_score=True, random_state=30, n_jobs=-1)
# rf_clf.fit(X_train, y_train)
# endingtime = time.time()
# print("-----rf的训练时间为：{}".format(endingtime-start_time))
# rf_predict = rf_clf.predict(X_test)
# print(rf_clf.oob_score_)

#训练支持向量机
# from sklearn import svm
# start_time = time.time()
# # svm_model = svm.svc(kernel='linear', c=1, gamma=1)
# svm_model = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=1, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
# svm_model.fit(X_train, y_train)
# endingtime = time.time()
# print("-----svm的训练时间为：{}".format(endingtime-start_time))
# svm_predict= svm_model.predict(X_test)

cm = confusion_matrix(y_test, np.round(lstm_predict))
# cm = confusion_matrix(y_test, svm_predict)
# cm = confusion_matrix(y_test, rf_predict)
cm_plot_labels = ['normal', 'failure']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# 评价
#precision & recall & f1-score
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test, y_pred=np.round(lstm_predict)))
# print(classification_report(y_true=y_test, y_pred=svm_predict))
# print(classification_report(y_true=y_test, y_pred=rf_predict))
