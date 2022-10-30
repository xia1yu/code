#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
os.chdir('./01_叶片结冰预测/train/15')  #运行前更改目录为train/15这个文件夹
import pandas as pd
import numpy as np


# 转化normal_data & fail_data时间列为datetime
def to_datetime(obj_pd):
    Ser1 = obj_pd.iloc[:, 0]
    Ser2 = obj_pd.iloc[:, 1]
    for i in range(len(Ser1)):
        Ser1[i] = pd.to_datetime(Ser1[i])
        Ser2[i] = pd.to_datetime(Ser2[i])
    obj_pd.iloc[:, 0] = Ser1
    obj_pd.iloc[:, 1] = Ser2
    return obj_pd


# print 数据信息
def data_judge(labels, total):
    sum_inv = 0
    for i in range(len(labels)):
        if (labels[i] == -1):
            sum_inv = sum_inv + 1
    print("sum of invalid data : %d , %.2f %%" % (sum_inv, sum_inv / total * 100))

    sum_nor = 0
    for i in range(len(labels)):
        if (labels[i] == 0):
            sum_nor = sum_nor + 1
    print("sum of normal data : %d , %.2f %% " % (sum_nor, sum_nor / total * 100))

    sum_fail = 0
    for i in range(len(labels)):
        if (labels[i] == 1):
            sum_fail = sum_fail + 1
    print("sum of failure data : %d , %.2f %% " % (sum_fail, sum_fail / total * 100))


if __name__ =='__main__':
    # Step1 读取数据
    data = pd.read_csv("15_data.csv")
    total = len(data)
    print("sum of data:%d" % total)
    des = data.describe()
    fail_data = pd.read_csv("15_failureInfo.csv")
    normal_data = pd.read_csv("15_normalInfo.csv")

    # label = 1: 故障时间区域
    # label = 0: 正常时间区域
    # label = -1:无效数据

    # 转化data时间列为datetime
    times = []
    for i in range(len(data)):
        dt = pd.to_datetime(data.ix[i][0])
        times.append(dt)
        if (i % 10000 == 0):
            print("complete %d / %d" % (i, len(data)))
    times = pd.Series(times)
    data.time = times

    normal_data = to_datetime(normal_data)
    fail_data = to_datetime(fail_data)

    # 根据datetime创建labels列表
    labels = []
    for i in range(len(times)):
        if (i % 10000 == 0):
            print("complete %d / %d" % (i, len(times)))
        flag = 0
        for j in range(len(normal_data)):
            if ((times[i] >= normal_data.startTime[j]) and (times[i] <= normal_data.endTime[j])):
                labels.append(0)
                flag = 1
                break
        for j in range(len(fail_data)):
            if (flag == 1):
                break
            elif ((times[i] >= fail_data.startTime[j]) and (times[i] <= fail_data.endTime[j])):
                labels.append(1)
                flag = 1
                break
        if (flag == 1):
            continue
        labels.append(-1)
    print("complete all")

    data_judge(labels, total)

    # 删除无效数据
    y = labels
    indexes = []
    for i in range(len(y)):
        if (y[i] == -1):
            indexes.append(i)
    data = data.drop(indexes)
    data = data.drop('time', axis=1)
    for i in range(len(y) - 1, -1, -1):
        if (y[i] == -1):
            y.pop(i)

    #拼接数据和标签
    y = np.array(y).reshape(len(y), 1)
    data = np.concatenate((data, y), axis=1)

    #不平衡数据处理
    # nor_data = np.zeros((1,28))
    # fai_data = np.zeros((1, 28))
    # for i in range(y.shape[0]):
    #     if (data[i,-1] == 0):
    #         nor_data = np.concatenate((nor_data, data[i,:].reshape(1, 28)), axis=0)
    #     if (data[i,-1] == 1):
    #         fai_data = np.concatenate((fai_data, data[i,:].reshape(1, 28)), axis=0)
    #
    #     fai_data = fai_data[1:,:]
    # nor_data = nor_data[1:fai_data.shape[0]+1,:]

    # data = np.concatenate((fai_data, nor_data), axis=0)

    data = pd.DataFrame(data)
    writer = pd.ExcelWriter('data1.xlsx')
    data.to_excel(writer)
    writer.save()



