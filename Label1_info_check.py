# -*- coding: utf-8 -*-

import numpy as np
import os
import datetime
import pandas as pd
import copy

# raw_data에 order 좌표 붙이기 -> angle 계산
BundleSize = 3
thres_num = 1

# 데이터 읽기
if BundleSize == 2:
    dir1 = 'E:/학교업무 동기화용/py_charm/BundleSimple/GXBoost2'
    target_data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # B2
elif BundleSize == 3:
    dir1 = 'E:/학교업무 동기화용/py_charm/BundleSimple/GXBoost3'
    target_data = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # B3
else:
    pass
data_list = os.listdir(dir1)
counts = 10

def JoiningData(dir1, BundleSize, index_start1=11, index_end1= 12, start_name1 = 'r', att_num1 = 24, att_num2 = 24):
    data_list = os.listdir(dir1)
    target_data1 = []
    target_data2 = []
    for _ in range(att_num1):
        target_data1.append(1)
    target_data1= np.array([target_data1])
    for _ in range(att_num2):
        target_data2.append(1)
    target_data2= np.array([target_data2])
    for name in data_list:
        if str(name)[index_start1:index_end1] == start_name1:
            loaded1 = np.load(dir1 +'/'+ name)
            #target_data1 = np.concatenate((target_data1, loaded1), axis=0)
            #다른 주문 합치기
            new_data = []
            loaded2 = np.load(dir1 + '/'+name[:11]+'saved_orders_' +name[-7] + '_'+str(BundleSize) + '.npy')
            for b_info in loaded1:
                tem = copy.deepcopy(b_info)
                for index in range(BundleSize):
                    ct_name = b_info[index]
                    tem += loaded2[ct_name]
                new_data.append(tem)
            new_data = np.array(new_data, dtype=np.float64)
            revised_name = name[:11]+'saved_orders_' +name[-7] + '_'+str(BundleSize) + 'merged.npy'
            np.save(dir1 + '/' + revised_name, new_data)
            """
            df = pd.DataFrame(data = new_data)
            columns = []
            for i in range(att_num1):
                columns.append('f'+str(i+1))
            for i in range(BundleSize):
                for j in range(att_num2):
                    columns.append('c'+str(i+1)+'i' + str(j + 1))
            df.columns = columns
            """
            test = np.load(dir1 + '/' + revised_name)
            print(np.shape(test))
            input('check_Data')
