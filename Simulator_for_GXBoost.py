# -*- coding: utf-8 -*-
from Simulator_fun_2207 import *
from re_A1_class import Store, Platform_pool
import numpy as np
import simpy
from A1_BasicFunc import  OrdergeneratorByCSV
import datetime


current_time = datetime.datetime.now()
save_id = str(current_time.day) + '_' + str(current_time.hour) + '_' + str(current_time.minute) + '_' + str(current_time.second)
env = simpy.Environment()
Platform_dict = Platform_pool()
Store_dict = {}
Orders = {}
test1 = []
store_dir = '송파구store_Coor.txt'
customer_dir = '송파구house_Coor.txt'
heuristic_theta = 10
heuristic_r1 = 10
ellipse_w = 10
speed = 3
Saved_data = []
DummyB2 = []
DummyB3 = []
#1 주문 생성
orders, stores, customers = OrderGen(store_dir, customer_dir, store_size = 100, customer_size = 1000, order_size = 1000, coor_random = True)

for data in stores:
    #['name', 'start_loc_x', 'start_loc_y', 'order_ready_time', 'capacity', 'slack']
    name = data[0]
    loc = [data[2], data[3]]
    order_ready_time = 3
    capacity = 3
    slack = 3
    store = Store(env, Platform_dict, name, loc=loc, order_ready_time=order_ready_time, capacity=capacity, print_para=False, slack = slack)
    Store_dict[name] = store

env.process(OrdergeneratorByCSV(env, test1, Orders, Store_dict, Platform_dict, custom_data=orders))

#2번들을 탐색하는 과정
env.process(BundleProcess(env, Orders,Platform_dict, heuristic_theta, heuristic_r1,ellipse_w,1.6,bundle_size=[3], bundle_permutation_option = True, speed = speed, Data = Saved_data, DummyB2_data = DummyB2, DummyB3_data = DummyB3))

env.run(200)
print(len(Orders))
order_np = np.array(orders, dtype=np.float64)
np.save('./GXBoost/'+save_id+'saved_orders', order_np)
#Feature saved Part
label_datas = []
count = 0
label1_names = []
print(len(Saved_data))
#input('Saved_data')
for data in Saved_data:
    # ver1: [route, unsync_t[0], round(sum(ftds) / len(ftds), 2), unsync_t[1], order_names, round(route_time, 2),min(time_buffer), round(P2P_dist - route_time, 2)]
    print(data)
    tem = [count, len(data[4])]
    tem += data[4]
    tem += data[0]
    tem += [data[2]]
    tem += [data[5]]
    label_datas.append(tem)
    label1_names.append(data[4])
label_datas_np = np.array(label_datas)
np.save('./GXBoost3/'+save_id+'label_datas_np', label_datas_np)

"""
#DummyB2
Dummy_B2_datas = []
Dummy_B2_datas_names = []
count = 0
for data in DummyB2:
    if data not in Dummy_B2_datas_names:
        tem = [count, 2]
        tem += data
        Dummy_B2_datas.append(tem)
        Dummy_B2_datas_names.append(data)
    #print('data',data)
    #print('더미3',Dummy_B2_datas_names)
Dummy_B2_datas_np = np.array(Dummy_B2_datas, dtype=int)
np.save('./GXBoost/'+save_id+'Dummy_B2_datas', Dummy_B2_datas)
"""
#DummyB3
Dummy_B3_datas = []
Dummy_B3_datas_names = []
count = 0
for data in DummyB3:
    tem = [count, 3]
    tem += data
    Dummy_B3_datas.append(tem)
    Dummy_B3_datas_names.append(data)
Dummy_B3_datas_np = np.array(Dummy_B3_datas, dtype=int)
np.save('./GXBoost3/'+save_id+'Dummy_B3_datas', Dummy_B3_datas)

print('입력1',len(label1_names))
label1_data = BundleFeaturesCalculator(order_np, label1_names, label = 1)
print('입력1_중복제거',len(label1_data))
#input('확인1')
#print('입력2',len(Dummy_B2_datas_names), Dummy_B2_datas_names[:5])
#label0_data = BundleFeaturesCalculator(order_np, Dummy_B2_datas_names, label = 0)
#input('확인1')
print('입력2',len(Dummy_B3_datas_names), Dummy_B3_datas_names[:5])
label0_data = BundleFeaturesCalculator(order_np, Dummy_B3_datas_names, label = 0)
print('입력2_중복제거',len(label0_data))
#input('확인2')
raw_data = label1_data + label0_data
raw_data_np = np.array(raw_data, dtype=np.float64)
np.save('./GXBoost3/'+save_id+'raw_data_np', raw_data_np)

#결과 확인
res= np.load('./GXBoost3/'+save_id+'raw_data_np.npy')
print('저장 결과',np.shape(res))
