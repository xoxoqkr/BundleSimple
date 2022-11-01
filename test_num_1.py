# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import random
import itertools

max_n = 1000
exp_val = (max_n*(max_n-1)*(max_n-2))/6
"""
start_time_sec = time.time()
try:
    test = []
    for i in range(max_n):
        if i % int(max_n/10) == 0:
            print('{}%진행; 현재 i ; {}'.format(i/max_n, i))
        for j in range(i+1,max_n):
            for k in range(j+1, max_n):
                test.append([i,j,k])
except:
    pass
print('예상 조합 수는?',len(test))
end_time_sec = time.time()
Appoxi_t = end_time_sec - start_time_sec
print('3 forall문 시간;{};'.format(Appoxi_t))
"""
start_time_sec = time.time()
possible_subset = []
try:
    customers = list(range(max_n))
    possible_subset = list(itertools.permutations(customers, 3))
except:
    pass
end_time_sec = time.time()
Appoxi_t = end_time_sec - start_time_sec
print('itertools.permutations ;{};길이;{};'.format(Appoxi_t, len(possible_subset)))

start_time_sec = time.time()
count = 0
try:
    subset = random.sample(possible_subset, 100)
    for i in subset:
        if i in possible_subset:
            count += 1
            pass
        else:
            pass
except:
    pass

end_time_sec = time.time()
Appoxi_t = end_time_sec - start_time_sec
print(count)
print('itertools.permutations에서 index 찾기 {};'.format(Appoxi_t))
"""
# -*- coding: utf-8 -*-
from A3_two_sided import XGBoost_Bundle_Construct, ConstructFeasibleBundle_TwoSided
import onnxruntime as rt
import time
import random
from re_A1_class import Platform_pool
from A1_BasicFunc import GenerateStoreByCSVStressTest, OrdergeneratorByCSVForStressTest, t_counter, counter2, counter, check_list
import simpy

print('시작')
test_type = 'XGBoost' # 'XGBoost' 'enumerate'

speed = 3


r2_onx = 'pipeline_xgboost2_r_ver11'
r3_onx = 'pipeline_xgboost3_r_ver11'
see_dir = 'C:/Users/xoxoq/OneDrive/Ipython/handson-gb-main/handson-gb-main/Chapter05/'
XGBmodel2 = rt.InferenceSession(see_dir + r2_onx + '.onnx')  # "pipeline_xgboost2_r_2_ver3.onnx"
XGBmodel3 = rt.InferenceSession(see_dir + r3_onx + '.onnx')  # pipeline_xgboost2_r_3

now_t = 0
bundle_permutation_option = True
thres = 100
p2 = 1
thres_label = 25

ite = 0
instance_type = 'Instance_random'
Rider_dict = {}
Orders = {}
Platform2 = Platform_pool()
Store_dict = {}

stress_lamda = 200
customer_p2 = 1
rider_speed = speed
unit_fee = 110
fee_type = 'linear'

rider_capacity = 3

customer_file = 'E:/학교업무 동기화용/py_charm/BundleSimple/' + instance_type + '/고객_coord_정보' + str(
    ite) + '_' + instance_type + '.txt'
store_file = 'E:/학교업무 동기화용/py_charm/BundleSimple/' + instance_type + '/가게_coord_정보' + str(
    ite) + '_' + instance_type + '.txt'

CustomerCoord = []
StoreCoord = []
f_c = open(customer_file, 'r')
lines = f_c.readlines()
for line in lines[:-1]:
    line1 = line.split(';')
    CustomerCoord.append(
        [int(line1[0]), float(line1[1]), float(line1[2]), int(line1[3]), float(line1[4]), float(line1[5]),
         float(line1[6])])
f_c.close()

f_s = open(store_file, 'r')
lines = f_s.readlines()
for line in lines[:-1]:
    line1 = line.split(';')
    StoreCoord.append([int(line1[0]), int(line1[1]), int(line1[2])])
f_s.close()


env = simpy.Environment()
GenerateStoreByCSVStressTest(env, 200, Platform2, Store_dict, store_type=instance_type, ITE=ite, output_data=StoreCoord)
env.process(OrdergeneratorByCSVForStressTest(env, Orders, Store_dict, stress_lamda, platform=Platform2, p2_ratio=customer_p2,rider_speed=rider_speed,unit_fee=unit_fee, fee_type=fee_type, output_data=CustomerCoord))
env.run(10)



print(len(Orders))


target_order = Orders[500]
considered_customers = {}

indexs = list(range(len(Orders)))
random.seed(1)
random.shuffle(indexs)

for i in indexs[:1000]:
    order = Orders[i]
    order.cancel = False
    considered_customers[order.name] = order
considered_customers[target_order.name] = Orders[target_order.name]
considered_customers[target_order.name].cancel = False
print(target_order, len(considered_customers))
#input('확인')
t_counter.t1 = 0
t_counter.t2 = 0
t_counter.t3 = 0
t_counter.t4 = 0
t_counter.t5 = 0
t_counter.t6 = 0
t_counter.t7 = 0
t_counter.t8 = 0
t_counter.t9 = 0
t_counter.t10 = 0
t_counter.t11 = 0
t_counter.t12 = 0
t_counter.t13 = 0
t_counter.t14 = 0
t_counter.t15 = 0
t_counter.t16 = 0
t_counter.t17 = 0
counter2.num1 = []
counter2.num2 = []
counter2.num3 = []
counter2.num4 = []
counter2.num5 = []
counter.dist1 = 0
counter.dist2 = 0
counter.dist3 = 0
counter.bundle_consist = 0
counter.bundle_consist2 = 0
check_list.b2 = []
check_list.b3 = []
check_list.b2_count = 0
check_list.b3_count = 0
check_list.suggested_bundle = []
print('XGStart')
start_time_sec = time.time()
cut_info3 = [7.5,15]
cut_info2 = [10,10]
size3bundle, label3data, test_b33 = XGBoost_Bundle_Construct(target_order, considered_customers, 3, p2, XGBmodel3,
                                                             now_t=now_t, speed=speed,
                                                             bundle_permutation_option=bundle_permutation_option,
                                                             thres=thres, thres_label=thres_label, feasible_return=False, cut_info= cut_info3)
size2bundle, label2data, test_b22 = XGBoost_Bundle_Construct(target_order, considered_customers, 2, p2, XGBmodel2,
                                                             now_t=now_t, speed=speed,
                                                             bundle_permutation_option=bundle_permutation_option,
                                                             thres=thres, thres_label=thres_label, feasible_return=False, cut_info= cut_info2)
end_time_sec = time.time()
XG_duration = end_time_sec - start_time_sec
infoXG = ';{};{}'.format(t_counter.t10,t_counter.t11)

t_counter.t1 = 0
t_counter.t2 = 0
t_counter.t3 = 0
t_counter.t4 = 0
t_counter.t5 = 0
t_counter.t6 = 0
t_counter.t7 = 0
t_counter.t8 = 0
t_counter.t9 = 0
t_counter.t10 = 0
t_counter.t11 = 0
t_counter.t12 = 0
t_counter.t13 = 0
t_counter.t14 = 0
t_counter.t15 = 0
t_counter.t16 = 0
t_counter.t17 = 0
counter2.num1 = []
counter2.num2 = []
counter2.num3 = []
counter2.num4 = []
counter2.num5 = []
counter.dist1 = 0
counter.dist2 = 0
counter.dist3 = 0
counter.bundle_consist = 0
counter.bundle_consist2 = 0
check_list.b2 = []
check_list.b3 = []
check_list.b2_count = 0
check_list.b3_count = 0
check_list.suggested_bundle = []
print('ENuStart')
start_time_sec = time.time()

size3bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 3, p2, speed=speed,
                                               bundle_permutation_option=bundle_permutation_option, thres=thres,
                                               now_t=now_t, print_option=False, feasible_return=False)
size2bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 2, p2, speed=speed,
                                               bundle_permutation_option=bundle_permutation_option, thres=thres,
                                               now_t=now_t, print_option=False, feasible_return=False)
end_time_sec = time.time()
Enu_duration = end_time_sec - start_time_sec
infoEnu = ';{};{}'.format(t_counter.t10,t_counter.t11)

print('XG',XG_duration, infoXG)
print('Enu',Enu_duration,infoEnu)
"""