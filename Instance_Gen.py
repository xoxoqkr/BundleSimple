# -*- coding: utf-8 -*-
#<11/22 version>
import copy

import matplotlib.pyplot as plt
import csv
import time

import numpy as np
import simpy
import random
from re_A1_class import scenario,Platform_pool
from A1_BasicFunc import ResultSave, GenerateStoreByCSV, RiderGeneratorByCSV, OrdergeneratorByCSV, distance, counter, check_list, t_counter, GenerateStoreByCSVStressTest, OrdergeneratorByCSVForStressTest, RiderGenerator, counter2
from A2_Func import ResultPrint
from re_platform import Platform_process5,Rider_Bundle_plt
from datetime import datetime
#import onnxmltools
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost  # noqa
#import onnxmltools.convert.common.data_types
import onnxruntime as rt
#from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
import platform
print(platform.architecture())
#input('버전 확인')





"""
#global variable
global instance_type
global ellipse_w
global heuristic_theta
global heuristic_r1
global heuristic_type
global rider_num
global mix_ratios
global scenario_indexs
global exp_range
global unit_fee
global fee_type
global service_time_diff
"""


instance_type = 'Instance_random' #'Instance_random' 'Instance_cluster'
ellipse_w=10
heuristic_theta=10
heuristic_r1=10
heuristic_type = 'enumerate'#'XGBoost'#'enumerate'
rider_num= 175 #8
mix_ratios=None
exp_range = [0,1,2,3,4,5,6,7,8,9]
unit_fee = 110
fee_type = 'linear'
service_time_diff = True
thres_label = 25


setting = 'stresstest'
stress_lamda = 40 # 분당 주문 발생 수 # (2400/60)/5 #기준은 한 구에 분당 3750/60 #원래 40
stress_rider_num = 320  #기준은 한 구에 400명
# Parameter define
interval = 5
run_time = 30 #120
cool_time = 30  # run_time - cool_time 시점까지만 고객 생성
uncertainty_para = True  # 음식 주문 불확실성 고려
rider_exp_error = 1.5  # 라이더가 가지는 불확실성
platform_exp_error = 1.2  # 플랫폼이 가지는 불확실성
cook_time_type = 'uncertainty'
cooking_time = [7, 1]  # [평균, 분산]
thres_p = 0
save_as_file = False
save_budnle_as_file = False
rider_working_time = 120
# env = simpy.Environment()
store_num = 20
rider_gen_interval = 2  # 라이더 생성 간격.
rider_speed = 3 #10
rider_capacity = 3
start_ite = 0
ITE_NUM = 1
option_para = True  # True : 가게와 고객을 따로 -> 시간 단축 가능 :: False : 가게와 고객을 같이 -> 시간 증가
customer_max_range = 50
store_max_range = 30
divide_option = True  # True : 구성된 번들에 속한 고객들을 다시 개별 고객으로 나눔. False: 번들로 구성된 고객들은 번들로만 구성
p2_set = True
rider_p2 = 2 #1.5
platform_p2 = 2 # rider_p2*0.8  #1.3 p2_set이 False인 경우에는 p2만큼의 시간이 p2로 고정됨. #p2_set이 True인 경우에는 p2*dis(가게,고객)/speed 만큼이 p2시간으로 설정됨.
customer_p2 = 1 #2
obj_types = ['simple_max_s'] #['simple_max_s', 'max_s+probability', 'simple_over_lt','over_lt+probability'] #todo : 0317_수정본. min_pr을 무의미한 제약식으로 설정
# order_p2 = [[1.5,2,3],[0.3,0.3,0.4]] #음식 별로 민감도가 차이남.
wait_para = False  # True: 음식조리로 인한 대기시간 발생 #False : 음식 대기로 인한 대기시간 발생X
scenarios = []
run_para = True  # True : 시뮬레이션 작동 #False 데이터 저장용
r2_onx = 'pipeline_xgboost2_r_ver11'# 'pipeline_xgboost2_r_2_ver7'
r3_onx = 'pipeline_xgboost3_r_ver11'#'pipeline_xgboost3_r_3_ver6'
c2_onx = 'pipeline_xgboost2_c_ver9'
c3_onx = 'pipeline_xgboost3_c_ver10'
f = open("결과저장0706.txt", 'a')
f.write('결과저장 시작' + '\n')
f.close()



order_select_type = 'simple' #oracle ; simple

sc_index = 0
for i in [True,False]:
    for j in [True,False]:
        sc = scenario('{}:P:{}/R:{}'.format(str(sc_index), i, j))
        sc.platform_recommend = i
        sc.rider_bundle_construct = j
        scenarios.append(sc)
        sc.obj_type = 'None'
        sc_index += 1
print('시나리오 확인1')
for sc1 in scenarios:
    print(sc1.platform_recommend, sc1.rider_bundle_construct,sc1.obj_type)

for j in [True,False]:
    for k in obj_types: #[obj_types[1], obj_types[3]]:
        sc = scenario('{}:P:{}/R:{}'.format(str(sc_index), i, j))
        sc.platform_recommend = True
        sc.obj_type = k
        sc.rider_bundle_construct = j
        scenarios.append(sc)
        sc_index += 1

print('시나리오 확인2')
for sc1 in scenarios:
    print(sc1.platform_recommend, sc1.rider_bundle_construct,sc1.obj_type)


print_fig = False
bundle_print_fig = False
rider_select_print_fig = False

#scenarios = scenarios[2:4]
#scenarios = [copy.deepcopy(scenarios[8]), copy.deepcopy(scenarios[8]),copy.deepcopy(scenarios[8]),scenarios[2],scenarios[3]]
#scenarios[0].search_type = 'enumerate'
#scenarios[2].search_type = 'ellipse'
#scenarios = [scenarios[2],scenarios[3],copy.deepcopy(scenarios[4]), copy.deepcopy(scenarios[8]),copy.deepcopy(scenarios[4]), copy.deepcopy(scenarios[8]),copy.deepcopy(scenarios[4]), copy.deepcopy(scenarios[8])]
#scenarios[0].search_type = 'enumerate'
#scenarios[4].search_type = 'enumerate'
#scenarios[5].search_type = 'enumerate'
#scenarios[6].search_type = 'ellipse'
#scenarios[7].search_type = 'ellipse'
#scenarios = [scenarios[2],copy.deepcopy(scenarios[8]),scenarios[3]]
#scenarios = [copy.deepcopy(scenarios[8])]
#scenarios[0].search_type = heuristic_type
scenarios = [scenarios[2],scenarios[2]]
scenarios[0].search_type = heuristic_type
if mix_ratios != None:
    for ratio in mix_ratios:
        test_sc = copy.deepcopy(scenarios[1])
        test_sc.mix_ratio = copy.deepcopy(ratio)
        scenarios.append(test_sc)
"""
scenarios = [scenarios[1]]*4

for count in range(len(scenarios)):
    scenarios[count].obj_type = obj_types[count]
    print(obj_types[count], scenarios[count].obj_type)
"""
print('시나리오 확인3')
print(heuristic_type)
for sc3 in scenarios:
    #sc3.search_type = heuristic_type
    sc3.platform_recommend = False
    sc3.rider_bundle_construct = False
    print(sc3.platform_recommend, sc3.rider_bundle_construct,sc3.obj_type, sc3.search_type)
scenarios = scenarios[:1]
scenarios[0].obj_type = 'simple_max_s'
print(scenarios, scenarios[0].obj_type)

input('시나리오 확인')

#exp_range = [0,2,3,4]*10 #인스턴스 1에러가 있음.
#global exp_range #인스턴스 1에러가 있음.
#instance_type = 'Instance_cluster' #'Instance_cluster' / 'Instance_random'
#input('instance_type {} '.format(instance_type))
#search_type = 'heuristic'
#input('확인 {}'.format(len(scenarios)))
if heuristic_type == 'XGBoost':
    see_dir = 'C:/Users/xoxoq/OneDrive/Ipython/handson-gb-main/handson-gb-main/Chapter05/'
    if instance_type == 'Instance_random':
        sees2 = rt.InferenceSession(see_dir + r2_onx + '.onnx')  # "pipeline_xgboost2_r_2_ver3.onnx"
        sess3 = rt.InferenceSession(see_dir + r3_onx + '.onnx')  #pipeline_xgboost2_r_3
        #sess3 = None
    else:
        sees2 = rt.InferenceSession(see_dir + c2_onx + '.onnx')  # "pipeline_xgboost2_c_2_ver1.onnx"
        sess3 = rt.InferenceSession(see_dir + c3_onx + '.onnx') #"pipeline_xgboost2_c_3_ver1.onnx"
        #sess3 = None
    #pred_onx = sess.run(None, {"input": X_test1[:5].astype(numpy.float32)}) #Input must be a list of dictionaries or a single numpy array for input 'input'.
    #print("predict", pred_onx[0])
    #print("predict_proba", pred_onx[1][:1])
    XGBmodel2 = sees2
    XGBmodel3 = sess3
else:
    XGBmodel2 = None
    XGBmodel3 = None


rv_count = 0
for ite in exp_range:#range(0, 1):
    rv_count += 1
    # instance generate
    lamda_list = []
    for rider_name in range(100):
        lamda_list.append(random.randint(4,7))
    lead_time_stroage = []
    foodlead_time_stroage = []
    foodlead_time_ratio_stroage =[]
    labels = []
    num_bundles = []
    for sc in scenarios:
        ##count 확인
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
        ##counter 정의
        t_counter.t1 = 0
        t_counter.t2 = 0
        t_counter.t3 = 0
        t_counter.t4 = 0
        counter2.num1 = []
        counter2.num2 = []
        counter2.num3 = []
        counter2.num4 = []
        counter2.num5 = []
        bundle_infos = {'size': [],'length':[],'od':[]}
        #start_time_sec = time.time()
        start_time_sec = datetime.now()
        try:
            labels.append('{}{}{}'.format(str(sc.platform_recommend)[0],str(sc.rider_bundle_construct)[0],obj_types.index(sc.obj_type)))
        except:
            labels.append('{}{}N'.format(str(sc.platform_recommend)[0],str(sc.rider_bundle_construct)[0]))
        print('시나리오 정보 {} : {} : {} : {}'.format(sc.platform_recommend,sc.rider_bundle_construct,sc.scoring_type,sc.bundle_selection_type))
        sc.store_dir = instance_type + '/Instancestore_infos'+str(ite) #Instance_random_store/Instancestore_infos
        sc.customer_dir = instance_type + '/Instancecustomer_infos'+str(ite) #Instance_random_store/Instancecustomer_infos
        sc.rider_dir = 'Instance_random/Instancerider_infos0' #+str(ite) #Instance_random_store/Instancerider_infos
        Rider_dict = {}
        Orders = {}
        Platform2 = Platform_pool()
        Store_dict = {}
        # run
        env = simpy.Environment()
        if setting == 'stresstest':
            GenerateStoreByCSVStressTest(env, 200, Platform2, Store_dict, store_type=instance_type, ITE = ite)
            env.process(OrdergeneratorByCSVForStressTest(env, Orders, Store_dict, stress_lamda, platform=Platform2, p2_ratio=customer_p2, rider_speed=rider_speed,
                                             unit_fee=unit_fee, fee_type=fee_type, ITE = ite, instance_type= instance_type, simpy_para= True))
        env.run(200)
        f3 = open("고객_coord_정보" + str(ite) + '_' + instance_type + ".txt", 'a')
        for order_name in Orders:
            order = Orders[order_name]
            con = '{};{};{};{};{};{};{};\n'.format(order.name, order.location[0], order.location[1], order.store,
                                                   order.store_loc[0], order.store_loc[1], order.time_info[0])
            f3.write(con)
        f3.write('Exp End' + '\n')
        f3.close()

