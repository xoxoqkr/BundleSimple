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
from A1_BasicFunc import ResultSave, RiderGeneratorByCSV, OrdergeneratorByCSV, distance, counter, check_list, t_counter, GenerateStoreByCSVStressTest, RiderGenerator, counter2, SaveScenario
from A2_Func import ResultPrint
from A3_two_sided import OrdergeneratorByCSVForStressTestDynamic
from re_platform import Platform_process5,Rider_Bundle_plt, DefreezeCustomers
from datetime import datetime
import os
os.environ["OMP_NUM_THREADS"] = '2'

from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost  # noqa
import onnxruntime as rt
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa
from Simulator_fun_2207 import BundleFeaturesCalculator2


global run_time
global customer_pend
global platform_recommend_input
global dynamic_env
global ct_gen_time
"""
run_time = 90
customer_pend = False
platform_recommend_input = True
dynamic_env = False
"""
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
heuristic_type = 'XGBoost'#'XGBoost'#'enumerate'
rider_num= 1 #8
mix_ratios=None
exp_range = [0] #list(range(10))#[0,1] [0,1,2,3,4,5,6,7,8,9]
unit_fee = 110
fee_type = 'linear'
service_time_diff = True
thres_label = 25
considered_customer_type = 'all' #'all' 'new'
search_range_index = 15
pr_para = False
save_data = False
bundle_start_fix = False
if save_data == True:
    save_root_dir = 'E:/GXBoost/old3/'
    save_id = 'xgb_1'
manual_cook_time = 7 #음식 조리 시간
cut_info3 = [12,24] # [12,24] [15,25] [7.5,10]#[7.5,10] #B3의 거리를 줄이는 함수
cut_info2 = [100,100]#[10,10]#[10,10]
stopping_index = 20 #40
clustering_para = True
revise_type_para = 'stopping' #'stopping' ; 'cut_info';'cut_info2';
cut_infoC = [100,100] #[8,16] #ConsideredCustomers 에서 잘리는 값 revise_type_para가 'cut_info';'cut_info2'; 경우에 작동
search_type2 = 'XGBoost' #'XGBoost'#'enumerate' -> 실제로 XGBoost냐 Enumerate냐를 결정


setting = 'stresstest'
stress_lamda = 40 # 분당 주문 발생 수 # (2400/60)/5 #기준은 한 구에 분당 3750/60 #원래 40
stress_rider_num = 320  #기준은 한 구에 400명
# Parameter define
interval = 5

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
rider_capacity = 1
start_ite = 0
ITE_NUM = 1
option_para = True  # True : 가게와 고객을 따로 -> 시간 단축 가능 :: False : 가게와 고객을 같이 -> 시간 증가
customer_max_range = 50
store_max_range = 30
divide_option = True  # True : 구성된 번들에 속한 고객들을 다시 개별 고객으로 나눔. False: 번들로 구성된 고객들은 번들로만 구성
p2_set = True
rider_p2 = 2 #1.5
platform_p2 = 2 # rider_p2*0.8  #1.3 p2_set이 False인 경우에는 p2만큼의 시간이 p2로 고정됨. #p2_set이 True인 경우에는 p2*dis(가게,고객)/speed 만큼이 p2시간으로 설정됨.
customer_p2 = 2 #2#todo 1118 : BundleConsist2 -> FLTCalculator -> 더 많은 수의 가능 번들을 만듬
obj_types = ['simple_max_s'] #['simple_max_s', 'max_s+probability', 'simple_over_lt','over_lt+probability'] #todo : 0317_수정본. min_pr을 무의미한 제약식으로 설정
# order_p2 = [[1.5,2,3],[0.3,0.3,0.4]] #음식 별로 민감도가 차이남.
wait_para = False  # True: 음식조리로 인한 대기시간 발생 #False : 음식 대기로 인한 대기시간 발생X
scenarios = []
run_para = True  # True : 시뮬레이션 작동 #False 데이터 저장용
r2_onx = 'pipeline_xgboost2_r_ver11'#'pipeline_xgboost2_r_ver13RandomTrue_0.9_25_5_5_0.2_dart_' # 'pipeline_xgboost2_r_ver11'# 'pipeline_xgboost2_r_2_ver7'
r3_onx = 'pipeline_xgboost3_r_ver11'#'pipeline_xgboost3_r_ver13RandomTrue_0.6_50_1_6_0.3_dart_'#'pipeline_xgboost3_r_ver11'#'pipeline_xgboost3_r_3_ver6'
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
        #global platform_recommend_input
        sc.platform_recommend = platform_recommend_input
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

print(platform_recommend_input)
#input('check')
for sc3 in scenarios:
    #sc3.search_type = heuristic_type
    sc3.platform_recommend = platform_recommend_input
    sc3.rider_bundle_construct = False
    print(sc3.platform_recommend, sc3.rider_bundle_construct,sc3.obj_type, sc3.search_type)
scenarios = scenarios[:1]
global obj_type
scenarios[0].obj_type = obj_type #'simple_max_s' #todo : simple_max_s Vs value+selective

print('시나리오 확인 Start')
print(scenarios)
#scenarios[1].platform_recommend = False
for sc4 in scenarios:
    print(sc4.platform_recommend, sc4.rider_bundle_construct, sc4.obj_type, sc4.search_type)

#dynamic 실험 관련 부분 #todo 1108 : 확인 필요
#run_time = 30
#dynamic_env = False
dynamic_infos = [0,0,0,0,0,0,0]
dynamic_infos[0] = platform_p2 #p2
dynamic_infos[1] = True #bundle_permutation_option
dynamic_infos[2] = True # feasible_return
dynamic_infos[3] = 10 #min_time_buffer
dynamic_infos[4] = 15 # max_dist
dynamic_infos[5] = 8 # sort_index
dynamic_infos[6] = False # fix_start
pr_off = True

#input('시나리오 확인')

#exp_range = [0,2,3,4]*10 #인스턴스 1에러가 있음.
#global exp_range #인스턴스 1에러가 있음.
#instance_type = 'Instance_cluster' #'Instance_cluster' / 'Instance_random'
#input('instance_type {} '.format(instance_type))
#search_type = 'heuristic'
#input('확인 {}'.format(len(scenarios)))
if heuristic_type == 'XGBoost':
    see_dir = 'C:/Ipython/handson-gb-main/handson-gb-main/Chapter05/'
    if instance_type == 'Instance_random':
        sees2 = rt.InferenceSession(see_dir + r2_onx + '.onnx',providers=['CPUExecutionProvider'])  # "pipeline_xgboost2_r_2_ver3.onnx"
        sess3 = rt.InferenceSession(see_dir + r3_onx + '.onnx',providers=['CPUExecutionProvider'])  #pipeline_xgboost2_r_3
        #sess3 = None
    else:
        sees2 = rt.InferenceSession(see_dir + c2_onx + '.onnx',providers=['CPUExecutionProvider'])  # "pipeline_xgboost2_c_2_ver1.onnx"
        sess3 = rt.InferenceSession(see_dir + c3_onx + '.onnx',providers=['CPUExecutionProvider']) #"pipeline_xgboost2_c_3_ver1.onnx"
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
    ML_Saved_Data_B2 = []
    ML_Saved_Data_B3 = []
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

    ###외부 데이터를 읽어 오는 과정

    customer_file = 'C:/Users/박태준/PycharmProjects/BundleSimple/' + instance_type + '/고객_coord_정보' + str(ite) + '_' + instance_type + '.txt'
    store_file = 'C:/Users/박태준/PycharmProjects/BundleSimple/' + instance_type + '/가게_coord_정보' + str(ite) + '_' + instance_type + '.txt'
    CustomerCoord = []
    StoreCoord = []
    f_c = open(customer_file, 'r')
    lines = f_c.readlines()
    for line in lines[:-1]:
        line1 = line.split(';')
        CustomerCoord.append([int(line1[0]),float(line1[1]),float(line1[2]),int(line1[3]),float(line1[4]),float(line1[5]),float(line1[6])])
    f_c.close()

    f_s = open(store_file, 'r')
    lines = f_s.readlines()
    for line in lines[:-1]:
        line1 = line.split(';')
        StoreCoord.append([int(line1[0]),int(line1[1]),int(line1[2])])
    f_s.close()


    for sc in scenarios:
        #if sc.rider_bundle_construct == False:
        #    rider_capacity = 3
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
            GenerateStoreByCSVStressTest(env, 200, Platform2, Store_dict, store_type=instance_type, ITE = ite, output_data= StoreCoord, customer_pend= customer_pend)
            env.process(
                OrdergeneratorByCSVForStressTestDynamic(env, Orders, Store_dict, stress_lamda, platform=Platform2,
                                                        customer_p2=customer_p2, platform_p2=platform_p2,
                                                        rider_speed=rider_speed,
                                                        unit_fee=unit_fee, fee_type=fee_type,
                                                        output_data=CustomerCoord, dynamic_infos=dynamic_infos,
                                                        riders=Rider_dict, pr_off=pr_off, end_t=ct_gen_time,
                                                        dynamic_para=dynamic_env, cooktime_detail=None,
                                                        customer_pend=customer_pend,
                                                        manual_cook_time=manual_cook_time,
                                                        search_range_index=stopping_index,
                                                        XGBmodel3 = XGBmodel3,
                                                        XGBmodel2 = XGBmodel2,
                                                        cut_info3 = cut_info3,
                                                        cut_info2 = cut_info2,
                                                        cal_type = 'XGBoost'))
            """
            if dynamic_env == True:
                env.process(OrdergeneratorByCSVForStressTestDynamic(env, Orders, Store_dict, stress_lamda, platform=Platform2,
                                                             customer_p2=customer_p2, platform_p2= platform_p2,rider_speed=rider_speed,
                                                             unit_fee=unit_fee, fee_type=fee_type,
                                                             output_data=CustomerCoord, dynamic_infos = dynamic_infos, riders = Rider_dict, pr_off= pr_off, end_t= run_time,
                                                                    dynamic_para = dynamic_env, cooktime_detail= None, customer_pend = customer_pend, 
                                                                    manual_cook_time = manual_cook_time, search_range_index = stopping_index))
            else:
                env.process(OrdergeneratorByCSVForStressTest(env, Orders, Store_dict, stress_lamda, platform=Platform2,
                                                             p2_ratio=customer_p2, rider_speed=rider_speed,
                                                             unit_fee=unit_fee, fee_type=fee_type,
                                                             output_data=CustomerCoord, cooktime_detail= None, customer_pend = customer_pend, manual_cook_time = manual_cook_time))
            """
            env.process(RiderGenerator(env, Rider_dict, Platform2, Store_dict, Orders, capacity=rider_capacity, speed=rider_speed,working_duration=run_time, interval=0.01,
                           gen_num=stress_rider_num,  wait_para=wait_para, platform_recommend = sc.platform_recommend, input_order_select_type = order_select_type,
                                       bundle_construct= sc.rider_bundle_construct))
        else:
            GenerateStoreByCSVStressTest(env, 200, Platform2, Store_dict, store_type=instance_type, ITE=ite,
                                         output_data=StoreCoord, customer_pend=customer_pend, csv_dir = sc.store_dir)
            env.process(OrdergeneratorByCSV(env, sc.customer_dir, Orders, Store_dict, Platform2, p2_ratio=customer_p2,
                                            rider_speed=rider_speed, unit_fee=unit_fee, fee_type=fee_type,
                                            service_time_diff=service_time_diff))
            env.process(RiderGeneratorByCSV(env, sc.rider_dir,  Rider_dict, Platform2, Store_dict, Orders, input_speed = rider_speed, input_capacity= rider_capacity,
                                            platform_recommend = sc.platform_recommend, input_order_select_type = order_select_type, bundle_construct= sc.rider_bundle_construct,
                                            rider_num = rider_num, lamda_list=lamda_list, p2 = rider_p2, rider_select_print_fig = rider_select_print_fig,ite = rv_count, mix_ratio = sc.mix_ratio))
        #env.process(OrdergeneratorByCSV(env, sc.customer_dir, Orders, Store_dict, Platform2, p2_ratio = customer_p2,rider_speed= rider_speed, unit_fee = unit_fee, fee_type = fee_type, service_time_diff = service_time_diff))
        if dynamic_env == False or sc.platform_recommend == True:
            env.process(Platform_process5(env, Platform2, Orders, Rider_dict, platform_p2,thres_p,interval, bundle_para= sc.platform_recommend, obj_type = sc.obj_type,
                                          search_type = sc.search_type, print_fig = print_fig, bundle_print_fig = bundle_print_fig, bundle_infos = bundle_infos,
                                          ellipse_w = ellipse_w, heuristic_theta = heuristic_theta,heuristic_r1 = heuristic_r1,XGBmodel3 = XGBmodel3, XGBmodel2 = XGBmodel2, thres_label = thres_label,
                                          considered_customer_type = considered_customer_type, search_range_index= search_range_index, pr_para = pr_para, ML_Saved_Data_B2=ML_Saved_Data_B2,
                                          ML_Saved_Data_B3=ML_Saved_Data_B3, fix_start = bundle_start_fix, ite = ite, cut_info3= cut_info3, cut_info2= cut_info2, stopping_index = stopping_index,
                                          clustering = clustering_para, revise_type = revise_type_para, cut_infoC = cut_infoC, search_type2 = search_type2, customer_pend = customer_pend))
        else:
            env.process(DefreezeCustomers(env, Orders, Rider_dict, Platform2, end_t = run_time, interval = interval, customer_pend = customer_pend))
        env.run(run_time)
        #번들 종류 count함수
        b_type_count = [0,0,0]
        b_type_len_sum = [0,0,0]
        print(Platform2.selected_bundle_type)
        #input('check')
        for select_info in Platform2.selected_bundle_type:
            if select_info[1] == 1:
                b_type_count[0] += 1
                b_type_len_sum[0] += select_info[2]
            elif select_info[1] == 2:
                b_type_count[1] += 1
                b_type_len_sum[1] += select_info[2]
            else:
                b_type_count[2] += 1
                b_type_len_sum[2] += select_info[2]
                pass
            #b_type_count[select_info[1]] += 1
            #b_type_len_sum[select_info[1]] += select_info[2]
        sc.bundle_type_infos = b_type_count + b_type_len_sum
        res = ResultPrint(sc.name + str(ite), Orders, speed=rider_speed, riders = Rider_dict)
        sc.res.append(res)
        #end_time_sec = time.time()
        end_time_sec = datetime.now()
        duration = end_time_sec - start_time_sec
        sc.durations.append(duration.seconds)
        sc.bundle_snapshots['size'] += bundle_infos['size']
        sc.bundle_snapshots['length'] += bundle_infos['length']
        sc.bundle_snapshots['od'] += bundle_infos['od']
        sc.countf[0] = counter.dist1
        sc.countf[1] = counter.bundle_consist
        sc.countf[2] = counter.bundle_consist2
        sc.countf[3] = counter.dist3
        sc.countt[0] = t_counter.t1
        sc.countt[1] = t_counter.t2
        sc.countt[2] = t_counter.t3
        sc.countt[3] = t_counter.t4
        print('Name :: dist :: p2 :: ratio')
        for ct_num in Orders:
            ct = Orders[ct_num]
            try:
                print(ct_num, '::', distance(ct.location[0],ct.location[1], ct.store_loc[0],ct.store_loc[1]), '::', ct.p2, '::',
                      distance(ct.location[0],ct.location[1], ct.store_loc[0],ct.store_loc[1]) / ct.p2)
            except:
                pass
        print(Platform2.suggested_bundles)
        #input('확인 ㅎ')
        sc.suggested_bundles_count = copy.deepcopy(Platform2.suggested_bundles)
        #input('확인')

        #저장 부
        res = []
        wait_time = 0
        candis = []
        b_select = 0
        store_wait_time = 0
        bundle_store_wait_time = []
        single_store_wait_time = []
        served_num = 0
        check_data = []
        rider_moving_time = []
        rider_fee = []
        for i in range(100):
            check_data.append(str(i)+';')
        for rider_name in Rider_dict:
            rider = Rider_dict[rider_name]
            res += rider.served
            wait_time += rider.idle_time
            rider_fee.append(rider.income)
            #candis += rider.candidates
            b_select += rider.b_select
            store_wait_time += rider.store_wait
            bundle_store_wait_time += rider.bundle_store_wait
            single_store_wait_time += rider.single_store_wait
            served_num += len(rider.served)
            #print('라이더 {} 경로 :: {}'.format(rider.name, rider.visited_route))
            check_t = 0
            #print('{};{};{};{};'.format(rider.visited_route[0][2][0],rider.visited_route[0][2][1], 0,check_t,rider.visited_route[0][3]))
            #check_data[rider_name][0] += ['x','y','계산시간','기록시간']
            check_data[0] += 'x;y;계산시간;기록시간;'
            for node_index in range(1,len(rider.visited_route)):
                #input('기록')
                check_t += distance(rider.visited_route[node_index-1][2][0],rider.visited_route[node_index-1][2][1],rider.visited_route[node_index][2][0],rider.visited_route[node_index][2][1])/rider_speed
                #check_data[rider_name][node_index] += [rider.visited_route[node_index-1][2][0],rider.visited_route[node_index-1][2][1], round(check_t,2), rider.visited_route[node_index-1][3]]
                tem_info = '{};{};{};{};'.format(rider.visited_route[node_index - 1][2][0], rider.visited_route[node_index - 1][2][1], round(check_t, 2),rider.visited_route[node_index - 1][3])
                check_data[node_index] += tem_info
                #print('{};{};{};{};'.format(rider.visited_route[node_index-1][2][0],rider.visited_route[node_index-1][2][1], round(check_t,2), rider.visited_route[node_index-1][3]))
            #print('라이더 {} 페이지 선택 난수 :: {}'.format(rider.name, rider.pages_history))
            #라이더 경로 그림 그리기
            x1 = []
            y1 = []
            x2 = []
            y2 = []
            # 3 확인
            for index in range(1, len(rider.visited_route)):
                start = rider.visited_route[index - 1][2]
                end = [rider.visited_route[index][2][0] - rider.visited_route[index - 1][2][0],
                       rider.visited_route[index][2][1] - rider.visited_route[index - 1][2][1]]
                plt.arrow(start[0], start[1], end[0], end[1], width=0.2, length_includes_head=True)
            for ct_name in rider.served:
                x1.append(Orders[ct_name].store_loc[0])
                y1.append(Orders[ct_name].store_loc[1])
                x2.append(Orders[ct_name].location[0])
                y2.append(Orders[ct_name].location[1])
            plt.scatter(x1, y1, marker='o', color='k', label='store')
            plt.scatter(x2, y2, marker='x', color='m', label='customer')
            plt.legend()
            plt.axis([0, 50, 0, 50])
            title = 'H: {}RiderBundle {} ;Rider {};T {}'.format(sc.search_type, rider.bundle_construct, rider.name, round(env.now, 2))
            plt.title(title)
            #plt.savefig(title + '.png', dpi=1000)
            #plt.show()
            #input('라이더 선택 확인2')
            plt.close()
            rider_moving_time.append(check_t)
        print('시나리오 결과 정리 시작')
        ave_moving_t = np.mean(rider_moving_time)
        print(sc.res)
        print(sc.res[-1])
        sc.res[-1].append(ave_moving_t)
        sc.res[-1].append(np.mean(rider_fee))
        print('아마도 에러..?')
        for info in check_data:
            print(info)
        wait_time_per_customer = bundle_store_wait_time + single_store_wait_time
        try:
            wait_time_per_customer = round(sum(wait_time_per_customer) / len(wait_time_per_customer), 2)
        except:
            wait_time_per_customer = None
        if len(bundle_store_wait_time) > 0:
            bundle_store_wait_time = round(sum(bundle_store_wait_time) / len(bundle_store_wait_time), 2)
        else:
            bundle_store_wait_time = None
        if len(single_store_wait_time) > 0:
            single_store_wait_time = round(sum(single_store_wait_time) / len(single_store_wait_time), 2)
        else:
            single_store_wait_time = None
        ave_wait_time = round(wait_time / len(Rider_dict), 2)
        try:
            print(
                '라이더 수 ;{} ;평균 수행 주문 수 ;{} ;평균 유휴 분 ;{} ;평균 후보 수 {} 평균 선택 번들 수 {} 가게 대기 시간 {} 번들가게대기시간 {} 단건가게대기시간 {} 고객 평균 대기 시간 {}'.format(
                    len(Rider_dict), round(len(res) / len(Rider_dict), 2), round(wait_time / len(Rider_dict), 2),
                    round(sum(candis) / len(candis), 2), b_select / len(Rider_dict),
                    round(store_wait_time / len(Rider_dict), 2), bundle_store_wait_time, single_store_wait_time,
                    wait_time_per_customer))
        except:
            print('에러 발생으로 프린트 제거')
            #input('WHY?')
        res_info = sc.res[-1]
        try:
            info = str(sc.name) + ';' + str(ite) + ';' + str(res_info[0]) + ';' + str(res_info[1]) + ';' + str(
                res_info[2]) + ';' + str(res_info[3]) + ';' + str(res_info[4]) + ';' + str(
                round(res_info[5], 4)) + ';' + str(ave_wait_time) + ';' + str(b_select) + ';'+ str(res_info[9]) +';'+\
                   str(sc.countf[0])+';'+str(sc.countf[1])+';'+str(sc.countf[2])+';'+'\n'

            f = open("결과저장0706.txt", 'a')
            f.write(info)
            f.close()
        except:
            pass
        # input('파일 확인')
        sub_info = 'divide_option : {}, p2: {}, divide_option: {}, unserved_order_break : {}, dynamic : {}, platform_recommend : {}, obj : {}'.format(divide_option, platform_p2,
                                                                                                     sc.platform_work,
                                                                                                     sc.unserved_order_break, dynamic_env,sc.platform_recommend,sc.obj_type)
        ResultSave(Rider_dict, Orders, title='Test', sub_info=sub_info, type_name=sc.name)
        # input('저장 확인')
        # 시나리오 저장
        #SaveInstanceAsCSV(Rider_dict, Orders, Store_dict, instance_name='res')
        #결과 저장 부
        tm = time.localtime()
        string = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
        try:
            info = [string, ite, sc.name, sc.considered_customer_type, sc.unserved_order_break, sc.scoring_type, sc.bundle_selection_type, 0, \
            sc.res[-1][0],sc.res[-1][1], sc.res[-1][2], sc.res[-1][3], sc.res[-1][4], sc.res[-1][5], sc.res[-1][6], sc.res[-1][7], sc.res[-1][8]]
        except:
            info = ['N/A']
        #[len(customers), len(TLT),served_ratio,av_TLT,av_FLT, av_MFLT, round(sum(MFLT)/len(MFLT),2), rider_income_var,customer_lead_time_var]
        f = open("InstanceRES.csv", 'a', newline='')
        wr = csv.writer(f)
        wr.writerow(info)
        f.close()
        tem = []
        tem2 = []
        tem3 = []
        canceled_ct = 0
        for customer_name in Orders:
            customer = Orders[customer_name]
            if customer.time_info[3] != None:
                tem.append(customer.time_info[3] - customer.time_info[0])
                tem2.append(customer.time_info[3] - customer.time_info[2])
                p2p_time = distance(customer.store_loc[0],customer.store_loc[1], customer.location[0],customer.location[1])/rider_speed
                try:
                    over_ratio = (customer.time_info[3] - customer.time_info[2])/ p2p_time
                except:
                    over_ratio = 1
                if customer.type == 'bundle' and over_ratio > 1:
                    tem3.append(over_ratio)
            else:
                if customer.cancel == True and customer.time_info[0]  < run_time - interval*2:
                    canceled_ct += 1
            print('문제 발생 고객 시작')
            if len(customer.who_picked) > 1:
                print(customer.name, customer.who_picked)
            print('문제 발생 고객 종료')
        lead_time_stroage.append(tem)
        foodlead_time_stroage.append(tem2)
        foodlead_time_ratio_stroage.append(tem3)
        num_bundle = 0
        near_bundle = []
        snapshot_dict = {1:0,2:0,3:0}
        for rider in Rider_dict:
            num_bundle += sum(Rider_dict[rider].bundle_count)
            for snapshot_info in Rider_dict[rider].snapshots:
                if snapshot_info[1] != None:
                    #input('확인{}'.format(snapshot_info))
                    snapshot_dict[snapshot_info[7]] += 1
                    near_bundle.append(snapshot_info[8])
            for b_info in Rider_dict[rider].bundles_infos:
                sc.bundle_select_infos[b_info[1]] += 1
            sc.bundle_select_infos[2] += Rider_dict[rider].count_info[0]
            sc.bundle_select_infos[3] += Rider_dict[rider].count_info[1]
        num_bundles.append(num_bundle)
        if save_budnle_as_file == True:
            #번들 그림 확인.
            for rider in Rider_dict:
                if len(Rider_dict[rider].bundles_infos) > 0:
                    Rider_Bundle_plt(Rider_dict[rider])
        try:
            print('페이지 밖 번들 순위 {}'.format(sum(near_bundle)/len(near_bundle)))
        except:
            print('페이지 밖 번들이 없음 {}'.format(near_bundle))
        print('번들 정보',snapshot_dict)
        #input('확인')
        SaveScenario(sc, len(Rider_dict), instance_type, ite, considered_customer_type = considered_customer_type, search_range_index = search_range_index, pr_para = pr_para, add_info= [canceled_ct], dynamic=dynamic_env)
        #탐색한 번들 저장
        b_size = 2
        for ML_Saved_Data in [ML_Saved_Data_B2, ML_Saved_Data_B3]:
            label_datas = []
            ML_count = 0
            label1_names = []
            label1_infos = []
            for data in ML_Saved_Data:
                # ver1: [route, unsync_t[0], round(sum(ftds) / len(ftds), 2), unsync_t[1], order_names, round(route_time, 2),min(time_buffer), round(P2P_dist - route_time, 2), round(route_time, 2) - 시작점 지점과 끝점 이동 시간]
                tem = [ML_count, len(data[4])]
                tem += data[4]
                tem += data[0]
                tem += [data[2]]
                tem += [data[5]]
                tem += [data[6]]  # todo: 추가된 부분
                # 거리 계산하기
                # print('거리 계산',Orders[data[0][0]- 1000].store_loc, Orders[data[0][-1]].location)
                # origin = Orders[data[0][0]- 1000].store_loc
                # destination = Orders[data[0][-1]].location
                # line_dist = data[5] - distance(origin[0],origin[1],destination[0],destination[1])/rider_speed
                tem += [data[8]]
                label_datas.append(tem)
                label1_names.append(data[4])
                label1_infos.append([data[8], data[5], data[9], data[10]])
                points = []
                vectors = []
                triangles = []
                for name in data[4]:
                    points += [Orders[name].store_loc, Orders[name].location]
                    vectors += [Orders[name].store_loc[0] - Orders[name].location[0],
                                Orders[name].store_loc[1] - Orders[name].location[1]]
                # label1_infos.app
                ML_count += 1
            if save_data == True:
                label_datas_np = np.array(label_datas)
                label1_data = BundleFeaturesCalculator2(Orders, label1_names, label=1, add_info=label1_infos, print_option=False)
                raw_data = label1_data
                raw_data_np = np.array(raw_data, dtype=np.float64)
                # np.save('./GXBoost'+str(gen_B_size)+'/'+save_id+'raw_data_np_'+instance_type_i+'_'+str(gen_B_size), raw_data_np)
                np.save(save_root_dir + str(ite) + 'raw_data_np_' + instance_type + '_' + str(b_size)+'_'+save_id, raw_data_np)
            b_size += 1

    rev_label = []
    count1 = 0
    for label in labels:
        rev_label.append(label + ':'+str(num_bundles[count1]))
        count1 += 1
    if save_as_file == True:
        plt.boxplot(lead_time_stroage, labels=rev_label, showmeans=True)
        name = 'LT_ITE{};ID{}'.format(ite, random.random())
        plt.savefig('Figure/' + name+'.png', dpi=1000)
        plt.close()
        plt.boxplot(foodlead_time_stroage, labels=rev_label, showmeans=True)
        name = 'FLT_ITE{};ID{}'.format(ite, random.random())
        plt.savefig('Figure/' + name+'.png', dpi=1000)
        plt.close()
        plt.boxplot(foodlead_time_ratio_stroage, labels=rev_label, showmeans=True)
        name = 'FLT_ratio_ITE{};ID{}'.format(ite, random.random())
        plt.savefig('Figure/' + name+'.png', dpi=1000)
        plt.close()
    ##b2번들 확인
    b_count = 2
    for b_infos in [check_list.b2, check_list.b3]:
        f = open("번들 저장.txt", 'a')
        f.write('Start;'+str(instance_type) +';'+ str(heuristic_type) +';'+ str(b_count)+ '\n')
        for info in b_infos:
            con = '{};{};'.format(info[0],info[-1])
            f.write(con + '\n')
        f.close()
        b_count += 1
    print('uniqe 번들 수::',len(check_list.suggested_bundle))
    print(check_list.suggested_bundle)


#input('테스트 종료')
for sc in scenarios:
    count = 1
    for res_info in sc.res:
        try:
            print(
                'SC:{}/플랫폼번들{}/라이더번들{}/ITE ;{}; /전체 고객 ;{}; 중 서비스 고객 ;{};/ 서비스율 ;{};/ 평균 LT ;{};/ 평균 FLT ;{};/직선거리 대비 증가분 ;'
                '{};원래 O-D길이;{};라이더 수익 분산;{};LT분산;{};OD증가수;{};OD증가 분산;{};OD평균;{}'.format(
                    sc.name, sc.platform_recommend,sc.rider_bundle_construct, count, res_info[0],
                    res_info[1], res_info[2], res_info[3], res_info[4], res_info[5], res_info[6], res_info[7], res_info[8],res_info[9],res_info[10],res_info[11]))
        except:
            print('시나리오 {} ITE {} 결과 없음'.format(sc.name, count))
        count += 1
print('"요약 정리/ 라이더 수 {}'.format(rider_num))
print_count = 0
f3 = open("결과저장1209.txt", 'a')
f3.write('결과저장 시작' + '\n')
for sc in scenarios:
    res_info = []
    #input(sc.res)
    for index in list(range(len(sc.res[0]))):
        tem = []
        for info in sc.res:
            if type(info) == list:
                tem.append(info[index])
            else:
                #print(info)
                pass
        if None in tem:
            res_info.append(None)
        else:
            res_info.append(sum(tem)/len(tem))
    try:
        res_info.append(sum(sc.bundle_snapshots['size'])/len(sc.bundle_snapshots['size']))
        res_info.append(sum(sc.bundle_snapshots['length']) / len(sc.bundle_snapshots['length']))
        res_info.append(sum(sc.bundle_snapshots['od']) / len(sc.bundle_snapshots['od']))
    except:
        res_info += [None,None,None]
    offered_bundle_num = len(sc.bundle_snapshots['size'])
    #print(len(res_info))
    #input(res_info)
    local_t = str(time.ctime(time.time()))
    if print_count == 0:
        f3.write('considered_customer_type;{};search_range_index;{};pr_para;{}; \n'.format(considered_customer_type, search_range_index,pr_para))
        head = 'local_t;customer_pend;dynamic;sc.platform_recommend;인스턴스종류;SC;번들탐색방식;연산시간(sec);플랫폼;라이더;라이더수;obj;전체 고객;서비스된 고객;서비스율;평균LT;평균FLT;직선거리 대비 증가분;원래 O-D길이;라이더 수익 분산;LT분산;' \
               'OD증가수;OD증가 분산;OD평균;수행된 번들 수;수행된번들크기평균;b1;b2;b3;b4;b5;b수;p1;p2;p3;p4;p수;r1;r2;r3;r4;r5;r수;평균서비스시간;(테스트)음식 대기 시간;(테스트)버려진 음식 수;(테)음식대기;' \
               '(테)라이더대기;(테)15분이하 음식대기분;(테)15분이상 음식대기분;(테)15분이하 음식대기 수;(테)15분이상 음식대기 수;(테)라이더 대기 수;라이더평균운행시간;제안된 번들수;라이더수수료;size;length;ods;ellipse_w; ' \
               'heuristic_theta; heuristic_r1;rider_ratio;#dist;#bc1;#bc2;#dist(xgb);#t1;#t2;#t3;예상X라이더 선택;예상O라이더 선택;예상X라이더 주문 선택;예상한 라이더 주문 선택;그외;Dynamic_B;Static_B;ave_dynamic;ave_static;' \
               'dtnamic번들 제안;Static번들 제안;'
        #print('인스턴스종류;SC;번들탐색방식;연산시간(sec);플랫폼;라이더;obj;전체 고객;서비스된 고객;서비스율;평균LT;평균FLT;직선거리 대비 증가분;원래 O-D길이;라이더 수익 분산;LT분산;'
        #     'OD증가수;OD증가 분산;OD평균;수행된 번들 수;수행된번들크기평균;제안된 번들수;size;length;ods')
        print(head)
        f3.write(head + '\n')
    ave_duration = sum(sc.durations)/len(sc.durations)
    ave_dynamic = 0
    ave_static = 0
    try:
        ave_dynamic = sc.bundle_type_infos[3]/sc.bundle_type_infos[0]
        ave_static = sc.bundle_type_infos[4]/sc.bundle_type_infos[1]
    except:
        pass
    print(sc.suggested_bundles_count)
    #input('확인 ㅎ')
    try:
        tem_data = '{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};'.format(
                local_t,customer_pend,dynamic_env, sc.platform_recommend,instance_type , str(sc.name[0]),sc.search_type, ave_duration,sc.platform_recommend,sc.rider_bundle_construct,stress_rider_num,sc.obj_type, res_info[0],res_info[1],
                res_info[2], res_info[3], res_info[4], res_info[5], res_info[6], res_info[7], res_info[8],res_info[9],res_info[10],res_info[11],res_info[12],res_info[13],
            res_info[14], res_info[15], res_info[16],res_info[17], res_info[18], res_info[19],res_info[20],res_info[21],res_info[22],res_info[23], res_info[24],res_info[25],
            res_info[26],res_info[27],res_info[28],res_info[29],res_info[30],res_info[31],res_info[32], res_info[33],res_info[34], res_info[35],res_info[36], res_info[37],res_info[38],res_info[39], res_info[40], res_info[41],
            offered_bundle_num,res_info[42], res_info[43], res_info[44],res_info[45],ellipse_w, heuristic_theta, heuristic_r1, sc.mix_ratio, sc.countf[0], sc.countf[1], sc.countf[2], sc.countf[3],
        sc.countt[0], sc.countt[1],sc.countt[2], sc.bundle_select_infos[0], sc.bundle_select_infos[1],sc.bundle_select_infos[2], sc.bundle_select_infos[3], sc.bundle_type_infos[2],sc.bundle_type_infos[0],sc.bundle_type_infos[1],
        ave_dynamic, ave_static,sc.suggested_bundles_count[0],sc.suggested_bundles_count[1])
        """
        print(
            '{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}'.format(
                instance_type , str(sc.name[0]),sc.search_type, ave_duration,sc.platform_recommend,sc.rider_bundle_construct,sc.obj_type, res_info[0],res_info[1],
                res_info[2], res_info[3], res_info[4], res_info[5], res_info[6], res_info[7], res_info[8],res_info[9],res_info[10],res_info[11],res_info[12],res_info[13],
            offered_bundle_num,res_info[14], res_info[15], res_info[16]))        
        """
        print(tem_data)
        f3.write(tem_data + '\n')
    except:
        tem_data = '시나리오 {} ITE {} 결과 없음'.format(sc.name, count)
        #print('시나리오 {} ITE {} 결과 없음'.format(sc.name, count))
        print(tem_data)
        f3.write(tem_data + '\n')
    print_count += 1
f3.write('Exp End' + '\n')
f3.close()

if setting == 'stresstest':
    f3 = open("고객_coord_정보 .txt", 'a')
    f3.write('결과저장 시작' + instance_type + '\n')
    for order_name in Orders:
        order = Orders[order_name]
        con = '{};{};{};{};{};{};\n'.format(order.name,order.location[0],order.location[1], order.store, order.store_loc[0], order.store_loc[1])
        f3.write(con)
    f3.write('Exp End' + '\n')
    f3.close()
    
print("수행 완료")