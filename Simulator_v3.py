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
from A1_BasicFunc import ResultSave, GenerateStoreByCSV, RiderGeneratorByCSV, OrdergeneratorByCSV, distance
from A2_Func import ResultPrint
from re_platform import Platform_process5,Rider_Bundle_plt
from datetime import datetime

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
# Parameter define
interval = 5
run_time = 180
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
rider_speed = 3
rider_capacity = 3
start_ite = 0
ITE_NUM = 1
option_para = True  # True : 가게와 고객을 따로 -> 시간 단축 가능 :: False : 가게와 고객을 같이 -> 시간 증가
customer_max_range = 50
store_max_range = 30
divide_option = True  # True : 구성된 번들에 속한 고객들을 다시 개별 고객으로 나눔. False: 번들로 구성된 고객들은 번들로만 구성
p2_set = True
rider_p2 = 2 #1.5
platform_p2 = rider_p2*0.8  #1.3 p2_set이 False인 경우에는 p2만큼의 시간이 p2로 고정됨. #p2_set이 True인 경우에는 p2*dis(가게,고객)/speed 만큼이 p2시간으로 설정됨.
customer_p2 = 1 #2
obj_types = ['simple_max_s', 'max_s+probability', 'simple_over_lt','over_lt+probability']
# order_p2 = [[1.5,2,3],[0.3,0.3,0.4]] #음식 별로 민감도가 차이남.
wait_para = False  # True: 음식조리로 인한 대기시간 발생 #False : 음식 대기로 인한 대기시간 발생X
scenarios = []
run_para = True  # True : 시뮬레이션 작동 #False 데이터 저장용
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
scenarios = [scenarios[2],copy.deepcopy(scenarios[8])]
#scenarios = [copy.deepcopy(scenarios[8])]
#scenarios[0].search_type = heuristic_type
scenarios[1].search_type = heuristic_type
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
for sc3 in scenarios:
    print(sc3.platform_recommend, sc3.rider_bundle_construct,sc3.obj_type)
#input('시나리오 확인')

#exp_range = [0,2,3,4]*10 #인스턴스 1에러가 있음.
#global exp_range #인스턴스 1에러가 있음.
#instance_type = 'Instance_cluster' #'Instance_cluster' / 'Instance_random'
#input('instance_type {} '.format(instance_type))
#search_type = 'heuristic'
#input('확인 {}'.format(len(scenarios)))

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
        GenerateStoreByCSV(env, sc.store_dir, Platform2, Store_dict)
        env.process(RiderGeneratorByCSV(env, sc.rider_dir,  Rider_dict, Platform2, Store_dict, Orders, input_speed = rider_speed, input_capacity= rider_capacity,
                                        platform_recommend = sc.platform_recommend, input_order_select_type = order_select_type, bundle_construct= sc.rider_bundle_construct,
                                        rider_num = rider_num, lamda_list=lamda_list, p2 = rider_p2, rider_select_print_fig = rider_select_print_fig,ite = rv_count, mix_ratio = sc.mix_ratio))
        env.process(OrdergeneratorByCSV(env, sc.customer_dir, Orders, Store_dict, Platform2, p2_ratio = customer_p2,rider_speed= rider_speed, unit_fee = unit_fee, fee_type = fee_type, service_time_diff = service_time_diff))
        env.process(Platform_process5(env, Platform2, Orders, Rider_dict, platform_p2,thres_p,interval, bundle_para= sc.platform_recommend, obj_type = sc.obj_type,
                                      search_type = sc.search_type, print_fig = print_fig, bundle_print_fig = bundle_print_fig, bundle_infos = bundle_infos,
                                      ellipse_w = ellipse_w, heuristic_theta = heuristic_theta,heuristic_r1 = heuristic_r1))
        env.run(run_time)
        res = ResultPrint(sc.name + str(ite), Orders, speed=rider_speed, riders = Rider_dict)
        sc.res.append(res)
        #end_time_sec = time.time()
        end_time_sec = datetime.now()
        duration = end_time_sec - start_time_sec
        sc.durations.append(duration.seconds)
        sc.bundle_snapshots['size'] += bundle_infos['size']
        sc.bundle_snapshots['length'] += bundle_infos['length']
        sc.bundle_snapshots['od'] += bundle_infos['od']
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
                check_t += distance(rider.visited_route[node_index-1][2],rider.visited_route[node_index][2])/rider_speed
                #check_data[rider_name][node_index] += [rider.visited_route[node_index-1][2][0],rider.visited_route[node_index-1][2][1], round(check_t,2), rider.visited_route[node_index-1][3]]
                tem_info = '{};{};{};{};'.format(rider.visited_route[node_index - 1][2][0], rider.visited_route[node_index - 1][2][1], round(check_t, 2),rider.visited_route[node_index - 1][3])
                check_data[node_index] += tem_info
                #print('{};{};{};{};'.format(rider.visited_route[node_index-1][2][0],rider.visited_route[node_index-1][2][1], round(check_t,2), rider.visited_route[node_index-1][3]))
            print('라이더 {} 페이지 선택 난수 :: {}'.format(rider.name, rider.pages_history))
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
        ave_moving_t = np.mean(rider_moving_time)
        sc.res[-1].append(ave_moving_t)
        sc.res[-1].append(np.mean(rider_fee))
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
        res_info = sc.res[-1]
        try:
            info = str(sc.name) + ';' + str(ite) + ';' + str(res_info[0]) + ';' + str(res_info[1]) + ';' + str(
                res_info[2]) + ';' + str(res_info[3]) + ';' + str(res_info[4]) + ';' + str(
                round(res_info[5], 4)) + ';' + str(ave_wait_time) + ';' + str(b_select) + ';'+ str(res_info[9]) +'\n'

            f = open("결과저장0706.txt", 'a')
            f.write(info)
            f.close()
        except:
            pass
        # input('파일 확인')
        sub_info = 'divide_option : {}, p2: {}, divide_option: {}, unserved_order_break : {}'.format(divide_option, platform_p2,
                                                                                                     sc.platform_work,
                                                                                                     sc.unserved_order_break)
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
        for customer_name in Orders:
            customer = Orders[customer_name]
            if customer.time_info[3] != None:
                tem.append(customer.time_info[3] - customer.time_info[0])
                tem2.append(customer.time_info[3] - customer.time_info[2])
                p2p_time = distance(customer.store_loc, customer.location)/rider_speed
                over_ratio = (customer.time_info[3] - customer.time_info[2])/ p2p_time
                if customer.type == 'bundle' and over_ratio > 1:
                    tem3.append(over_ratio)
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
    if print_count == 0:
        head = '인스턴스종류;SC;번들탐색방식;연산시간(sec);플랫폼;라이더;라이더수;obj;전체 고객;서비스된 고객;서비스율;평균LT;평균FLT;직선거리 대비 증가분;원래 O-D길이;라이더 수익 분산;LT분산;' \
               'OD증가수;OD증가 분산;OD평균;수행된 번들 수;수행된번들크기평균;b1;b2;b3;b4;b5;p1;p2;p3;p4;r1;r2;r3;r4;r5;평균서비스시간;(테스트)음식 대기 시간;(테스트)버려진 음식 수;(테)음식대기;' \
               '(테)라이더대기;(테)15분이하 음식대기분;(테)15분이상 음식대기분;(테)15분이하 음식대기 수;(테)15분이상 음식대기 수;(테)라이더 대기 수;라이더평균운행시간;제안된 번들수;라이더수수료;size;length;ods;ellipse_w; heuristic_theta; heuristic_r1;rider_ratio'
        #print('인스턴스종류;SC;번들탐색방식;연산시간(sec);플랫폼;라이더;obj;전체 고객;서비스된 고객;서비스율;평균LT;평균FLT;직선거리 대비 증가분;원래 O-D길이;라이더 수익 분산;LT분산;'
        #     'OD증가수;OD증가 분산;OD평균;수행된 번들 수;수행된번들크기평균;제안된 번들수;size;length;ods')
        print(head)
        f3.write(head + '\n')
    ave_duration = sum(sc.durations)/len(sc.durations)
    try:
        tem_data = '{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};'.format(
                instance_type , str(sc.name[0]),sc.search_type, ave_duration,sc.platform_recommend,sc.rider_bundle_construct,rider_num,sc.obj_type, res_info[0],res_info[1],
                res_info[2], res_info[3], res_info[4], res_info[5], res_info[6], res_info[7], res_info[8],res_info[9],res_info[10],res_info[11],res_info[12],res_info[13],
            res_info[14], res_info[15], res_info[16],res_info[17], res_info[18], res_info[19],res_info[20],res_info[21],res_info[22],res_info[23], res_info[24],res_info[25],
            res_info[26],res_info[27],res_info[28],res_info[29],res_info[30],res_info[31],res_info[32], res_info[33],res_info[34], res_info[35],res_info[36], res_info[37],res_info[38],
            offered_bundle_num,res_info[39], res_info[40], res_info[41],res_info[42],ellipse_w, heuristic_theta, heuristic_r1, sc.mix_ratio)
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