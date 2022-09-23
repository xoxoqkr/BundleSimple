# -*- coding: utf-8 -*-
import random
import time
import math
from datetime import datetime
from A1_BasicFunc import PrintSearchCandidate, check_list, t_counter, counter2, counter
from A2_Func import CountUnpickedOrders, CalculateRho, RequiredBreakBundleNum, BreakBundle, GenBundleOrder,  LamdaMuCalculate, NewCustomer
from A3_two_sided import BundleConsideredCustomers, CountActiveRider,  ConstructFeasibleBundle_TwoSided, SearchRaidar_heuristic, SearchRaidar_ellipse, SearchRaidar_ellipseMJ, XGBoost_Bundle_Construct, XGBoost_Bundle_Construct2, BundleConsist2
import operator
from Bundle_selection_problem import Bundle_selection_problem3, Bundle_selection_problem4
import numpy
import matplotlib.pyplot as plt

def distance(p1, p2):
    """
    Calculate 4 digit rounded euclidean distance between p1 and p2
    :param p1:
    :param p2:
    :return: 4 digit rounded euclidean distance between p1 and p2
    """
    euc_dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return round(euc_dist,4)


def Platform_process5(env, platform, orders, riders, p2,thres_p,interval, end_t = 1000,
                      divide_option = False,unserved_bundle_order_break = True, bundle_para = False,
                      delete_para = True, obj_type = 'simple_max_s', search_type = 'enumerate', print_fig = False,
                      bundle_print_fig = False, bundle_infos = None,ellipse_w = 1.5, heuristic_theta = 100,heuristic_r1 = 10,
                      XGBmodel3 = None, XGBmodel2 = None, thres_label = 1, considered_customer_type = 'all',search_range_index = 15,
                      pr_para = False):
    f = open("loop시간정보.txt", 'a')
    f.write('연산 시작; obj;{};searchTypt;{};threslabel;{};#Riders;{};'.format(obj_type,search_type,thres_label,len(riders)))
    f.write('\n')
    f.close()
    yield env.timeout(5) #warm-up time
    while env.now <= end_t:
        #loop_s = datetime.now()
        loop_s = time.time()
        mip_duration = None
        lens_b = None
        num_const = None
        unique_bundle_num = 0
        active_rider_names = []
        if bundle_para == True:
            lamda1, lamda2, mu1, mu2 = LamdaMuCalculate(orders, riders, env.now, interval=interval, return_type='class')
            p = CalculateRho(lamda1, lamda2, mu1, mu2)
            active_rider_names, d_infos = CountActiveRider(riders, interval, min_pr= 0.05, t_now=env.now, option='w', point_return=True)
            if p > thres_p:
                #pr_para = True
                #if search_type == 'XGBoost' or search_type == 'enumerate':
                #    pr_para = False
                #feasible_bundle_set, phi_b, d_matrix, s_b, D, lt_matrix = Bundle_Ready_Processs(env.now, platform, orders, riders, p2, interval,
                #                                                                                speed = riders[0].speed, bundle_permutation_option= True, search_type = search_type, print_fig = print_fig,
                #                                                                                ellipse_w = ellipse_w, heuristic_theta = heuristic_theta,heuristic_r1 = heuristic_r1)

                feasible_bundle_set, phi_b, d_matrix, s_b, D, lt_matrix, act_considered_ct_num, label_infos = Bundle_Ready_Processs2(env.now, platform, orders, riders, p2, interval,
                                                                                                speed = riders[0].speed, bundle_permutation_option= True, search_type = search_type, print_fig = print_fig,
                                                                                                ellipse_w = ellipse_w, heuristic_theta = heuristic_theta,heuristic_r1 = heuristic_r1, XGBmodel3 = XGBmodel3,
                                                                                                                        XGBmodel2 = XGBmodel2, considered_customer_type = considered_customer_type,
                                                                                                                                     thres_label=thres_label,pr_para=pr_para, search_range_index = search_range_index)
                """
                feasible_bundle_set, phi_b, d_matrix, s_b, D, lt_matrix, act_considered_ct_num, label_infos = Bundle_Ready_Processs3(
                    env.now, platform, orders, riders, p2, interval,
                    speed=riders[0].speed, bundle_permutation_option=True, search_type=search_type, print_fig=print_fig,
                    ellipse_w=ellipse_w, heuristic_theta=heuristic_theta, heuristic_r1=heuristic_r1,
                    XGBmodel3=XGBmodel3,
                    XGBmodel2=XGBmodel2, considered_customer_type='new', thres_label=thres_label)
                """
                print('phi_b {}:{} d_matrix {}:{} s_b {}:{}'.format(len(phi_b), numpy.average(phi_b),
                                                                    d_matrix.shape, numpy.average(d_matrix),len(s_b),numpy.average(s_b),))
                print('d_matrix : {}'.format(d_matrix))
                #문제 풀이
                #unique_bundle_indexs = Bundle_selection_problem3(phi_b, d_matrix, s_b, min_pr = 0.05)
                mip_s = time.time()
                unique_bundle_indexs, num_const = Bundle_selection_problem4(phi_b, D, s_b, lt_matrix, min_pr = 1, obj_type= obj_type, pr_para=pr_para) #todo : 0317_수정본. min_pr을 무의미한 제약식으로 설정
                mip_end = time.time()
                mip_duration = mip_end - mip_s

                start_test9 = time.time()

                lens_b = len(s_b)
                print(len(s_b), num_const)
                #input('제약식 확인')
                if len(feasible_bundle_set) > 0:
                    print('T',int(env.now), '가능 번들 수:',len(feasible_bundle_set) )
                    print('선택된 번들',unique_bundle_indexs)
                    #input('feasible_bundle_set 확인')
                #input('결과 확인')
                unique_bundles = []
                for index in unique_bundle_indexs:
                    unique_bundles.append(feasible_bundle_set[index])
                print('문제 풀이 결과 {} '.format(unique_bundles[:10]))
                # 번들을 업로드
                task_index = max(list(platform.platform.keys())) + 1
                unique_bundle_num = len(unique_bundles)
                if len(unique_bundles) > 0:
                    check_list('unique', unique_bundles)
                    #플랫폼에 새로운 주문을 추가하는 작업이 필요.
                    print('주문 수 {} :: 추가 주문수 {}'.format(len(platform.platform),len(unique_bundles)))
                    x1 = []
                    y1 = []
                    x2 = []
                    y2 = []
                    #input('그림 확인 시작2')
                    for info in unique_bundles:
                        #input('info 확인{}'.format(info))
                        ods = 0
                        for customer_name in info[4]:
                            ods += distance(orders[customer_name].location,orders[customer_name].store_loc)/riders[0].speed
                        bundle_infos['size'].append(len(info[4]))
                        bundle_infos['length'].append(info[5])
                        bundle_infos['od'].append(ods)
                        #bundle_infos.append([len(info[4]), info[5], ods])
                        o = GenBundleOrder(task_index, info, orders, env.now)
                        o.old_info = info
                        platform.platform[task_index] = o
                        task_index += 1
                        seq_x = []
                        seq_y = []
                        #print(o.route)
                        if  bundle_print_fig == True:
                            for index in range(1, len(o.route)):
                                start = o.route[index-1][2]
                                end = [o.route[index][2][0] - o.route[index-1][2][0],
                                       o.route[index][2][1] - o.route[index-1][2][1]]
                                plt.arrow(start[0], start[1], end[0], end[1] ,width=0.2, length_includes_head = True)
                            for ct_name in o.customers:
                                x1.append(orders[ct_name].store_loc[0])
                                y1.append(orders[ct_name].store_loc[1])
                                x2.append(orders[ct_name].location[0])
                                y2.append(orders[ct_name].location[1])
                        #if bundle_infos != None:
                        #    bundle_infos.append([len(o.customers),])
                    if bundle_print_fig == True:
                        plt.scatter(x1, y1, marker='o', color='k', label='store')
                        plt.scatter(x2, y2, marker='x', color='m', label='customer')
                        plt.legend()
                        plt.axis([0, 50, 0, 50])
                        title = 'ST;{};T;{};;Selected Bundle Size;{}'.format(search_type, round(env.now, 2),
                                                                             len(unique_bundles))
                        plt.title(title)
                        plt.savefig(title+'.png',dpi = 1000)
                        plt.show()
                        input('계산된 번들 확인')
                        plt.close()
                    #선택된 번들의 그래프 그리기
                    """
                    new_orders = PlatformOrderRevise4(unique_bundles, orders, platform, now_t=env.now,
                                                      unserved_bundle_order_break=unserved_bundle_order_break,
                                                      divide_option=divide_option)                    
                    print(new_orders)
                    input('주문 수 {} -> {} : 추가 번들 수 {}'.format(len(platform.platform), len(new_orders), len(unique_bundles)))
                    platform.platform = new_orders                    
                    """
                    print('주문 수2 {}'.format(len(platform.platform)))
                end_test9 = time.time()
                t_counter('test9', end_test9 - start_test9)
            else:
                start_test10 = time.time()
                print('번들 삭제 수행')
                org_bundle_num, rev_bundle_num = RequiredBreakBundleNum(platform, lamda2, mu1, mu2, thres=thres_p)
                Break_the_bundle(platform, orders, org_bundle_num, rev_bundle_num)
                end_test10 = time.time()
                t_counter('test10', end_test10 - start_test10)
        loop_e = time.time()
        yield env.timeout(interval)
        loop_s1 = time.time()
        if bundle_para == True and delete_para == True:
            #input('경로 지우기')
            delete_task_names = []
            for task_name in platform.platform:
                if len(platform.platform[task_name].customers) > 1 : # and platform.platform[task_name].picked == False:
                    delete_task_names.append(task_name)
            for task_name in delete_task_names:
                del platform.platform[task_name]
        #loop_e = datetime.now()
        loop_e1 = time.time()
        #duration = loop_e - loop_s
        #duration = duration.seconds+ duration.microseconds/1000000
        duration = (loop_e1 - loop_s)  #전체 실험이 돌아가는 시간
        duration1 = (loop_e - loop_s)  # 알고리즘 part 1
        duration2 = (loop_e1 - loop_s1) #알고리즘 Part 2
        duration3 = (loop_s1 - loop_e) #시뮬레이션 시간
        f = open("loop시간정보.txt", 'a')
        if env.now < 6 + interval:
            f.write('considered_customer_type;{};search_range_index;{};pr_para;{}; \n'.format(considered_customer_type, search_range_index,pr_para))
            head_str = '현재시간;연산소요시간;플랫폼연산시간1;플랫폼연산시간2;시뮬레이션시간;undone_num;선택을 기다리는 고객 수;전체 고객 수;라이더 수;MIP 푸는데 걸린 시간;의사결정변수;제약식 수;xgboost;old;sess;enumerate;점수계산;sess1;sess2;sess3;enu1;enu2;bundle_consist2호출수;수정된 고려 고객;0;1;2;3;4;unique_bundle#;선택할라이더수;'
            head_str += 'test1;test2;test3;test4;test5;test6;test7;test8;test9;test10;test11;test12;'
            f.write(head_str + '\n')
        unselected_num = 0
        done_num = 0
        for order_name in orders:
            if orders[order_name].time_info[1] == None:
                unselected_num += 1
            if orders[order_name].time_info[4] == None:
                done_num += 1
        try:
            info = '{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};'.format(round(env.now,2),duration,duration1,duration2,duration3, done_num,unselected_num,len(orders), len(riders),mip_duration, lens_b,num_const,t_counter.t1,t_counter.t2,t_counter.t3,t_counter.t4,t_counter.t5,
                                                                    sum(counter2.num1), sum(counter2.num2),sum(counter2.num3),sum(counter2.num4),sum(counter2.num5), counter.bundle_consist2,act_considered_ct_num,label_infos[0],label_infos[1],label_infos[2],label_infos[3],label_infos[4],
                                                                                                                 unique_bundle_num, len(active_rider_names))
            info += '{};{};{};{};{};{};{};{};{};{};{};{};'.format(t_counter.t6,t_counter.t7,t_counter.t8,t_counter.t9,t_counter.t10,t_counter.t11,t_counter.t12,t_counter.t13,t_counter.t14,t_counter.t15,t_counter.t16,t_counter.t17)
            f.write(info)
            f.write('\n')
        except:
            pass
        #info = [round(env.now,2),duration,done_num,unselected_num,len(orders), len(riders)]
        f.close()
        print('Simulation Time : {}'.format(env.now))


def Calculate_Phi(rider, customers, bundle_infos, l=4):
    #라이더와 번들들에 대해서, t시점에서 phi_b...r을 계산 할 것.
    dp_br = []
    dist_list = []
    displayed_values = []
    gamma = 1 - numpy.exp(-rider.search_lamda)
    for customer_name in customers:
        customer = customers[customer_name]
        dist = distance(rider.last_departure_loc,customer.location)
        try:
            displayed_value = customer.fee / ((dist + distance(customer.store_loc, customer.location))/rider.speed)
        except:
            #input('dist {} dist2{} speed{} '.format(dist, distance(customer.store_loc, customer.location), rider.speed))
            pass
        dist_list.append([customer.name, dist, displayed_value])
    dist_list.sort(key = operator.itemgetter(1))
    displayed_dist = []
    for info in dist_list:
        displayed_values.append(info[2])
        displayed_dist.append(info[1])
    dist_list = dist_list[:(len(rider.p_j)+1)*l] #todo : 계산 확인
    for bundle_info in bundle_infos:
        #input(bundle_info)
        bundle_dist = distance(rider.last_departure_loc, customers[bundle_info[4][0]].store_loc)
        bundle_fee = 0
        for customer_name in bundle_info[4]:
            bundle_fee += customers[customer_name].fee
        bundle_value = bundle_fee / bundle_info[6]
        tem_dp_br = []
        for page_index in range(len(rider.p_j)):
            #print('dist_list {} page_index {} displayed_values {} bundle_info {}'.format(dist_list,page_index,displayed_values, bundle_info[6]))
            if bundle_value > max(displayed_values[:(page_index + 1)*l]) and  bundle_dist < max(displayed_dist[:(page_index + 1)*l]):
            #if len(dist_list) <= (page_index + 1)*l or (bundle_dist < dist_list[(page_index + 1)*l][1] and bundle_info[6] > max(displayed_values[:(page_index + 1)*l])):
                tem_dp_br.append(rider.p_j[page_index])
            else:
                tem_dp_br.append(0)
        dp_br.append(1-gamma*sum(tem_dp_br))
        #dp_br.append(gamma * (1 - sum(tem_dp_br)))
    return dp_br

def Calculate_Phi2(rider, customers, ava_customernames, bundle_infos, l=4, M = 10000, search_index = 30, search_dist = 15):
    #라이더와 번들들에 대해서, t시점에서 phi_b...r을 계산 할 것.
    #1 라이더가 주문을 선택하는 시점에서 고려할 주문들 정보 계산
    dist_list = []
    displayed_values = []
    rider_end_point = rider.resource.users[0].loc
    for customer_name in ava_customernames:
        customer = customers[customer_name]
        dist = distance(rider_end_point,customer.location)
        if customer.done == False and customer.cancel == False and dist < search_dist:
            try:
                displayed_value = customer.fee / ((dist + distance(customer.store_loc, customer.location))/rider.speed)
            except:
                pass
        dist_list.append([customer.name, dist, displayed_value])
    dist_list.sort(key = operator.itemgetter(1))
    displayed_dist = []
    for info in dist_list:
        displayed_values.append(info[2])
        displayed_dist.append(info[1])
    #2 우리 번들이 선택될 가능성 탐색
    dp_br = []
    for bundle_info in bundle_infos:
        #input(bundle_info)
        tem_dp_br = []
        bundle_start_point = customers[M - bundle_info[0][0]].store_loc
        bundle_fee = 0
        for customer_name in bundle_info[4]:
            bundle_fee += customers[customer_name].fee
        tobundle = distance(rider_end_point, bundle_start_point)/ rider.speed
        bundle_value = bundle_fee / (tobundle + bundle_info[6]) #번들의 분당 가치
        for page_index in range(len(rider.p_j)):
            # print('dist_list {} page_index {} displayed_values {} bundle_info {}'.format(dist_list,page_index,displayed_values, bundle_info[6]))
            if bundle_value > max(displayed_values[:(page_index + 1) * l]) and bundle_dist < max(
                    displayed_dist[:(page_index + 1) * l]): #라이더에게 번들이 보임
                # if len(dist_list) <= (page_index + 1)*l or (bundle_dist < dist_list[(page_index + 1)*l][1] and bundle_info[6] > max(displayed_values[:(page_index + 1)*l])):
                dp_br.append(rider.p_j[page_index])
            else:
                dp_br.append(0) #라이더에게 번들이 보이지 않음
        return dp_br


def Bundle_Ready_Processs(now_t, platform_set, orders, riders, p2,interval, bundle_permutation_option = False, speed = 1, min_pr = 0.05,
                      unserved_bundle_order_break = True, considered_customer_type = 'all', search_type = 'enumerate', print_fig = False,
                          ellipse_w = 1.5, heuristic_theta = 100,heuristic_r1 = 10):
    # 번들이 필요한 라이더에게 번들 계산.
    if considered_customer_type == 'new':
        considered_customers_names = NewCustomer(orders, now_t, interval=interval)
    else:
        considered_customers_names, interval_orders = CountUnpickedOrders(orders, now_t, interval=interval,return_type='name')
    print('탐색 대상 고객들 {}'.format(considered_customers_names))
    active_rider_names = CountActiveRider(riders, interval, min_pr=min_pr, t_now=now_t, option='w')
    print('돌아오는 시기에 주문 선택 예쌍 라이더 {}'.format(active_rider_names))
    #weight2 = WeightCalculator2(riders, active_rider_names, now_t, interval=interval)
    #sorted_dict = sorted(weight2.items(), key=lambda item: item[1])
    #print('C!@ T {} // 과거 예상 라이더 선택 순서{}'.format(now_t, sorted_dict))
    Feasible_bundle_set = []
    BundleCheck = []
    for index in platform_set.platform:
        task = platform_set.platform[index]
        if len(task.customers) > 1:
            BundleCheck.append([index, len(task.customers)])
            print('Index;{};Cts;{};picked;{};'.format(index, task.customers, task.picked))
    if len(BundleCheck) > 0:
        input('T {};;번들 존재{}'.format(now_t, BundleCheck))
    for customer_name in considered_customers_names:
        start = time.time()
        target_order = orders[customer_name]
        """
        enumerate_C_T = BundleConsideredCustomers(target_order, platform_set, riders, orders,
                                                             bundle_search_variant=unserved_bundle_order_break,
                                                             d_thres_option=True, speed=speed)
        searchRaidar_heuristic_C_T = SearchRaidar_heuristic(target_order, orders, platform_set, theta = heuristic_theta,r1= heuristic_r1,now_t = now_t)
        searchRaidarEllipse_C_T = SearchRaidar_ellipse(target_order, orders, platform_set, w = ellipse_w)        
        """
        #input('image 확인')
        if search_type == 'enumerate':
            #input('확인')
            enumerate_C_T = BundleConsideredCustomers(target_order, platform_set, riders, orders,
                                                      bundle_search_variant=unserved_bundle_order_break,
                                                      d_thres_option=True, speed=speed)
            considered_customers = enumerate_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, enumerate_C_T, now_t=now_t, titleinfo=search_type)
        elif search_type == 'heuristic':
            searchRaidar_heuristic_C_T = SearchRaidar_heuristic(target_order, orders, platform_set, theta=heuristic_theta,
                                                      r1=heuristic_r1, now_t=now_t)
            considered_customers = searchRaidar_heuristic_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, searchRaidar_heuristic_C_T, now_t=now_t, titleinfo=search_type)
        elif search_type == 'ellipse':
            searchRaidarEllipse_C_T = SearchRaidar_ellipse(target_order, orders, platform_set, w=ellipse_w)
            considered_customers = searchRaidarEllipse_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, searchRaidarEllipse_C_T, now_t=now_t, titleinfo=search_type)
        else:
            searchRaidarEllipseMJ_C_T = SearchRaidar_ellipseMJ(target_order, orders, platform_set, delta=ellipse_w)
            considered_customers = searchRaidarEllipseMJ_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, searchRaidarEllipseMJ_C_T, now_t=now_t, titleinfo=search_type)
        print('번들 대상 고객 확인')
        print('T:{}/탐색타입:{} / 번들 탐색 대상 고객들 {}'.format(now_t, search_type, len(considered_customers)))
        thres = 0.1
        size3bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 3, p2, speed=speed, bundle_permutation_option = bundle_permutation_option, thres= thres, now_t = now_t)
        size2bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 2, p2, speed=speed,bundle_permutation_option=bundle_permutation_option , thres= thres, now_t = now_t)
        max_index = 50
        tem_infos = []
        try:
            size3bundle.sort(key=operator.itemgetter(6))
            for info in size3bundle[:max_index]:
                tem_infos.append(info)
        except:
            pass
        try:
            size2bundle.sort(key=operator.itemgetter(6))
            for info in size2bundle[:max_index]:
                tem_infos.append(info)
        except:
            pass
        Feasible_bundle_set += tem_infos
        #Feasible_bundle_set += size3bundle + size2bundle
        end = time.time()
        print('고객 당 계산 시간 {} : B2::{} B3::{}'.format(end - start, len(size2bundle),len(size3bundle)))
        if len(size3bundle + size2bundle) > 1:
            print('번들 생성 가능')
    print('T {} 번들 수 {}'.format(now_t, len(Feasible_bundle_set)))
    #문제에 필요한 데이터 계산
    #1 phi 계산
    phi_br = []
    for rider_name in riders:
        rider = riders[rider_name]
        phi_r = Calculate_Phi(rider, orders, Feasible_bundle_set)
        phi_br.append(phi_r)
        #input('라이더 {} / 확인 phi_r {}'.format(rider.name, phi_r))
    phi_b = []
    for bundle_index in range(len(phi_br[0])):
        tem = 1
        for rider_index in range(len(phi_br)):
            value = phi_br[rider_index][bundle_index]
            if value > 0:
                tem = tem * value
        phi_b.append(tem)
    #input('phi_b {}'.format(phi_b))
    #2 d-matrix계산
    d_matrix = numpy.zeros((len(Feasible_bundle_set),len(Feasible_bundle_set)))
    for index1 in range(len(Feasible_bundle_set)):
        b1 = Feasible_bundle_set[index1][4]
        for index2 in range(len(Feasible_bundle_set)):
            b2 = Feasible_bundle_set[index2][4]
            if index1 > index2 and len(list(set(b1 + b2))) < len(b1) + len(b2) : #겹치는 것이 존재
                    d_matrix[index1, index2] = 1
                    d_matrix[index2, index1] = 1
            else:
                d_matrix[index1, index2] = 0
                d_matrix[index2, index1] = 0
    #3 s_b계산
    s_b = []
    for info in Feasible_bundle_set:
        s_b.append(info[6])
    #4 D 계산
    D = []
    #info1 : [route, round(max(ftds), 2), round(sum(ftds) / len(ftds), 2), round(min(ftds), 2), order_names, round(route_time, 2)]
    info1_index = 0
    info2_index = 0
    #input('D확인1 {}'.format(Feasible_bundle_set))
    for info1 in Feasible_bundle_set:
        for info2 in Feasible_bundle_set:
            if info1 != info2 and len(info1[4]) + len(info2[4]) != len(list(set(info1[4] + info2[4]))):
                #print('b1 {}/ b2 {}/ b1+b2 {}'.format(info1[4],info2[4], list(set(info1[4] + info2[4]))))
                D.append([Feasible_bundle_set.index(info1), Feasible_bundle_set.index(info2)])
            info2_index += 1
        info1_index += 1
    lt_vector = []
    print('D확인2 {}'.format(D[:10]))
    for info in Feasible_bundle_set:
        bundle_names = info[4]
        over_t = []
        for name in bundle_names:
            customer = orders[name]
            over_t.append(now_t - customer.time_info[0])
        lt_vector.append(sum(over_t)/len(over_t))
    return Feasible_bundle_set, phi_b, d_matrix, s_b, D, lt_vector


def Bundle_Ready_Processs2(now_t, platform_set, orders, riders, p2,interval, bundle_permutation_option = False, speed = 1, min_pr = 0.05,
                      unserved_bundle_order_break = True, considered_customer_type = 'all', search_type = 'enumerate', print_fig = False,
                          ellipse_w = 1.5, heuristic_theta = 100,heuristic_r1 = 10, min_time_buffer = 5, XGBmodel3 = None, XGBmodel2 = None, thres_label = 1,
                           MAX_RouteConstruct = 100000, MAX_MIP = 5000, pr_para = True, search_range_index = 15):
    # 번들이 필요한 라이더에게 번들 계산.
    start_test7 = time.time()

    if considered_customer_type == 'new':
        considered_customers_names = NewCustomer(orders, now_t, interval=interval)
    else:
        considered_customers_names, interval_orders = CountUnpickedOrders(orders, now_t, interval=interval,return_type='name')
    print('탐색 대상 고객들 {}'.format(considered_customers_names))
    active_rider_names, d_infos = CountActiveRider(riders, interval, min_pr=min_pr, t_now=now_t, option='w', point_return= True)
    ###todo : 0913 Time save IDEA 01 START
    max_d_list = []
    tem_count = 0
    search_index = search_range_index #이전에는 30
    for p1 in d_infos:
        tem = []
        for order_name in orders:
            order = orders[order_name]
            if order.time_info[1] == None and order.cancel == False:
                tem.append(distance(p1, order.location)/speed)
        tem.sort()
        try:
            max_d = tem[min(search_index,len(tem) - 1)]
        except:
            max_d = 10
        max_d_list += [[active_rider_names[tem_count], p1, max_d]]
    print('max_d_list정보',max_d_list)
    ###todo : 0913 Time save IDEA 01 END
    print('돌아오는 시기에 주문 선택 예쌍 라이더 {}'.format(active_rider_names))
    #input('확인'+ str(now_t))
    Feasible_bundle_set = []
    BundleCheck = []
    for index in platform_set.platform:
        task = platform_set.platform[index]
        if len(task.customers) > 1:
            BundleCheck.append([index, len(task.customers)])
            print('Index;{};Cts;{};picked;{};'.format(index, task.customers, task.picked))
    if len(BundleCheck) > 0:
        input('T {};;번들 존재{}'.format(now_t, BundleCheck))
    check_considered_customers = []
    check_label = numpy.array([])

    end_test7 = time.time()
    t_counter('test7', end_test7 - start_test7)

    for customer_name in considered_customers_names:

        start = time.time()
        start_test8 = time.time()
        target_order = orders[customer_name]
        in_max_d = False
        for d_info in max_d_list:
            dist3 = distance(target_order.store_loc, d_info[1])/ speed
            if dist3 <= d_info[2]:
                in_max_d = True
                target_order.near_rider.append(d_info[0])
                break
        if in_max_d == False:
            continue
        check_considered_customers.append(target_order.name)
        if search_type == 'enumerate':
            #input('확인')
            enumerate_C_T = BundleConsideredCustomers(target_order, platform_set, riders, orders,
                                                      bundle_search_variant=unserved_bundle_order_break,
                                                      d_thres_option=True, speed=speed)
            considered_customers = enumerate_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, enumerate_C_T, now_t=now_t, titleinfo=search_type)
        elif search_type == 'heuristic':
            searchRaidar_heuristic_C_T = SearchRaidar_heuristic(target_order, orders, platform_set, theta=heuristic_theta,
                                                      r1=heuristic_r1, now_t=now_t)
            considered_customers = searchRaidar_heuristic_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, searchRaidar_heuristic_C_T, now_t=now_t, titleinfo=search_type)
        elif search_type == 'ellipse':
            searchRaidarEllipse_C_T = SearchRaidar_ellipse(target_order, orders, platform_set, w=ellipse_w)
            considered_customers = searchRaidarEllipse_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, searchRaidarEllipse_C_T, now_t=now_t, titleinfo=search_type)
        elif search_type == 'EllipseMJ':
            searchRaidarEllipseMJ_C_T = SearchRaidar_ellipseMJ(target_order, orders, platform_set, delta=ellipse_w)
            considered_customers = searchRaidarEllipseMJ_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, searchRaidarEllipseMJ_C_T, now_t=now_t, titleinfo=search_type)
        elif search_type == 'XGBoost':
            enumerate_C_T = BundleConsideredCustomers(target_order, platform_set, riders, orders,
                                                      bundle_search_variant=unserved_bundle_order_break,
                                                      d_thres_option=True, speed=speed) #todo : 확인 할 것
            considered_customers = enumerate_C_T
        thres = 100
        #start_time_sec = datetime.now()

        end_test8 = time.time()
        t_counter('test8', end_test8 - start_test8)


        start_time_sec = time.time()
        if search_type == 'XGBoost':
            #input('XGBoost')
            size3bundle, label3data = XGBoost_Bundle_Construct(target_order, considered_customers, 3, p2, XGBmodel3, now_t = now_t, speed = speed , bundle_permutation_option = bundle_permutation_option, thres = thres, thres_label=thres_label, label_check=check_label, feasible_return= False)
            #size3bundle = []
            #label3data = []
            size2bundle, label2data = XGBoost_Bundle_Construct(target_order, considered_customers, 2, p2, XGBmodel2, now_t = now_t, speed = speed , bundle_permutation_option = bundle_permutation_option, thres = thres, thres_label=thres_label, label_check=check_label, feasible_return= False)
            #size2bundle = XGBoost_Bundle_Construct(target_order, considered_customers, 2, p2, XGBmodel2, now_t = now_t, speed = speed , bundle_permutation_option = bundle_permutation_option, thres = thres)
            #this_loop_label = numpy.concatenate((label3data, label2data))
            check_label = numpy.concatenate((check_label, label3data, label2data))
            #print('확인용2', tr1, tr2)
        else:
            size3bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 3, p2, speed=speed, bundle_permutation_option = bundle_permutation_option, thres= thres, now_t = now_t, print_option= False, feasible_return= False)
            size2bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 2, p2, speed=speed,bundle_permutation_option=bundle_permutation_option , thres= thres, now_t = now_t, print_option= False, feasible_return= False)
            label3data = []
            label2data = []
        #end_time_sec = datetime.now()
        end_time_sec = time.time()
        duration = end_time_sec - start_time_sec
        ##duration = duration.seconds + duration.microseconds/1000000
        t_counter(search_type, duration)
        ##번들 내용물 확인 필요
        b_count = 2
        for b_infos in [size2bundle, size3bundle]:
            if len(b_infos) == 0:
                continue
            for info in b_infos:
                #print(b_infos)
                #print(info)
                #input('확인 T:'+str(now_t))
                coord = []
                for ct_name in info[4]:
                    coord += [orders[ct_name].store_loc]
                    coord += [orders[ct_name].location]
                check_list('b'+str(b_count),info + coord)
            b_count += 1
        max_index = 100
        tem_infos = []
        try:
            size3bundle.sort(key=operator.itemgetter(6))
            for info in size3bundle[:max_index]:
                tem_infos.append(info)
            #input('번들 추가됨1: 추가 길이::'+ str(len(tem_infos)))
        except:
            pass
        try:
            size2bundle.sort(key=operator.itemgetter(6))
            for info in size2bundle[:max_index]:
                tem_infos.append(info)
            #input('번들 추가됨2: 추가 길이::'+ str(len(tem_infos)))
        except:
            pass
        Feasible_bundle_set += tem_infos
        end = time.time()
        if len(size3bundle) +len(size2bundle) > 0:
            print('T now',int(now_t),'bundle_size', len(size3bundle),len(size2bundle))
            #print('번들3',size3bundle)
            #print('번들2',size2bundle)
            #input('확인22')
        print('고객 당 계산 시간 {} : B2::{} B3::{}'.format(end - start, len(size2bundle),len(size3bundle)))
    #check_label
    #check_label.count()
    score_cal_s = time.time()
    start_test1 = time.time()
    try:
        unique, counts = numpy.unique(check_label, return_counts=True)
        #unique, counts = numpy.unique(this_loop_label, return_counts=True)
        print('label 정보',str(dict(zip(unique, counts))))
        label_count_dict = dict(zip(unique, counts))
    except:
        unique = [0,1,2,3,4]
        counts = [0,0,0,0,0]
    print('unique',unique)
    print('counts',counts)
    res_label_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for key in label_count_dict:
        res_label_count[int(key)] = label_count_dict[key]
    """
    for label_val in range(5):
        if label_val in unique:
            try:
                res_label_count[label_val] = counts[label_val]
            except:
                pass
        else:
            pass
    """
    end_test1 = time.time()
    t_counter('test1', end_test1 - start_test1)

    start_test2 = time.time()
    phi_br = []
    phi_b = []
    if pr_para == True:
        #print(unique, counts)
        print(res_label_count[:25])
        #input('label 정보 확인')
        print('대상 고객들', check_considered_customers[:5])
        #input('확인:' + str(now_t))
        print('T {} 번들 수 확인2::{}'.format(now_t, len(Feasible_bundle_set)))
        for rider_name in riders:
            rider = riders[rider_name]
            phi_r = Calculate_Phi(rider, orders, Feasible_bundle_set)
            phi_br.append(phi_r)
            #input('라이더 {} / 확인 phi_r {}'.format(rider.name, phi_r))
        print('T {} Phi 계산::{}'.format(now_t, len(phi_br)))
        for bundle_index in range(len(phi_br[0])):
            tem = 1
            for rider_index in range(len(phi_br)):
                value = phi_br[rider_index][bundle_index]
                if value > 0:
                    tem = tem * value
            phi_b.append(tem)

    end_test2 = time.time()
    t_counter('test2', end_test2 - start_test2)
    start_test3 = time.time()
    #input('phi_b {}'.format(phi_b))
    #2 d-matrix계산
    d_matrix = numpy.zeros((len(Feasible_bundle_set),len(Feasible_bundle_set)))
    for index1 in range(len(Feasible_bundle_set)):
        b1 = Feasible_bundle_set[index1][4]
        for index2 in range(len(Feasible_bundle_set)):
            b2 = Feasible_bundle_set[index2][4]
            if index1 > index2 and len(list(set(b1 + b2))) < len(b1) + len(b2) : #겹치는 것이 존재
                    d_matrix[index1, index2] = 1
                    d_matrix[index2, index1] = 1
            else:
                d_matrix[index1, index2] = 0
                d_matrix[index2, index1] = 0
    end_test3 = time.time()
    t_counter('test3', end_test3 - start_test3)
    start_test4 = time.time()
    #3 s_b계산 #파레토 계산으로 대체
    print('Phi 계산1')
    count1 = 0
    #print('FS',Feasible_bundle_set)
    if search_type == 'XGBoost' or search_type == 'enumerate':
        s_matrix = []
        for info in Feasible_bundle_set:
            s_matrix.append(info[5]/len(info[4]))
    else:
        s_matrix = numpy.zeros((len(Feasible_bundle_set), 1))
        for info in Feasible_bundle_set:
            try:
                """
                val = 0
                if type(info[6]) == float or  type(info[6]) == int:
                    val = info[6]
                elif type(info[6]) == list:
                    val = min(info[6])
                else:
                    print(type(info[6]))
                    input('Feasible_bundle_set INFO ERROR')
                """
                try:
                    val = min(info[6])
                except:
                    val = info[6]
                if val >= min_time_buffer:
                    count2 = 0
                    for info1 in Feasible_bundle_set:
                        if info == info1:
                            continue
                        if phi_b[count1] > phi_b[count2] and info[7] > info1[7]:
                            s_matrix[count1]+= 1
                        count2 += 1
            except:
                print(info)
                input('info 에러')
            count1 += 1
    #print('s_matrix',s_matrix)
    print('Phi 계산2')
    end_test4 = time.time()
    t_counter('test4', end_test4 - start_test4)

    start_test5 = time.time()

    #4 D 계산
    D = []
    print('D확인1')
    #info1 : [route, round(max(ftds), 2), round(sum(ftds) / len(ftds), 2), round(min(ftds), 2), order_names, round(route_time, 2)]
    info1_index = 0
    print('D확인1 {}'.format(len(Feasible_bundle_set)))
    for info1 in Feasible_bundle_set:
        info2_index = 0
        for info2 in Feasible_bundle_set[info1_index + 1:]: #todo: 0907 heavy computation part. -> 중복 연산 제거
            if info1 != info2 and len(info1[4]) + len(info2[4]) != len(list(set(info1[4] + info2[4]))): #todo: 0907 heavy computation part. -> info1 != info2 이 필요한가?
                #print('b1 {}/ b2 {}/ b1+b2 {}'.format(info1[4],info2[4], list(set(info1[4] + info2[4]))))
                D.append([info1_index, info2_index]) #todo: 0907 heavy computation part.  -> index() 함수 대체
                #D.append([Feasible_bundle_set.index(info1), Feasible_bundle_set.index(info2)])
            info2_index += 1
        if info1_index % 100 == 0:
            print('현재 진행 ::{}'.format(info1_index))
        info1_index += 1

    end_test5 = time.time()
    t_counter('test5', end_test5 - start_test5)

    start_test6 = time.time()

    lt_vector = []
    print('D확인2 {}'.format(D[:10]))


    for info in Feasible_bundle_set:
        bundle_names = info[4]
        over_t = []
        for name in bundle_names:
            customer = orders[name]
            over_t.append(now_t - customer.time_info[0])
        lt_vector.append(sum(over_t)/len(over_t))
    score_cal_e = time.time()
    score_cal_duration = score_cal_e - score_cal_s
    t_counter('score_cal', score_cal_duration)
    end_test6 = time.time()
    t_counter('test6', end_test6 - start_test6)
    return Feasible_bundle_set, phi_b, d_matrix, s_matrix, D, lt_vector, len(check_considered_customers), res_label_count


def Bundle_Ready_Processs3(now_t, platform_set, orders, riders, p2,interval, bundle_permutation_option = False, speed = 1, min_pr = 0.05,
                      unserved_bundle_order_break = True, considered_customer_type = 'all', search_type = 'enumerate', print_fig = False,
                          ellipse_w = 1.5, heuristic_theta = 100,heuristic_r1 = 10, min_time_buffer = 5, XGBmodel3 = None, XGBmodel2 = None, thres_label = 1,
                           MAX_RouteConstruct = 100000, MAX_MIP = 5000):
    # 번들이 필요한 라이더에게 번들 계산.
    if considered_customer_type == 'new':
        considered_customers_names = NewCustomer(orders, now_t, interval=interval)
    else:
        considered_customers_names, interval_orders = CountUnpickedOrders(orders, now_t, interval=interval,return_type='name')
    print('탐색 대상 고객들 수 :: {}'.format(len(considered_customers_names)))

    active_rider_names, d_infos = CountActiveRider(riders, interval, min_pr=min_pr, t_now=now_t, option='w', point_return= True)
    ###todo : 0913 Time save IDEA 01 START
    max_d_list = []
    tem_count = 0
    search_index = 5
    for p1 in d_infos:
        tem = []
        for order_name in orders:
            order = orders[order_name]
            if order.time_info[1] == None and order.cancel == False:
                tem.append(distance(p1, order.location)/speed)
        tem.sort()
        try:
            max_d = tem[search_index]
        except:
            max_d = 10
        max_d_list += [[active_rider_names[tem_count], p1, max_d]]
    print('max_d_list정보',max_d_list)
    ###todo : 0913 Time save IDEA 01 END
    print('돌아오는 시기에 주문 선택 예쌍 라이더 {}'.format(active_rider_names))
    #input('확인'+ str(now_t))
    Feasible_bundle_set = []
    BundleCheck = []
    for index in platform_set.platform:
        task = platform_set.platform[index]
        if len(task.customers) > 1:
            BundleCheck.append([index, len(task.customers)])
            #print('Index;{};Cts;{};picked;{};'.format(index, task.customers, task.picked))
    if len(BundleCheck) > 0:
        input('T {};;번들 존재{}'.format(now_t, BundleCheck))
    check_considered_customers = []
    check_label = numpy.array([])
    B2_masters = []
    B3_masters = []
    for customer_name in considered_customers_names:
        start = time.time()
        target_order = orders[customer_name]
        in_max_d = False
        for d_info in max_d_list:
            dist3 = distance(target_order.store_loc, d_info[1])/ speed
            if dist3 <= d_info[2]:
                in_max_d = True
                target_order.near_rider.append(d_info[0])
                break
        if in_max_d == False:
            continue
        check_considered_customers.append(target_order.name)
        if search_type == 'enumerate':
            #input('확인')
            enumerate_C_T = BundleConsideredCustomers(target_order, platform_set, riders, orders,
                                                      bundle_search_variant=unserved_bundle_order_break,
                                                      d_thres_option=True, speed=speed)
            considered_customers = enumerate_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, enumerate_C_T, now_t=now_t, titleinfo=search_type)
        elif search_type == 'heuristic':
            searchRaidar_heuristic_C_T = SearchRaidar_heuristic(target_order, orders, platform_set, theta=heuristic_theta,
                                                      r1=heuristic_r1, now_t=now_t)
            considered_customers = searchRaidar_heuristic_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, searchRaidar_heuristic_C_T, now_t=now_t, titleinfo=search_type)
        elif search_type == 'ellipse':
            searchRaidarEllipse_C_T = SearchRaidar_ellipse(target_order, orders, platform_set, w=ellipse_w)
            considered_customers = searchRaidarEllipse_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, searchRaidarEllipse_C_T, now_t=now_t, titleinfo=search_type)
        elif search_type == 'EllipseMJ':
            searchRaidarEllipseMJ_C_T = SearchRaidar_ellipseMJ(target_order, orders, platform_set, delta=ellipse_w)
            considered_customers = searchRaidarEllipseMJ_C_T
            if print_fig == True:
                PrintSearchCandidate(target_order, searchRaidarEllipseMJ_C_T, now_t=now_t, titleinfo=search_type)
        elif search_type == 'XGBoost':
            enumerate_C_T = BundleConsideredCustomers(target_order, platform_set, riders, orders,
                                                      bundle_search_variant=unserved_bundle_order_break,
                                                      d_thres_option=True, speed=speed) #todo : 확인 할 것
            considered_customers = enumerate_C_T
        thres = 100
        #start_time_sec = datetime.now()
        start_time_sec = time.time()
        print('XGBoost 시작2')
        if search_type == 'XGBoost':
            print('XGBoost 3')
            start_time_sec3 = time.time()
            B3_masters += XGBoost_Bundle_Construct2(target_order, orders, 3, XGBmodel3)
            end_time_sec3 = time.time()
            duration3 = end_time_sec3 - start_time_sec3
            print('B3계산시간', duration3)
            start_time_sec2 = time.time()
            B2_masters += XGBoost_Bundle_Construct2(target_order, orders, 2, XGBmodel2)
            end_time_sec2 = time.time()
            duration2 = end_time_sec2 - start_time_sec2
            print('B2계산시간', duration2)
        else:
            size3bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 3, p2, speed=speed, bundle_permutation_option = bundle_permutation_option, thres= thres, now_t = now_t, print_option= False)
            size2bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 2, p2, speed=speed,bundle_permutation_option=bundle_permutation_option , thres= thres, now_t = now_t, print_option= False)
            label3data = []
            label2data = []
        end_time_sec = time.time()
        duration = end_time_sec - start_time_sec
        print('개선된 계산 시간', duration)
        #duration = duration.seconds + duration.microseconds/1000000
        t_counter(search_type, duration)
    try:
        B2_masters.sort(key=operator.itemgetter(1))
    except:
        input('B2 번들거리X')
    try:
        B3_masters.sort(key=operator.itemgetter(1))
    except:
        input('B3 번들거리X')
    print('작업 이전 RC::', len(B2_masters) + len(B3_masters))
    B2_masters = B2_masters[:int(MAX_RouteConstruct/2)]
    B3_masters = B3_masters[:int(MAX_RouteConstruct/2)]
    print('작업 이후 RC::', len(B2_masters) + len(B3_masters))
    Master_datas = [[],[]]
    index3 = 0
    check_label = []
    print(B2_masters)
    print(B3_masters)
    for infos in [B2_masters, B3_masters]:
        for info in infos:
            if len(list(info)) > 0:
                print(info[0])
                #input('확인222')
                order_subset = []
                for name in list(info[0]):
                    order_subset.append(orders[name])
                tem = BundleConsist2(order_subset, orders, p2, speed=speed,
                                     bundle_permutation_option=bundle_permutation_option, uncertainty=False,
                                     platform_exp_error=1, feasible_return=True, now_t=now_t, max_dist=15)  # max_dist= 15
                check_label.append(info[1])
                if len(tem) > 0:
                    #Master_datas[index3].append(tem)
                    Master_datas[index3] += tem
        #print(Master_datas[index3][:1])
        if len(Master_datas[index3]) > 0:
            print(len(Master_datas[index3]))
        else:
            print("00")
        print(Master_datas[index3])
        input('sort전 len 확인')
        if len(Master_datas[index3]) > 2:
            Master_datas[index3].sort(key=operator.itemgetter(8))
        index3 += 1
    #Feasible_bundle_set = Master_datas[0] + Master_datas[1]
    #Rider 별로 줄 세우기
    rv_select_ratio = 0.8
    thres_MIP = int(((MAX_MIP/len(active_rider_names)) * rv_select_ratio * 1/2))
    #random_thres_MIP = int(((MAX_MIP/len(active_rider_names)) * (1 - rv_select_ratio) * 1/2))
    random_thres_MIP = 1 - rv_select_ratio
    end_MIP = int(MAX_MIP/len(active_rider_names))
    rev_Feasible_bundle_set = []
    #rev_Feasible_bundle_set = Master_datas[0] + Master_datas[1]
    added_customer_sets = []
    for rider_name in active_rider_names:
        add_count = 0
        b_size = 2
        for infos in Master_datas:
            for info in infos:
                #print(orders[info[4][b_size]],info[4])
                if len(info) > 1 and list(info[4]) not in added_customer_sets:
                #if rider_name in orders[info[4][b_size]].near_rider and info[4] not in added_customer_sets:
                    if add_count <= thres_MIP :
                        rev_Feasible_bundle_set.append(info)
                        added_customer_sets.append(list(info[4]))
                        add_count += 1
                    else:
                        if random.random() < random_thres_MIP:
                            rev_Feasible_bundle_set.append(info)
                            added_customer_sets.append(info[4])
                            add_count += 1
                    if add_count > end_MIP:
                        break
            b_size += 1
    Feasible_bundle_set = rev_Feasible_bundle_set
    print('MIP의사결정변수', len(Feasible_bundle_set))
    #near rider reset
    for ct_name in orders:
        orders[ct_name].near_rider = []
    try:
        unique, counts = numpy.unique(check_label, return_counts=True)
        #unique, counts = numpy.unique(this_loop_label, return_counts=True)
        print('label 정보',str(dict(zip(unique, counts))))
        label_count_dict = dict(zip(unique, counts))
    except:
        unique = list(range(35))
        counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    print('unique',unique)
    print('counts',counts)
    res_label_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for key in label_count_dict:
        res_label_count[int(key)] = label_count_dict[key]
    #print(unique, counts)
    print(res_label_count[:25])
    input('label 정보 확인')
    print('대상 고객들', check_considered_customers[:5])
    #input('확인:' + str(now_t))
    print('T {} 번들 수 확인2::{}'.format(now_t, len(Feasible_bundle_set)))
    phi_br = []
    for rider_name in riders:
        rider = riders[rider_name]
        phi_r = Calculate_Phi(rider, orders, Feasible_bundle_set)
        phi_br.append(phi_r)
        #input('라이더 {} / 확인 phi_r {}'.format(rider.name, phi_r))
    print('T {} Phi 계산::{}'.format(now_t, len(phi_br)))
    phi_b = []
    for bundle_index in range(len(phi_br[0])):
        tem = 1
        for rider_index in range(len(phi_br)):
            value = phi_br[rider_index][bundle_index]
            if value > 0:
                tem = tem * value
        phi_b.append(tem)
    #input('phi_b {}'.format(phi_b))
    #2 d-matrix계산
    d_matrix = numpy.zeros((len(Feasible_bundle_set),len(Feasible_bundle_set)))
    for index1 in range(len(Feasible_bundle_set)):
        b1 = Feasible_bundle_set[index1][4]
        for index2 in range(len(Feasible_bundle_set)):
            b2 = Feasible_bundle_set[index2][4]
            if index1 > index2 and len(list(set(b1 + b2))) < len(b1) + len(b2) : #겹치는 것이 존재
                    d_matrix[index1, index2] = 1
                    d_matrix[index2, index1] = 1
            else:
                d_matrix[index1, index2] = 0
                d_matrix[index2, index1] = 0
    #3 s_b계산 #파레토 계산으로 대체
    print('Phi 계산1')
    count1 = 0
    #print('FS',Feasible_bundle_set)
    if search_type == 'XGBoost' or search_type == 'enumerate':
        s_matrix = []
        for info in Feasible_bundle_set:
            s_matrix.append(info[5]/len(info[4]))
    else:
        s_matrix = numpy.zeros((len(Feasible_bundle_set), 1))
        for info in Feasible_bundle_set:
            try:
                """
                val = 0
                if type(info[6]) == float or  type(info[6]) == int:
                    val = info[6]
                elif type(info[6]) == list:
                    val = min(info[6])
                else:
                    print(type(info[6]))
                    input('Feasible_bundle_set INFO ERROR')
                """
                try:
                    val = min(info[6])
                except:
                    val = info[6]
                if val >= min_time_buffer:
                    count2 = 0
                    for info1 in Feasible_bundle_set:
                        if info == info1:
                            continue
                        if phi_b[count1] > phi_b[count2] and info[7] > info1[7]:
                            s_matrix[count1]+= 1
                        count2 += 1
            except:
                print(info)
                input('info 에러')
            count1 += 1
    #print('s_matrix',s_matrix)
    print('Phi 계산2')
    #4 D 계산
    D = []
    print('D확인1')
    #info1 : [route, round(max(ftds), 2), round(sum(ftds) / len(ftds), 2), round(min(ftds), 2), order_names, round(route_time, 2)]
    info1_index = 0
    print('D확인1 {}'.format(len(Feasible_bundle_set)))
    for info1 in Feasible_bundle_set:
        info2_index = 0
        for info2 in Feasible_bundle_set[info1_index + 1:]: #todo: 0907 heavy computation part. -> 중복 연산 제거
            if info1 != info2 and len(info1[4]) + len(info2[4]) != len(list(set(info1[4] + info2[4]))): #todo: 0907 heavy computation part. -> info1 != info2 이 필요한가?
                #print('b1 {}/ b2 {}/ b1+b2 {}'.format(info1[4],info2[4], list(set(info1[4] + info2[4]))))
                D.append([info1_index, info2_index]) #todo: 0907 heavy computation part.  -> index() 함수 대체
                #D.append([Feasible_bundle_set.index(info1), Feasible_bundle_set.index(info2)])
            info2_index += 1
        if info1_index % 100 == 0:
            print('현재 진행 ::{}'.format(info1_index))
        info1_index += 1
    lt_vector = []
    print('D확인2 {}'.format(D[:10]))
    for info in Feasible_bundle_set:
        bundle_names = info[4]
        over_t = []
        for name in bundle_names:
            customer = orders[name]
            over_t.append(now_t - customer.time_info[0])
        lt_vector.append(sum(over_t)/len(over_t))
    input('ITE 확인')
    return Feasible_bundle_set, phi_b, d_matrix, s_matrix, D, lt_vector, len(check_considered_customers), res_label_count



def Break_the_bundle(platform, orders, org_bundle_num, rev_bundle_num):
    if sum(rev_bundle_num) < sum(org_bundle_num):
        break_info = [org_bundle_num[0] - rev_bundle_num[0],
                      org_bundle_num[1] - rev_bundle_num[1]]  # [B2 해체 수, B3 해체 수]
        # 번들의 해체가 필요
        platform.platform = BreakBundle(break_info, platform, orders)


def Rider_Bundle_plt(rider):
    count = 0
    for bundle_info in rider.bundles_infos:
        save_name = 'R{};C{}'.format(rider.name, count)
        #nodes = []
        #for info in bundle_info:
        #    nodes.append(info[2])
        print('Title:{}save_name/Route:{}'.format(save_name,bundle_info))
        Bundle_plot(bundle_info, save_name)
        count += 1
        #input('확인2')
def Bundle_plot(bundle_points, save_name='nan'):
    ax = plt.axes()
    # arrow
    #ax.title(save_name)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    ax.set_title(save_name)
    for index in range(1, len(bundle_points)):
        bf = bundle_points[index - 1][2]
        af = bundle_points[index][2]
        ax.arrow(bf[0], bf[1], af[0]-bf[0], af[1]-bf[1], head_width=0.5, head_length=0.5, fc='k', ec='k')
    for index in range(0, len(bundle_points)):
        node = bundle_points[index][2]
        if bundle_points[index][1] == 1:
            ax.plot(node[0], node[1], 'r-o', lw=1)
        else:
            ax.plot(node[0], node[1], 'b-*', lw=1)
    plt.savefig('test' + save_name + '.png', dpi=100)
    plt.close()

