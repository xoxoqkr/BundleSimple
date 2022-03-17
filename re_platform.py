# -*- coding: utf-8 -*-
import time
import math
from A1_BasicFunc import PrintSearchCandidate
from A2_Func import CountUnpickedOrders, CalculateRho, RequiredBreakBundleNum, BreakBundle, GenBundleOrder,  LamdaMuCalculate, NewCustomer
from A3_two_sided import BundleConsideredCustomers, CountActiveRider,  ConstructFeasibleBundle_TwoSided, SearchRaidar_heuristic, SearchRaidar_ellipse, SearchRaidar_ellipseMJ
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
                      bundle_print_fig = False, bundle_infos = None,ellipse_w = 1.5, heuristic_theta = 100,heuristic_r1 = 10):
    yield env.timeout(5) #warm-up time
    while env.now <= end_t:
        if bundle_para == True:
            lamda1, lamda2, mu1, mu2 = LamdaMuCalculate(orders, riders, env.now, interval=interval, return_type='class')
            p = CalculateRho(lamda1, lamda2, mu1, mu2)
            if p > thres_p:
                #feasible_bundle_set, phi_b, d_matrix, s_b, D, lt_matrix = Bundle_Ready_Processs(env.now, platform, orders, riders, p2, interval,
                #                                                                                speed = riders[0].speed, bundle_permutation_option= True, search_type = search_type, print_fig = print_fig,
                #                                                                                ellipse_w = ellipse_w, heuristic_theta = heuristic_theta,heuristic_r1 = heuristic_r1)
                feasible_bundle_set, phi_b, d_matrix, s_b, D, lt_matrix = Bundle_Ready_Processs2(env.now, platform, orders, riders, p2, interval,
                                                                                                speed = riders[0].speed, bundle_permutation_option= True, search_type = search_type, print_fig = print_fig,
                                                                                                ellipse_w = ellipse_w, heuristic_theta = heuristic_theta,heuristic_r1 = heuristic_r1)


                print('phi_b {}:{} d_matrix {}:{} s_b {}:{}'.format(len(phi_b), numpy.average(phi_b),
                                                                    d_matrix.shape, numpy.average(d_matrix),len(s_b),numpy.average(s_b),))
                print('d_matrix : {}'.format(d_matrix))
                #문제 풀이
                #unique_bundle_indexs = Bundle_selection_problem3(phi_b, d_matrix, s_b, min_pr = 0.05)
                unique_bundle_indexs = Bundle_selection_problem4(phi_b, D, s_b, lt_matrix, min_pr = 1, obj_type= obj_type) #todo : 0317_수정본. min_pr을 무의미한 제약식으로 설정
                #input('결과 확인')
                unique_bundles = []
                for index in unique_bundle_indexs:
                    unique_bundles.append(feasible_bundle_set[index])
                print('문제 풀이 결과 {} '.format(unique_bundles[:10]))
                # 번들을 업로드
                task_index = max(list(platform.platform.keys())) + 1
                if len(unique_bundles) > 0:
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
                    plt.scatter(x1, y1, marker = 'o', color = 'k', label = 'store')
                    plt.scatter(x2, y2, marker='x', color='m', label='customer')
                    plt.legend()
                    plt.axis([0,50,0,50])
                    title = 'ST;{};T;{};;Selected Bundle Size;{}'.format(search_type,round(env.now,2), len(unique_bundles))
                    plt.title(title)
                    if bundle_print_fig == True:
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
            else:
                print('번들 삭제 수행')
                org_bundle_num, rev_bundle_num = RequiredBreakBundleNum(platform, lamda2, mu1, mu2, thres=thres_p)
                Break_the_bundle(platform, orders, org_bundle_num, rev_bundle_num)
        yield env.timeout(interval)
        if bundle_para == True and delete_para == True:
            #input('경로 지우기')
            delete_task_names = []
            for task_name in platform.platform:
                if len(platform.platform[task_name].customers) > 1 : # and platform.platform[task_name].picked == False:
                    delete_task_names.append(task_name)
            for task_name in delete_task_names:
                del platform.platform[task_name]
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
            input('dist {} dist2{} speed{} '.format(dist, distance(customer.store_loc, customer.location), rider.speed))
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
                          ellipse_w = 1.5, heuristic_theta = 100,heuristic_r1 = 10, min_time_buffer = 5):
    # 번들이 필요한 라이더에게 번들 계산.
    if considered_customer_type == 'new':
        considered_customers_names = NewCustomer(orders, now_t, interval=interval)
    else:
        considered_customers_names, interval_orders = CountUnpickedOrders(orders, now_t, interval=interval,return_type='name')
    print('탐색 대상 고객들 {}'.format(considered_customers_names))
    active_rider_names = CountActiveRider(riders, interval, min_pr=min_pr, t_now=now_t, option='w')
    print('돌아오는 시기에 주문 선택 예쌍 라이더 {}'.format(active_rider_names))
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
        thres = 100
        size3bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 3, p2, speed=speed, bundle_permutation_option = bundle_permutation_option, thres= thres, now_t = now_t)
        size2bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, 2, p2, speed=speed,bundle_permutation_option=bundle_permutation_option , thres= thres, now_t = now_t)
        max_index = 100
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
        end = time.time()
        print('고객 당 계산 시간 {} : B2::{} B3::{}'.format(end - start, len(size2bundle),len(size3bundle)))
    print('T {} 번들 수 {}'.format(now_t, len(Feasible_bundle_set)))
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
    #3 s_b계산 #파레토 계산으로 대체
    s_matrix = numpy.zeros((len(Feasible_bundle_set), 1))
    count1 = 0
    for info in Feasible_bundle_set:
        if min(info[6]) >= min_time_buffer:
            count2 = 0
            for info1 in Feasible_bundle_set:
                if info == info1:
                    continue
                if phi_b[count1] > phi_b[count2] and info[7] > info1[7]:
                    s_matrix[count1]+= 1
                count2 += 1
        count1 += 1
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
    return Feasible_bundle_set, phi_b, d_matrix, s_matrix, D, lt_vector


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

