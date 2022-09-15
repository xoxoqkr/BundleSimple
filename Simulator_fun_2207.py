# -*- coding: utf-8 -*-
import random
import numpy as np
import copy
from A1_BasicFunc import distance
import operator
import time
import itertools
from A1_Class import Customer
from A3_two_sided import ConstructFeasibleBundle_TwoSided, BundleConsideredCustomers, SearchRaidar_ellipse, SearchRaidar_ellipseMJ, SearchRaidar_heuristic


def BundleFilter(Route):
    #INPUT: 라이더 경로
    # Route example : [[-1, -1, [25, 25], 2], [1, 0, [18, 25], 10], [1, 1, [45, 0], 27], [7, 0, [23, 13], 35], [7, 1, [25, 45], 50]]
    #OUTPUT : 번들 뭉치 C_{R}
    # Bundle example : [ [1, 0, [18, 25], 10],  [7, 0, [23, 13], 35], [1, 1, [45, 0], 27], [7, 1, [25, 45], 50]]
    bundles = []
    bundles_cts = []
    count = 0
    for info in Route:
        if info[0] > 1 and info[1] == 0: #시작점
            s = count
            e = count
            for info2 in Route[s+1:]:
                e += 1
                if info[0] == info2[0] and info[1] == 1:
                    break
            if e-s > 2: #bundle임
                bundle_cts = []
                in_route_cts = []
                for info3 in Route[s:e+1]:
                    in_route_cts.append(info3[0])
                for ct_name in in_route_cts: #자동으로 sort됨
                    ct_num = in_route_cts.count(ct_name)
                    if ct_num == 1:
                        print('반만 걸침;', ct_name)
                        pass
                    elif ct_num == 2:
                        if ct_name not in bundle_cts:
                            bundle_cts.append(ct_name)
                    else:
                        print('문제 발생;', ct_name, '::',ct_num)
            if bundle_cts not in bundles_cts:
                saved_info = [bundle_cts, Route[s:e+1]]
                bundles.append(saved_info)
                bundles_cts.append(bundle_cts)
        count += 1
    #bundles = [[1],[[1, 0, [18, 25], 10], [1, 1, [45, 0], 27]]]
    return bundles


def BundleProcess(env, customers,dummy_platform, heuristic_theta, heuristic_r1,ellipse_w,p2, bundle_permutation_option, bundle_size = [2], interval= 10, speed = 2,Data = [],DummyB2_data=[], DummyB3_data=[]):
    for t in range(100):
        now_t = env.now
        used_target = []
        Feasible_bundle_set, DummyB2, DummyB3 = IdealBundleCalculator(now_t, customers, used_target, dummy_platform, [],
                                                    heuristic_theta, heuristic_r1,ellipse_w,p2, bundle_permutation_option,
                                                    speed=speed, search_type='enumerate', bundle_size=bundle_size)
        Data +=Feasible_bundle_set
        DummyB2_data += DummyB2
        print('더미2',DummyB2_data)
        DummyB3_data += DummyB3
        print('T: {} 가능해 번들 수:{}'.format(int(env.now),len(Feasible_bundle_set)))
        try:
            test = random.sample(Feasible_bundle_set, 5)
            print(test)
        except:
            pass
        #input('확인')
        yield env.timeout(interval)


def IdealBundleCalculator(now_t, customers, used_target, dummy_platform , riders, heuristic_theta,heuristic_r1,ellipse_w,
                          p2,bundle_permutation_option, speed = 1, search_type = 'enumerate', bundle_size = [2]):
    #INPUT : 고객 정보
    #OUTPUT : Feasible 번들
    Feasible_bundle_set = []
    start = time.time()
    for customer_name in customers:
        if customer_name in used_target:
            continue
        target_order = customers[customer_name]
        #print(target_order)
        #input("확인")
        if search_type == 'enumerate':
            #input('확인')
            enumerate_C_T = BundleConsideredCustomers(target_order, dummy_platform, riders, customers,d_thres_option=True, speed=speed)
            considered_customers = enumerate_C_T
        elif search_type == 'heuristic':
            searchRaidar_heuristic_C_T = SearchRaidar_heuristic(target_order, customers, dummy_platform, theta=heuristic_theta,
                                                      r1=heuristic_r1, now_t=now_t)
            considered_customers = searchRaidar_heuristic_C_T
        elif search_type == 'ellipse':
            searchRaidarEllipse_C_T = SearchRaidar_ellipse(target_order, customers, dummy_platform, w=ellipse_w)
            considered_customers = searchRaidarEllipse_C_T
        else:
            searchRaidarEllipseMJ_C_T = SearchRaidar_ellipseMJ(target_order, customers, dummy_platform, delta=ellipse_w)
            considered_customers = searchRaidarEllipseMJ_C_T
        thres = 100
        max_index = 100
        for b_size in bundle_size:
            #print(b_size)
            #input("확인")
            tem_bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, b_size, p2, speed=speed, bundle_permutation_option = bundle_permutation_option, thres= thres, now_t = now_t)
            tem_infos = []
            try:
                tem_bundle.sort(key=operator.itemgetter(6))
                for info in tem_bundle[:max_index]:
                    tem_infos.append(info)
            except:
                pass
            Feasible_bundle_set += tem_infos
            end = time.time()
            print('고객 {} 번들 계산 시간 {} : B{}::{}'.format(customer_name, end - start, b_size, len(tem_infos)))
        #더미 변수
    #겹치는 것 빼기
    res = []
    in_res_names = []
    for bundle_info in Feasible_bundle_set:
        if bundle_info[0] not in in_res_names:
            in_res_names.append(bundle_info[0])
            res.append(bundle_info)
    DummyB2 = DummyBundleCalculator(res, customers, size = 2)
    print('더미',DummyB2[:5])
    DummyB3 = DummyBundleCalculator(res, customers, size = 3)
    return res, DummyB2, DummyB3

def DummyBundleCalculator(bundle_infos, customers, size = 2):
    #INPUT : 주문들, 구성된 번들
    #OUTPUT : 더미 고객들
    customer_combination_set = []
    for names in list(itertools.combinations(list(customers.keys()),size)):
        customer_combination_set.append(list(names))
    used_set = []
    for bundle_info in bundle_infos:
        tem = bundle_info[4]
        tem.sort()
        used_set.append(tem)
        #print(customer_combination_set)
        if len(tem) == size and tem not in customer_combination_set:
            index = customer_combination_set.index(tem)
            del customer_combination_set[index]
    return customer_combination_set

def BundleFeaturesCalculator(customer_data, names_set, label = 0):
    #INPUT : 번들 고객
    #OUTPUT : 번들, 번들 Features
    datas = []
    used_names = []
    for names in names_set:
        if list(names) in used_names:
            continue
        used_names.append(list(names))
        if len(names) > 3:
            print(names)
            input('중지')
        #Dist_Feature
        distOD = []
        for name in names:
            #print('name', name)
            #print(customer_data[name])
            #input('확인')
            distOD.append(distance(customer_data[int(name)][2:4],customer_data[int(name)][5:7]))
        eachother = itertools.combinations(names,2)
        distS = []
        distC = []
        for info in eachother:
            distS.append(distance(customer_data[int(info[0])][2:4],customer_data[int(info[1])][2:4]))
            distC.append(distance(customer_data[int(info[0])][5:7], customer_data[int(info[1])][5:7]))
        #Times_Feature
        gen_t = []
        for name in names:
            gen_t.append(customer_data[int(name)][1])
        # service_times
        #ser_t = []
        #for name in names:
        #    ser_t.append(customer_data[int(name)][12])
        #service_times
        ser_t = []
        for name in names:
            ser_t.append(customer_data[int(name)][12])
        #print('확인',distOD,distS,distS,gen_t)
        tem = list(copy.deepcopy(names))
        tem += sorted(distOD)
        tem += sorted(distS)
        tem += sorted(distC)
        tem += sorted(gen_t)
        #tem += sorted(cook_t)
        tem += sorted(ser_t)
        tem += [label]
        datas.append(tem)
    return datas

def BundleFeaturesCalculator2(customer_data, names_set, label = 0, add_info = None, print_option = False):
    #INPUT : 번들 고객
    #OUTPUT : 번들, 번들 Features
    datas = []
    used_names = []
    count = 0
    for names in names_set:
        if list(names) in used_names:
            continue
        used_names.append(list(names))
        if len(names) > 3:
            print(names)
            input('중지')
        #Dist_Feature
        distOD = []
        for name in names:
            #print('name', name)
            #print(customer_data[name])
            #input('확인')
            distOD.append(distance(customer_data[int(name)].location,customer_data[int(name)].store_loc))
        eachother = itertools.combinations(names,2)
        distS = []
        distC = []
        for info in eachother:
            distS.append(distance(customer_data[int(info[0])].location,customer_data[int(info[1])].location))
            distC.append(distance(customer_data[int(info[0])].store_loc, customer_data[int(info[1])].store_loc))
        #Times_Feature
        gen_t = []
        for name in names:
            gen_t.append(customer_data[int(name)].name)
        # cook_times
        cook_t = []
        for name in names:
            cook_t.append(customer_data[int(name)].time_info[6])
        #service_times
        ser_t = []
        for name in names:
            ser_t.append(customer_data[int(name)].time_info[7])
        #print('확인',distOD,distS,distS,gen_t)
        tem = list(copy.deepcopy(names))
        tem += sorted(distOD)
        tem += sorted(distS)
        tem += sorted(distC)
        tem += sorted(gen_t)
        #tem += sorted(cook_t)
        tem += sorted(ser_t)
        dummy = []
        if type(add_info) == int:
            for _ in range(add_info):
                dummy.append(0)
        if add_info != None:
            if type(add_info) == int:
                tem += dummy
            else:
                tem += add_info[count]
        tem += [label]
        datas.append(tem)
        count += 1
        if print_option == True:
            print(tem)
    return datas





def ObjectGenerator(dir, coor_random= False, x_s = 0, x_e = 50, y_s = 0, y_e = 50):
    #dir을 받아서, object 생성
    #INPUT : dir
    #OUTPUT : object list
    basket = []
    f = open(dir, 'r')
    lines = f.readlines()
    count = 0
    for line in lines[1:]:
        info = line.split(';')
        #print(info)
        #info = info.split(',')
        index = int(info[0])
        if coor_random == False:
            x = float(info[2].split(',')[0][1:])
            y = float(info[2].split(',')[1][:-1])
        else:
            x = random.choice(range(x_s,x_e)) + round(random.random(),2)
            y = random.choice(range(y_s,y_e)) + round(random.random(),2)
        basket.append([count, index, x,y, int(info[1]), int(info[3])])
        #print(basket[-1])
        count += 1
    f.close()
    return basket


def ObjectGenerator2(dir, coor_random= False):
    #dir을 받아서, object 생성
    #INPUT : dir
    #OUTPUT : object list
    basket = []
    f = open(dir, 'r')
    lines = f.readlines()
    count = 0
    for line in lines[1:]:
        info = line.split(';')
        #print(info)
        #info = info.split(',')
        index = int(info[0])
        if coor_random == False:
            x = float(info[2].split(',')[0][1:])
            y = float(info[2].split(',')[1][:-1])
        else:
            x = random.choice(range(0,50)) + round(random.random(),2)
            y = random.choice(range(0,50)) + round(random.random(),2)
        basket.append([count, index, x,y, int(info[1]), int(info[3])])
        #print(basket[-1])
        count += 1
    f.close()
    return basket


def OrderGen(store_dir, customer_dir, store_size = 100, customer_size = 1000, order_size = 1000, coor_random= False, input_data = None, s_x_s = 0, s_x_e = 50, s_y_s = 0, s_y_e = 50):
    #store dir, customer_dir을 받아서,order 생성
    #INPUT : dir1, dir2, size
    #OUTPUT : order list, stores, customers
    orders = []
    stores = random.sample(ObjectGenerator(store_dir, coor_random = coor_random, x_s = s_x_s, x_e = s_x_e, y_s = s_y_s, y_e = s_y_e), store_size)
    customers = random.sample(ObjectGenerator(customer_dir, coor_random = coor_random), customer_size)
    t = 0
    for count in range(order_size):
        store = random.choice(stores)
        customer = random.choice(customers)
        t += random.randrange(1,5)
        try:
            order = [count,t , float(customer[2]), float(customer[3]),int(store[0]),float(store[2]),float(store[3]), 10, 7,0,1,7]
            order.append(random.randrange(4,9))  # service_times
            order.append(random.randrange(2, 4))  #interval
            orders.append(order)
        except:
            print(store)
            print(customer)
    return orders, stores, customers


def OrderGenerator(env, order_datas, Order_dict):
    index = len(Order_dict)
    for order in order_datas:
        yield env.timeout(order[6])
        Order_dict[index] = Customer(env,index, order[4:6], store = order[0], store_loc = order[1:3])
    print('Customer Generatio End')





def SavedDataCalculator(Routes, Customers, thres1 = 15):
    #1 Select bundle from Routes
    bundles = []
    for Route in Routes:
        bundle4route = BundleFilter(Route)
        bundles += bundle4route
    #2 Add state to bundle
    ct_data_dict = {}
    for bundle in bundles:
        tem_info = [[],[]]
        for ct_name1 in bundle[0]: #Attach additional information
            customer1 = Customers[ct_name1]
            for ct_name2 in Customers:
                customer2 = Customers[ct_name2]
                if ct_name1 != ct_name2 and distance(customer1.store_loc,customer2.store_loc) <= thres1 \
                        and customer2.time_info[0] <= customer1.time_info[1] \
                        and (customer1.time_info[1] <= customer2.time_info[1] or customer2.time_info[1] == None):
                    tem_info[0].append(ct_name2) #주문 선택 당시의 플랫폼 state
            #who_serve.append([self.name, round(env.now,2),current_coord,self.onhand])
            tem_info[1].append(customer1.who_serve[1:4]) #주문 선택 당시의 플랫폼 state
    #3 Calculate Reward for the bundle
    reward = []
    for bundle in bundles:
        # Attach reward
        p2p_dist = 0
        tw = 0
        for ct_name1 in bundle[0]: #Attach additional information
            customer1 = Customers[ct_name1]
            p2p_dist += distance(customer1.store_loc,customer1.location)
            tw += customer1.time_info[0] + customer1.time_info[5] - customer1.time_info[4]
        reward.append(p2p_dist, tw)
    return bundles