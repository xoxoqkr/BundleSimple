# -*- coding: utf-8 -*-
import random

from A1_BasicFunc import distance
from A3_two_sided import ConstructFeasibleBundle_TwoSided, BundleConsideredCustomers, SearchRaidar_ellipse, SearchRaidar_ellipseMJ, SearchRaidar_heuristic
import operator
import time
import itertools

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
            tem_bundle = ConstructFeasibleBundle_TwoSided(target_order, considered_customers, bundle_size, p2, speed=speed, bundle_permutation_option = bundle_permutation_option, thres= thres, now_t = now_t)
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
    return Feasible_bundle_set

def DummyBundleCalculator(bundle_infos, customers, size = 2):
    #INPUT : 주문들, 구성된 번들
    #OUTPUT : 더미 고객들
    customer_combination_set = itertools.combinations(list(customers.keys()),size)
    used_set = []
    for bundle_info in bundle_infos:
        tem = bundle_info[4]
        tem.sort()
        used_set.append(tem)
        if tem not in customer_combination_set:
            index = customer_combination_set.index(tem)
            del customer_combination_set[index]
    return customer_combination_set

def BundleFeaturesCalculator(bundle_infos, customers, now_t):
    #INPUT : 번들 고객
    #OUTPUT : 번들, 번들 Features
    datas = []
    for bundle_info in bundle_infos:
        #Feature 기록
        for customer_name in bundle_info[4]:
            #todo: Feature 선정
            pass
    return datas

def OrderGenerator(dir):
    #dir을 받아서, object 생성
    #INPUT : dir
    #OUTPUT : object list
    basket = []
    f = open(dir, 'r')
    lines = f.readlines()
    count = 0
    for line in lines[1:]:
        info = line.split(';')
        info = info.split(',')
        index = int(info[0])
        x = info[2].split(',')[0][1:]
        y = info[2].split(',')[0][:-1]
        basket.append([[count, index, x,y, int(info[1]), int(info[3])]])
        count += 1
    f.close()
    return basket

def OrderGen(store_dir, customer_dir, store_size = 100, customer_size = 1000, order_size = 1000):
    #store dir, customer_dir을 받아서,order 생성
    #INPUT : dir1, dir2, size
    #OUTPUT : order list, stores, customers
    orders = []
    stores = random.sample(OrderGenerator(store_dir), store_size)
    customers = random.sample(OrderGenerator(customer_dir), customer_size)
    for count in range(order_size):
        store = random.choice(stores)
        customer = random.choice(customers)
        order = [store[0],store[2],store[3], customer[0], customer[2], customer[3]]
        orders.append(order)
    return orders, stores, customers



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