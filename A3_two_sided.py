# -*- coding: utf-8 -*-

#from scipy.stats import poisson
import operator
import itertools
from A1_BasicFunc import distance, ActiveRiderCalculator
from A2_Func import BundleConsist
import math
import matplotlib.pyplot as plt

def CountActiveRider(riders, t, min_pr = 0, t_now = 0, option = 'w'):
    """
    현대 시점에서 t 시점내에 주문을 선택할 확률이 min_pr보다 더 높은 라이더를 계산
    @param riders: RIDER CLASS DICT
    @param t: t시점
    @param min_pr: 최소 확률(주문 선택확률이 min_pr 보다는 높아야 함.)
    @return: 만족하는 라이더의 이름. LIST
    """
    names = []
    for rider_name in riders:
        rider = riders[rider_name]
        if ActiveRiderCalculator(rider, t_now, option = option) == True and rider.select_pr(t) >= min_pr:
            names.append(rider_name)
    return names


def BundleConsideredCustomers(target_order, platform, riders, customers, speed = 1, bundle_search_variant = True, d_thres_option = True):
    not_served_ct_name_cls = {}
    not_served_ct_names = [] #번들 구성에 고려될 수 있는 고객들
    in_bundle_names = []
    for order_index in platform.platform:
        order = platform.platform[order_index]
        if len(order.customers) > 1 or order.picked == True:
            in_bundle_names += order.customers
    for customer_name in customers:
        customer = customers[customer_name]
        if customer.time_info[1] == None and customer.time_info[2] == None:
            if customer.type == 'single_order' and customer_name not in in_bundle_names:
                pass
            else:
                if bundle_search_variant == True:
                    pass
                else:
                    continue
            if d_thres_option == False:
                d_thres = 1000
            else:
                d_thres = customer.p2
            dist = distance(target_order.store_loc, customer.store_loc) / speed
            if target_order.name != customer.name and dist <= d_thres:
                not_served_ct_names.append(customer_name)
                not_served_ct_name_cls[customer_name] = customer
    current_in_bundle = []
    current_in_single = []
    for order_index in platform.platform:
        order = platform.platform[order_index]
        if order.type == 'bundle':
            current_in_bundle += platform.platform[order_index].customers
        else:
            current_in_single += platform.platform[order_index].customers
    rider_on_hand = []
    rider_finished = []
    for rider_name in riders:
        rider = riders[rider_name]
        rider_on_hand += rider.onhand
        rider_finished += rider.served
    res = {}
    for ct_name in not_served_ct_names:
        if ct_name in rider_on_hand + rider_finished:
            input('ERROR {} :: 고려 고객 {} 제외1 {} 제외 2 {}'.format(ct_name, not_served_ct_names, rider_on_hand, rider_finished))
        else:
            res[ct_name] = customers[ct_name]
    res[target_order.name] = target_order
    return res


def MIN_OD_pair(orders, q,s,):
    # 1 OD-pair 계산
    Q = itertools.permutations(q, s)  # 기존 OD pair의 가장 짧은 순서를 결정 하기 위함.
    OD_pair_dist = []
    for seq in Q:
        route_dist = 0
        tem_route = []
        for name in seq:
            tem_route += [orders[name].store_loc, orders[name].location]
        for index in range(1, len(tem_route)):
            before = tem_route[index - 1]
            after = tem_route[index]
            route_dist += distance(before, after)
        OD_pair_dist.append(route_dist)
    p2p_dist = 0
    for order_name in orders:
        p2p_dist += distance(orders[order_name].store_loc,orders[order_name].location)
    return min(OD_pair_dist), p2p_dist

#todo: 번들 생성 관련자
def ConstructFeasibleBundle_TwoSided(target_order, orders, s, p2, thres = 0.05, speed = 1, bundle_permutation_option = False, uncertainty = False,
                                     platform_exp_error = 1, print_option = True, sort_index = 5):
    """
    Construct s-size bundle pool based on the customer in orders.
    And select n bundle from the pool
    Required condition : customer`s FLT <= p2
    :param new_orders: new order genrated during t_bar
    :param orders: userved customers : [customer class, ...,]
    :param s: bundle size: 2 or 3
    :param p2: max FLT
    :param speed:rider speed
    :parm option:
    :parm uncertainty:
    :parm platform_exp_error:
    :parm bundle_search_variant: 번들 탐색시 대상이 되는 고객들 결정 (True : 기존에 번들의 고객들은 고려 X , False : 기존 번들의 고객도  고려)
    :return: constructed bundle set
    """
    d = []
    for customer_name in orders:
        if customer_name != target_order.name and orders[customer_name].time_info[1] == None and orders[customer_name].cancel == False:
            d.append(customer_name)
    if len(d) > s - 1:
        M = itertools.permutations(d, s - 1)
        b = []
        if print_option == True:
            print('대상 고객 {} 고려 고객들 {}'.format(target_order.name, d))
        for m in M:
            #print('대상 seq :: {}'.format(m))
            q = list(m) + [target_order.name]
            subset_orders = []
            time_thres = 0 #3개의 경로를 연속으로 가는 것 보다는
            for name in q:
                subset_orders.append(orders[name])
                time_thres += orders[name].distance/speed
            #input('확인 1 {} : 확인2 {}'.format(subset_orders, time_thres))
            tem_route_info = BundleConsist(subset_orders, orders, p2, speed = speed, bundle_permutation_option= bundle_permutation_option, time_thres= time_thres, uncertainty = uncertainty, platform_exp_error = platform_exp_error, feasible_return = True)
            #feasible_routes.append([route, round(max(ftds),2), round(sum(ftds)/len(ftds),2), round(min(ftds),2), order_names, round(route_time,2)])
            #print('계산{} :: {}'.format(q, tem_route_info))
            if len(tem_route_info) > 0:
                OD_pair_dist, p2p_dist = MIN_OD_pair(orders, q, s)
                for info in tem_route_info:
                    #info.append((OD_pair_dist - info[5]) / s)
                    #info.append((info[5] / s))
                    info.append((info[5]/p2p_dist)/s)
            b += tem_route_info
        #input('가능 번들 수 {} : 정보 d {} s {}'.format(len(b), d, s))
        comparable_b = []
        sort_index = 6 # 5: route time, 6: s_b
        if len(b) > 0:
            #b.sort(key=operator.itemgetter(6))  # s_b 순으로 정렬  #target order를 포함하는 모든 번들에 대해서 s_b를 계산.
            b.sort(key=operator.itemgetter(sort_index))
            b_star = b[0][sort_index]
            ave = []
            for ele in b:
                ave.append(ele[sort_index])
                if (ele[sort_index] - b_star)/b_star <= thres: #percent loss 가 thres 보다 작아야 함.
                    comparable_b.append(ele)
            print('평균 {}'.format(sum(ave)/len(ave)))
        return comparable_b
    else:
        return []

def SearchRaidar_heuristic(target, customers, platform, r1 = 10, theta = 90, now_t = 0, print_fig = False):
    """
    기준 고객을 중심으로 h,theta를 사용해 번들 연산이 가능한 고객을 만드는 과정
    :param target:
    :param customers:
    :param platform:
    :param r1:
    :param theta:
    :param now_t:
    :param print_fig:
    :return:
    """
    #Step 1: 가게 정리
    C_T = []
    for task_index in platform.platform:
        task = platform.platform[task_index]
        if len(task.customers) > 1:
            continue
        customer2 = customers[task.customers[0]]
        dist = distance(target.store_loc, customer2.store_loc)
        if dist < r1:
            C_T.append(customer2)
    #Step 2: SearchArea정의
    #2가지 조건으로 해당 주문의 Search_area 포함 여부를 계산
    #Step 3: C_T에 대해서 SearchArea 내에 있는지 여부 확인
    p0 = target.store_loc
    p1 = target.location
    thera_range = math.cos(math.pi * ((theta / 2) / 180))
    res_C_T = {}
    res_C_T[target.name] = customers[target.name]
    len_a = distance(p0, p1)
    for customer in C_T:
        p2 = customer.location
        len_b = distance(p0, p2)
        if len_b > len_a or len_a == 0 or len_b == 0: #3개의 점이 필요하기 때문
            continue
        len_c = distance(p1, p2)
        cos_c = (len_a ** 2 + len_b ** 2 - len_c ** 2) / (2 * len_a * len_b)
        if cos_c >= thera_range:
            res_C_T[customer.name] = customer
    return res_C_T


def SearchRaidar_ellipse(target, customers, platform, r1 = 10, w = 1):
    #Step 1: 가게 정리
    res_C_T = {}
    res_C_T[target.name] = customers[target.name]
    middle = [min(target.store_loc[0], target.location[0]) + abs(target.store_loc[0] - target.location[0]) ,
    min(target.store_loc[1], target.location[1]) + abs(target.store_loc[1] - target.location[1])]
    for task_index in platform.platform:
        task = platform.platform[task_index]
        if len(task.customers) > 1:
            continue
        customer2 = customers[task.customers[0]]
        dist0 = distance(target.store_loc, target.location)*w
        dist1 = distance(target.store_loc, customer2.store_loc)
        dist2 = distance(target.store_loc, customer2.location)
        dist3 = distance(target.location, customer2.location)
        dist4 = distance(middle, customer2.store_loc)
        dist5 = distance(middle, customer2.location)
        #if dist1 < r1 and (dist2 < dist0 and dist3 < dist0):
        if dist1 < r1 and (dist4 < dist0 and dist5 < dist0):
            res_C_T[customer2.name] = customers[customer2.name]
    return res_C_T


def SearchRaidar_ellipseMJ(target, customers, platform, delta = 5):
    #Step 1: 가게 정리
    res_C_T = {}
    res_C_T[target.name] = customers[target.name]
    for task_index in platform.platform:
        task = platform.platform[task_index]
        if len(task.customers) > 1:
            continue
        customer2 = customers[task.customers[0]]
        dist0 = distance(target.store_loc, target.location)
        dist1 = distance(target.store_loc, customer2.store_loc)
        dist2 = distance(target.location, customer2.store_loc)
        dist3 = distance(target.store_loc, customer2.location)
        dist4 = distance(target.location, customer2.location)
        if dist1 + dist2 <= dist0 + delta and dist3 + dist4 <= dist0 +delta:
            res_C_T[customer2.name] = customers[customer2.name]
    return res_C_T