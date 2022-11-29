# -*- coding: utf-8 -*-

#from scipy.stats import poisson
from datetime import datetime
from A1_BasicFunc import RouteTime, distance, FLT_Calculate, counter, t_counter
import operator
import itertools
import re_A1_class as A1_Class
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from Bundle_selection_problem import RebaseProblem
import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans


def LamdaMuCalculate(orders, riders, now_t, interval = 5, return_type = 'class'):
    unpicked_orders, lamda2 = CountUnpickedOrders(orders, now_t, interval=interval, return_type=return_type)  # lamda1
    lamda1 = len(unpicked_orders)
    idle_riders, mu2 = CountIdleRiders(riders, now_t, interval=interval, return_type=return_type)
    mu1 = len(idle_riders)
    return lamda1, lamda2, mu1, mu2

def NewCustomer(cusotmers, now_t, interval = 5):
    new_customer_names = []
    for customer_name in cusotmers:
        customer = cusotmers[customer_name]
        if now_t - interval <= customer.time_info[0] and customer.time_info[1] == None and customer.cancel == False:
            new_customer_names.append(customer.name)
    return new_customer_names


def CustomerValueForRiderCalculator(rider, customer):
    """
    rider가 customer에 대해 가지는 가치 계산
    :param rider:
    :param customer:
    """
    value = 10
    return value

def RiderChoiceCustomer(rider, customers):
    """
    Rider pick the highest score orders
    rider의 시각에서 customers 중 가장 높은 가치를 가지는 customer 계산
    :param rider: class rider
    :param customers: customer list [customer, customer,...,]
    :return: highest score customer class
    """
    customer_values = []
    for customer in customers:
        if customer.time_info[1] == None:
            value = CustomerValueForRiderCalculator(rider, customer)
            customer_values.append([value, customer])
    customer_values.sort(key = operator.itemgetter(0), reverse = True)
    if len(customer_values) > 0:
        return customer_values[0][1]
    else:
        return None


def CalculateRho(lamda1, lamda2, mu1, mu2, add_lamda = 0, add_mu = 0):
    """
    Calculate rho
    :param lamda1: current lamda
    :param lamda2: expected lamda of the near future time slot
    :param mu1: current mu
    :param mu2: expected mu of the near future time slot
    :param add_lamda: additional lamda
    :param add_mu: additional mu
    :return: rho
    """
    if mu1 + mu2 + add_mu > 0:
        rho = (lamda1 + lamda2 + add_lamda) / (mu1 + mu2 + add_mu)
    else:
        rho = 2
    return round(rho, 4)


def RequiredBundleNumber(lamda1, lamda2, mu1, mu2, thres = 1):
    """
    Cacluate required b2 and b3 number
    condition : rho <= thres
    :param lamda1: current un-selected order
    :param lamda2: future generated order
    :param mu1: current rider
    :param mu2: future rider
    :param thres: rho thres: system over-load
    :return: b2, b3
    """
    b2 = 0
    b3 = 0
    for index in range(lamda1+lamda2):
        b2 += 1
        rho = CalculateRho(lamda1, lamda2, mu1, mu2, add_lamda = -b2)
        #rho = (lamda1 + lamda2 - b2)/(mu1 + mu2)
        if rho <= thres:
            return b2, b3
    for index in range(lamda1+lamda2):
        b2 -= 1
        b3 += 1
        rho = CalculateRho(lamda1, lamda2, mu1, mu2, add_lamda=-(b2+b3))
        #rho = (lamda1 + lamda2 - b2 - b3)/(mu1 + mu2)
        if rho <= thres:
            return b2, b3
    return b2, b3


def RequiredBreakBundleNum(platform_set, lamda2, mu1, mu2, thres = 1):
    """
    Caclculate availiable break-down bundle number
    :param platform_set: orders set : [order,...]
    :param lamda2: expected lamda of the near future time slot
    :param mu1: current mu
    :param mu2: expected mu of the near future time slot
    :param thres: system level.
    :return:
    """
    org_b2_num = 0
    org_b3_num = 0
    b2_num = 0
    b3_num = 0
    customer_num = 0
    for order_name in platform_set.platform:
        order = platform_set.platform[order_name]
        if order.type == 'bundle':
            if len(order.customers) == 2:
                b2_num += 1
                org_b2_num += 1
            else:
                b3_num += 1
                org_b3_num += 1
        else:
            customer_num += 1
    end_para = False
    for count in range(org_b3_num): #break b3 first
        if b3_num > 0:
            b3_num -= 1
            customer_num += 3
        else:
            pass
        p = CalculateRho(b2_num + b3_num + customer_num, lamda2, mu1, mu2)
        if p >= thres:
            end_para = True
            break
    if end_para == False: #if p < thres, than break b2
        for count in range(org_b2_num):
            if b2_num > 0:
                b2_num -= 1
                customer_num += 2
            else:
                pass
            p = CalculateRho(b2_num + b3_num + customer_num, lamda2, mu1, mu2)
            if p >= thres:
                break
    return [org_b2_num,org_b3_num],[b2_num, b3_num]


def BreakBundle(break_info, platform_set, customer_set):
    """
    Break bundle by break_info
    And return the revised platform_set
    :param break_info: bundle breaking info [b2 decrcase num, b2 decrcase num]
    :param platform_set: orders set : [order,...]
    :param customer_set: customer set : [customer class,...]
    :return: breaked platform set
    """
    b2 = []
    b3 = []
    single_orders = []
    breaked_customer_names = []
    for order_name in platform_set.platform:
        order = platform_set.platform[order_name]
        if order.type == 'bundle':
            if len(order.customers) == 2:
                b2.append(order)
            else:
                b3.append(order)
        else:
            single_orders.append(order)
    b2.sort(key=operator.attrgetter('average_ftd'), reverse=True)
    b3.sort(key=operator.attrgetter('average_ftd'), reverse=True)
    for break_b2 in range(min(break_info[0],len(b2))):
        #breaked_customer_names.append(b2[0].customers)
        breaked_customer_names += b2[0].customers
        del b2[0]
    for break_b3 in range(min(break_info[1],len(b3))):
        #breaked_customer_names.append(b3[0].customers)
        breaked_customer_names += b3[0].customers
        del b3[0]
    breaked_customers = []
    order_nums = []
    for order_name in platform_set.platform:
        order = platform_set.platform[order_name]
        order_nums += order.customers
    order_num = max(order_nums) + 1
    for customer_name in breaked_customer_names:
        route = [[customer_name, 0, customer_set[customer_name].store_loc, 0],[customer_name, 1, customer_set[customer_name].location, 0 ]]
        order = A1_Class.Order(order_num,[customer_name], route, 'single', fee = customer_set[customer_name].fee)
        breaked_customers.append(order)
    res = {}
    for order in single_orders + b2 + b3 + breaked_customers:
        res[order.index] = order
    return res


def BundleConsist(orders, customers, p2, time_thres = 0, speed = 1,M = 10000, bundle_permutation_option = False, uncertainty = False, platform_exp_error =  1, feasible_return = False, now_t = 0):
    """
    Construct bundle consists of orders
    :param orders: customer order in the route. type: customer class
    :param customers: customer dict  {[KY]customer name: [Value]class customer,...}
    :param p2: allowable FLT increase
    :param M: big number for distinguish order name and store name
    :param speed: rider speed
    :return: feasible route
    """
    counter('bundle_consist')
    order_names = [] #가게 이름?
    for order in orders:
        order_names.append(order.name)
    store_names = []
    for name in order_names:
        store_names.append(name + M)
    candi = order_names + store_names
    if bundle_permutation_option == False:
        subset = itertools.permutations(candi, len(candi))
    else:
        store_subset = itertools.permutations(store_names, len(store_names))
        store_subset = list(store_subset)
        order_subset = itertools.permutations(order_names, len(order_names))
        order_subset = list(order_subset)
        test = []
        test_names = itertools.permutations(order_names, 2)
        for names in test_names:
            dist = distance(customers[names[0]].location[0],customers[names[0]].location[1], customers[names[1]].location[0],customers[names[1]].location[1])
            if dist > 15:
                #print('거리에 의한 종료')
                return []
        subset = []
        for store in store_subset:
            for order in order_subset:
                tem = store + order
                subset.append(tem)
        pass
    #print('번들 처리 시작. 대상 subset{}'.format(subset))
    #print('번들 고려시 탐색 수 {}'.format(len(list(subset))))
    feasible_subset = []
    for route in subset:
        #print('고객이름',order_names,'가게이름',store_names,'경로',route)
        sequence_feasiblity = True #모든 가게가 고객 보다 앞에 오는 경우.
        feasible_routes = []
        for order_name in order_names: # order_name + M : store name ;
            if route.index(order_name + M) < route.index(order_name):
                pass
            else:
                sequence_feasiblity = False
                break
        if sequence_feasiblity == True:
            #input('feasilbe 통과1')
            ftd_feasiblity, ftds = FLT_Calculate(orders, customers, route, p2, [],M = M ,speed = speed, uncertainty =uncertainty, exp_error=platform_exp_error, now_t = now_t)
            #customer_in_order, customers, route, p2, except_names, M = 1000, speed = 1, now_t = 0
            if ftd_feasiblity == True:
                #input('feasilbe 통과2')
                route_time, unsync_t = RouteTime(orders, route, speed=speed, M=M, uncertainty = uncertainty, error = platform_exp_error, sync_output_para = True, now_t = now_t, bywho='Platform')
                #feasible_routes.append([route, round(max(ftds), 2), round(sum(ftds) / len(ftds), 2), round(min(ftds), 2), order_names,round(route_time, 2)])
                feasible_routes.append([route, unsync_t[0], round(sum(ftds) / len(ftds), 2), unsync_t[1], order_names, round(route_time, 2)])
                #print('시간 정보 번들 경로 시간 {} : 가능한 짧은 시간 {}'.format(route_time, time_thres))
                #if route_time < time_thres :
                #    feasible_routes.append([route, round(max(ftds),2), round(sum(ftds)/len(ftds),2), round(min(ftds),2), order_names, round(route_time,2)])
                #    input('번들 생성 절약 시간 {}'.format(time_thres - route_time))
                #[경로, 최대FTD, 평균FTD, 최소FTD]
        if len(feasible_routes) > 0:
            feasible_routes.sort(key = operator.itemgetter(2)) #todo: ftd 평균이 짧은 순으로 정렬
            feasible_subset.append(feasible_routes[0])
    if len(feasible_subset) > 0:
        feasible_subset.sort(key = operator.itemgetter(2))
        #GraphDraw(feasible_subset[0], customers)
        if feasible_return == True:
            return feasible_subset
        else:
            return feasible_subset[0]
    else:
        return []


def BundleConsist2(orders, customers, p2, time_thres = 0, speed = 1,M = 10000, bundle_permutation_option = False,
                   uncertainty = False, platform_exp_error =  1, feasible_return = False, now_t = 0, min_time_buffer = 10, max_dist = 15, sort_index = 8, fix_start = True):
    """
    Construct bundle consists of orders
    :param orders: customer order in the route. type: customer class
    :param customers: customer dict  {[KY]customer name: [Value]class customer,...}
    :param p2: allowable FLT increase
    :param M: big number for distinguish order name and store name
    :param speed: rider speed
    :return: feasible route
    """
    counter('bundle_consist2')
    #print('BundleConsist2 시작')
    #start_time_sec = datetime.now()
    start_time_sec = time.time()
    order_names = [] #가게 이름?
    if fix_start == True:
        fixed_start_point = orders[-1].name + M
    else:
        fixed_start_point = None
    bin_temperature = []
    for order in orders:
        order_names.append(order.name)
        bin_temperature.append(order.temperature)
    if 'C' in bin_temperature and 'W' in bin_temperature: # todo : 221101실험을 현실적으로 변경. - > 음식의 온도에 따라 번들이 될 수 없는 규칙을 설정(Cold와 Warm이 동시에 실릴 수 없음)
        #print('온도 차이로 불발:: 온도::', bin_temperature)
        #input('확인1101')
        return []
    store_names = []
    for name in order_names:
        store_names.append(name + M)
    if fix_start == True: #제일 끝에 target 고객이 위치
        store_names = store_names[:-1]
    candi = order_names + store_names
    #print(candi)
    #input('candi check')
    if bundle_permutation_option == False:
        subset = itertools.permutations(candi, len(candi))
    else:
        store_subset = itertools.permutations(store_names, len(store_names))
        store_subset = list(store_subset)
        order_subset = itertools.permutations(order_names, len(order_names))
        order_subset = list(order_subset)
        test = []
        test_names = itertools.permutations(order_names, 2)
        for names in test_names:
            dist = distance(customers[names[0]].location[0],customers[names[0]].location[1], customers[names[1]].location[0],customers[names[1]].location[1])
            if dist > max_dist:
                #print('거리에 의한 종료')
                return []
        subset = []
        for store in store_subset:
            for order in order_subset:
                tem = store + order
                subset.append(tem)
        pass
    #print(subset)
    #input('경로 확인')
    feasible_subset = []
    for route_org in subset:
        route = list(route_org)
        if fix_start == True:
            route = [fixed_start_point] + list(route_org)
        sequence_feasiblity = True #모든 가게가 고객 보다 앞에 오는 경우.
        for order_name in order_names: # order_name + M : store name ;
            if route.index(order_name + M) < route.index(order_name):
                pass
            else:
                sequence_feasiblity = False
                break
        P2P_dist = 0
        for order_name in order_names:
            P2P_dist += distance(customers[order_name].store_loc[0],customers[order_name].store_loc[1], customers[order_name].location[0],customers[order_name].location[1])/speed
        if sequence_feasiblity == True:
            ftd_feasiblity, ftds = FLT_Calculate(orders, customers, route, p2, [],M = M ,speed = speed, uncertainty =uncertainty, exp_error=platform_exp_error, now_t = now_t)
            if ftd_feasiblity == True:
                route_time, unsync_t, time_buffer = RouteTime(orders, route, speed=speed, M=M, uncertainty = uncertainty, error = platform_exp_error,
                                                              sync_output_para = True, now_t = now_t, bywho='Platform', time_buffer_para= True)
                if min(time_buffer) >= min_time_buffer:
                    try:
                        origin = customers[route[0] - M].store_loc
                        destination = customers[route[-1]].location
                    except:
                        print(route)
                        #input('잘못된 경로')
                        return []
                    line_dist = round(route_time, 2) - (distance(origin[0],origin[1], destination[0],destination[1])/speed)
                    tem = [route, unsync_t[0], round(sum(ftds) / len(ftds), 2), unsync_t[1], order_names, round(route_time, 2),min(time_buffer), round(P2P_dist - route_time, 2), line_dist, round(P2P_dist,4), distance(origin[0],origin[1], destination[0],destination[1])/speed]
                    if line_dist <= 0.1 :
                        print_route = []
                        try:
                            for value in route:
                                if value >= M:
                                    print_route.append(customers[value - M].store_loc)
                                else:
                                    print_route.append(customers[value].location)
                            #print(origin, destination, round(route_time, 2) , distance(origin[0],origin[1], destination[0],destination[1])/speed, line_dist)
                            #print(print_route)
                            #input('거리 에러')
                        except:
                            pass
                    feasible_subset.append(tem)
                else:
                    #print('조리 시간 초과1:',min(time_buffer),'::', min_time_buffer)
                    pass
            else:

                #print('음식 조리 시간 초과2::',ftd_feasiblity,ftds)
                pass
    #end_time_sec = datetime.now()
    end_time_sec = time.time()
    duration = end_time_sec - start_time_sec
    #duration = duration.seconds + duration.microseconds / 1000000
    t_counter('old', duration)
    if len(feasible_subset) > 0:
        feasible_subset.sort(key = operator.itemgetter(sort_index))
        if feasible_return == True:
            #print('반환 번들 수',len(feasible_subset), feasible_subset[:5])
            return feasible_subset
        else:
            return [feasible_subset[0]]
    else:
        return []


def GraphDraw(infos, customers, M = 10000):
    # 그래프 그리기
    x = []
    y = []
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    store_label = []
    loc_label = []
    locs = []
    for node in infos[0]:
        if node > M:
            name = node - M
            x.append(customers[name].store_loc[0])
            y.append(customers[name].store_loc[1])
            x1.append(customers[name].store_loc[0])
            y1.append(customers[name].store_loc[1])
            store_label.append('S' + str(name))
            locs.append(customers[name].store_loc)
        else:
            x.append(customers[node].location[0])
            y.append(customers[node].location[1])
            x2.append(customers[node].location[0])
            y2.append(customers[node].location[1])
            loc_label.append('C' + str(node))
            locs.append(customers[name].location)
    # plt.plot(x, y, linestyle='solid', color='blue', marker = 6)
    x3 = np.array(x)
    y3 = np.array(y)
    plt.quiver(x3[:-1], y3[:-1], x3[1:] - x3[:-1], y3[1:] - y3[:-1], scale_units='xy', angles='xy', scale=1)
    plt.scatter(x1, y1, marker="X", color='g')
    plt.scatter(x2, y2, marker="o", color='r')
    plt.title("Bundle coordinate")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.show()
    # name = str(random.random)
    tm = time.localtime(time.time())
    # print("hour:", tm.tm_hour)
    # print("minute:", tm.tm_min)
    # print("second:", tm.tm_sec)
    plt.savefig('B{}Hr{}Min{}Sec{}.png'.format(len(infos[4]), tm.tm_hour, tm.tm_min, tm.tm_sec))
    plt.close()
    print('경로 {}'.format(locs))
    #input('번들 경로 {} 시간 {}'.format(infos[0],infos[5]))


def ConstructBundle(orders, s, n, p2, speed = 1, option = False, uncertainty = False, platform_exp_error = 1, now_t = 0):
    """
    Construct s-size bundle pool based on the customer in orders.
    And select n bundle from the pool
    Required condition : customer`s FLT <= p2
    :param orders: userved customers : [customer class, ...,]
    :param s: bundle size: 2 or 3
    :param n: needed bundle number
    :param p2: max FLT
    :param speed:rider speed
    :return: constructed bundle set
    """
    B = []
    for order_name in orders:
        order = orders[order_name]
        d = []
        dist_thres = order.p2
        for order2_name in orders:
            order2 = orders[order2_name]
            dist = distance(order.store_loc[0],order.store_loc[1], order2.store_loc[0],order2.store_loc[1])/speed
            if order2 != order and dist <= dist_thres:
                d.append(order2.name)
        M = itertools.permutations(d, s - 1)
        #print('번들 구성 고려 subset 수 {}'.format(len(list(M))))
        #M = list(M)
        b = []
        for m in M:
            q = list(m) + [order.name]
            subset_orders = []
            time_thres = 0 #3개의 경로를 연속으로 가는 것 보다는
            for name in q:
                subset_orders.append(orders[name])
                time_thres += orders[name].distance/speed
            tem_route_info = BundleConsist(subset_orders, orders, p2, speed = speed, bundle_permutation_option= option, time_thres= time_thres, uncertainty = uncertainty, platform_exp_error = platform_exp_error, now_t = now_t)
            if len(tem_route_info) > 0:
                b.append(tem_route_info)
        if len(b) > 0:
            b.sort(key = operator.itemgetter(2))
            B.append(b[0])
            #input('삽입되는 {}'.format(b[0]))
    #n개의 번들 선택
    B.sort(key = operator.itemgetter(5))
    selected_bundles = []
    selected_orders = []
    print('번들들 {}'.format(B))
    for bundle_info in B:
        # bundle_info = [[route,max(ftds),average(ftds), min(ftds), names],...,]
        unique = True
        for name in bundle_info[4]:
            if name in selected_orders:
                unique = False
                break
        if unique == True:
            selected_orders += bundle_info[4]
            selected_bundles.append(bundle_info)
            if len(selected_bundles) >= n:
                break
    if len(selected_bundles) > 0:
        #print("selected bundle#", len(selected_bundles))
        print("selected bundle#", selected_bundles)
        #input('멈춤7')
        pass
    #todo: 1)겹치는 고객을 가지는 번들 중 1개를 선택해야함. 2)어떤 번들이 더 좋은 번들인가?
    return selected_bundles


def CountUnpickedOrders(orders, now_t , interval = 10, return_type = 'class'):
    """
    return un-picked order
    :param orders: order list : [order class,...]
    :param now_t : now time
    :param interval : platform`s bundle construct interval # 플랫폼에서 번들을 생성하는 시간 간격.
    :param return_type: 'class'/'name'
    :return: unpicked_orders, lamda2(future generated order)
    """
    unpicked_orders = []
    interval_orders = []
    for order_name in orders:
        order = orders[order_name]
        if order.time_info[1] == None and order.cancel == False:
            if return_type == 'class':
                unpicked_orders.append(order)
            elif return_type == 'name':
                unpicked_orders.append(order.name)
            else:
                pass
        if now_t- interval <= order.time_info[0] < now_t:
            interval_orders.append(order.name)
    return unpicked_orders, len(interval_orders)


def CountIdleRiders(riders, now_t , interval = 10, return_type = 'class'):
    """
    return idle rider
    :param riders: rider list : [rider class,...]
    :param now_t : now time
    :param interval : platform`s bundle construct interval # 플랫폼에서 번들을 생성하는 시간 간격.
    :param return_type: 'class'/'name'
    :return: idle_riders, mu2(future generated rider)
    """
    idle_riders = []
    interval_riders = []
    for rider_name in riders:
        #Count current idle rider
        rider = riders[rider_name]
        if len(rider.resource.users) == 0:
            if return_type == 'class':
                idle_riders.append(rider)
            elif return_type == 'name':
                idle_riders.append(rider.name)
            else:
                pass
        #count rider occurred from (now_t - interval, now)
        if now_t- interval <= rider.start_time < now_t:
            interval_riders.append(rider.name)
    return idle_riders, len(interval_riders)


def GenSingleOrder(order_index, customer, platform_exp_error = 1):
    pool = np.random.normal(customer.cook_info[1][0], customer.cook_info[1][1] * platform_exp_error, 1000)
    customer.platform_exp_cook_time = random.choice(pool)
    route = [[customer.name, 0, customer.store_loc, 0], [customer.name, 1, customer.location, 0]]
    o = A1_Class.Order(order_index, [customer.name], route, 'single', fee=customer.fee, parameter_info=None)
    return o


def GenBundleOrder(order_index, bundie_info, customer_set, now_t, M = 10000, platform_exp_error = 1, add_fee = 0, bundle_type = 0):
    """

    @param order_index:
    @param bundie_info:
    @param customer_set:
    @param now_t:
    @param M:
    @param platform_exp_error:
    @param add_fee:
    @param bundle_type: 0 : single 1: Dynamic 2: Static
    @return:
    """
    route = []
    for node in bundie_info[0]:
        if node >= M:
            customer_name = node - M
            customer = customer_set[customer_name]
            route.append([customer_name, 0, customer.store_loc, 0])
        else:
            customer_name = node
            customer = customer_set[customer_name]
            route.append([customer_name, 1, customer.location, 0])
    fee = 0
    for customer_name in bundie_info[4]:
        fee += customer_set[customer_name].fee  # 주문의 금액 더하기.
        #customer_set[customer_name].in_bundle_time = now_t
        pool = np.random.normal(customer.cook_info[1][0], customer.cook_info[1][1] * platform_exp_error, 1000)
        customer_set[customer_name].platform_exp_cook_time = random.choice(pool)
        customer_set[customer_name].in_bundle_t = now_t
    fee += add_fee
    o = A1_Class.Order(order_index, bundie_info[4], route, 'bundle', fee=fee, parameter_info=bundie_info[7:10])
    o.gen_t = now_t
    o.old_info = bundie_info
    o.average_ftd = bundie_info[2]
    o.bundle_type = bundle_type
    return o


def ResultPrint(name, customers, speed = 1, riders = None):
    rider_income_var = None
    done_bundle = []
    if riders != None:
        riders_incomes = []
        for rider_name in riders:
            rider = riders[rider_name]
            riders_incomes.append(rider.income)
            done_bundle += rider.bundle_count
        rider_income_var = np.var(riders_incomes)
    try:
        ave_done_bundle = sum(done_bundle)/len(done_bundle)
    except:
        ave_done_bundle = 0
    served_customer = []
    TLT = []
    FLT = []
    MFLT = []
    OD_ratios = []
    b1 = [] #14
    b2 = [] #15
    b3 = [] #16
    b4 = [] #17
    b5 = [] #18
    p1 = [] #18 #19
    p2 = [] #19 #20
    p3 = [] # 20 #21
    p4 = [] #22
    r1 = [] #23
    r2 = [] #24
    r3 = [] #25
    r4 = [] #26
    r5 = [] #27
    count_b = 0 #28
    count_p = 0
    count_r = 0
    test1 = []
    service_times = []
    unselected_ct = 0 #조리가 시작 되었지만, 조리 되지 못하고 버려지는 음식
    food_wait3 = []
    rider_wait3 = []
    food_wait1 = [] #15분 미만 음식
    food_wait2 = [] #15분 이상 음식
    cancel_ct = 0
    for customer_name in customers:
        if customer_name == 0 :
            continue
        customer = customers[customer_name]
        if customer.time_info[3] != None:
            lt = customer.time_info[3] - customer.time_info[0]
            try:
                flt = round(customer.time_info[3] - customer.time_info[2],5)
            except:
                flt  = 10
                input('FLT = 0 : {}'.format(customer.time_info))
            mflt = round(distance(customer.store_loc[0],customer.store_loc[1], customer.location[0],customer.location[1])/speed,5)
            TLT.append(lt)
            FLT.append(flt)
            MFLT.append(mflt)
            try:
                OD_ratio = flt/mflt
            except:
                OD_ratio = 1
            service_times.append(customer.time_info[7])
            if OD_ratio > 1.0:
                OD_ratios.append(OD_ratio - 1.0)
            if flt < mflt:
                print('고객 확인; 고객 {}; 실제FLT{}; 최소FLT{}; 선택 이력{}; 시간정보 {}'.format(customer.name, flt, mflt, customer.who_picked, customer.time_info))
            if customer.in_bundle_t > 0 and customer.dynamic_type != None:
                b1.append(customer.in_bundle_t - customer.time_info[0])
                b2.append(customer.time_info[1] - customer.in_bundle_t)
                b3.append(customer.time_info[2] - customer.time_info[1])
                b4.append(customer.time_info[3] - customer.time_info[2])
                b5.append(customer.time_info[3] - customer.time_info[2] - mflt)
                #b5.append(customer.time_info[4] - customer.time_info[2] - mflt) #가게 도착 부터 고객 도착까지 걸린 시간 -
                count_b += 1
            else:
                if customer.time_info[3] - customer.time_info[2] - mflt > 0.001:
                    r1.append(max(0,customer.rider_bundle_t - customer.time_info[0]))
                    r2.append(customer.time_info[1] - customer.rider_bundle_t)
                    r3.append(customer.time_info[2] - customer.time_info[1])
                    r4.append(customer.time_info[3] - customer.time_info[2])
                    r5.append(customer.time_info[3] - customer.time_info[2] - mflt)
                    count_r += 1
                else:
                    p1.append(customer.time_info[1] - customer.time_info[0])
                    p2.append(customer.time_info[2] - customer.time_info[1])
                    p3.append(customer.time_info[3] - customer.time_info[2])
                    p4.append(customer.time_info[3] - customer.time_info[2] - mflt)
                    count_p += 1
            test1.append(customer.time_info[3] - (customer.cook_start_time + customer.actual_cook_time))
            if customer.food_wait3 != None:
                food_wait3.append(customer.food_wait3)
                if customer.dp_cook_time < 15:
                    food_wait1.append(customer.food_wait3)
                else:
                    food_wait2.append(customer.food_wait3)
            if customer.rider_wait3 != None:
                rider_wait3.append(customer.rider_wait3)
            print('확인용',customer.name, customer.food_wait3, customer.rider_wait3)
        else:
            if customer.cancel == True:
                cancel_ct += 1
            if customer.cook_start_time > 0 and customer.time_info[1] == None:
                unselected_ct += 1
    customer_lead_time_var = np.var(TLT)
    try:
        served_ratio = round(len(TLT)/len(customers),2)
        av_TLT = round(sum(TLT)/len(TLT),2)
        av_FLT = round(sum(FLT)/len(FLT),2)
        av_MFLT = av_FLT - round(sum(MFLT)/len(MFLT),2)
        if len(OD_ratios) > 0:
            OD_ratio_value = np.std(OD_ratios)
            ave_OD_ratio_value = sum(OD_ratios)/len(OD_ratios)
        else:
            OD_ratio_value = None
            ave_OD_ratio_value = None
        if len(b1) > 0:
            ave_b1 = np.average(b1)
            ave_b2 = np.average(b2)
            ave_b3 = np.average(b3)
            ave_b4 = np.average(b4)
            ave_b5 = np.average(b5)
        else:
            ave_b1 = 0
            ave_b2 = 0
            ave_b3 = 0
            ave_b4 = 0
            ave_b5 = 0
        if len(p1) > 0:
            ave_p1 = np.average(p1)
            ave_p2 = np.average(p2)
            ave_p3 = np.average(p3)
            ave_p4 = np.average(p4)
        else:
            ave_p1 = 0
            ave_p2 = 0
            ave_p3 = 0
            ave_p4 = 0
        if len(r1) > 0:
            ave_r1 = np.average(r1)
            ave_r2 = np.average(r2)
            ave_r3 = np.average(r3)
            ave_r4 = np.average(r4)
            ave_r5 = np.average(r5)
        else:
            ave_r1 = 0
            ave_r2 = 0
            ave_r3 = 0
            ave_r4 = 0
            ave_r5 = 0
        ave_food_wait3 = None
        if len(food_wait3) > 0:
            ave_food_wait3 = sum(food_wait3) / len(food_wait3)
        ave_rider_wait3 = None
        if len(rider_wait3) > 0:
            ave_rider_wait3 = sum(rider_wait3) / len(rider_wait3)
        ave_test1 = None
        if len(test1) > 0:
            ave_test1 = sum(test1) / len(test1)
        ave_food_wait1 = None
        if len(food_wait1) > 0:
            ave_food_wait1 = sum(food_wait1)/len(food_wait1)
        ave_food_wait2 = None
        if len(food_wait2) > 0:
            ave_food_wait2 = sum(food_wait2)/len(food_wait2)
        ave_servie_time = round(sum(service_times)/len(service_times),4)
        print('시나리오 명 {} 전체 고객 {} 중 서비스 고객 {}/ 서비스율 {}/ 평균 LT :{}/ 평균 FLT : {}/직선거리 대비 증가분 : {}'.format(name, len(customers), len(TLT),served_ratio,av_TLT,
                                                                             av_FLT, av_MFLT))
        return [len(customers), len(TLT),served_ratio,av_TLT,av_FLT, av_MFLT, round(sum(MFLT)/len(MFLT),2), rider_income_var,customer_lead_time_var,len(OD_ratios),OD_ratio_value,ave_OD_ratio_value,len(done_bundle),ave_done_bundle,
                ave_b1,ave_b2,ave_b3,ave_b4,ave_b5,count_b,ave_p1,ave_p2,ave_p3,ave_p4,count_p,ave_r1,ave_r2,ave_r3,ave_r4,ave_r5, count_r,ave_servie_time, ave_test1 ,unselected_ct,ave_food_wait3,ave_rider_wait3,ave_food_wait1,ave_food_wait2,len(food_wait1),len(food_wait2),len(rider_wait3)]
    except:
        print('TLT 수:  {}'.format(len(TLT)))
        return None

def RebaseCustomer1(order_names, orders, r_inc = 0.2, max_r = 3, end_ite = 10, num_thres = 50):
    ite = 0
    r = 1
    re_points = []
    while r <= max_r and ite <= end_ite:
        N_set = []
        N_set_names = []
        index1 = 0
        for name in order_names:
            ct1 = orders[name]
            index2 = 0
            tem = [name]
            tem2 = [index1]
            for name2 in order_names:
                ct2 = orders[name2]
                if name != name2 and abs(ct1.store_loc[0] - ct2.store_loc[0]) < r and abs(ct1.store_loc[1] - ct2.store_loc[1]) < r :
                    ct1.neighbors.append(name2)
                    tem.append(name2)
                    tem2.append(index2)
                index2 += 1
            index1 += 1
            N_set.append(tem2)
            N_set_names.append(tem)
            ct1.neighbors.sort()
        re_points, obj_val = RebaseProblem(N_set, print_para=False)
        ite += 1
        r += r_inc
        if len(re_points) <= num_thres:
            break
    res = []
    if len(re_points) > 0:
        for index in re_points:
            res.append(N_set_names[index])
        print('문제 풀이 됨')
        return True, res
    else:
        res = order_names
        print('문제 풀이 X')
        return False, res

def RebaseCustomer2(order_names, orders, n_clusters_divider = 3):
    #order names의 1 번째 주문이 기준 주문
    ct_dict = {}
    ct_coord = []
    for ct_name in order_names:
        order = orders[ct_name]
        ct_coord.append(order.location)
        ct_dict[tuple(order.location)] = ct_name
    ct_coord = np.array(ct_coord)
    n_clusters = int(len(ct_coord)/n_clusters_divider)
    tem_res = []
    if n_clusters > 1:
        kmeans = KMeans(n_clusters = n_clusters, random_state=0).fit(ct_coord)
        count = 0
        for label in kmeans.labels_:
            #ct_dict[tuple(ct_coord[count])] = label
            tem_res.append([ct_dict[tuple(ct_coord[count])], label])
            count += 1
    else:
        for name in order_names:
            tem_res.append([name,0])
    labelled = {}
    if len(tem_res) > 0:
        for i in range(max(1,n_clusters)):
            labelled[i] = []
        for info in tem_res:
            labelled[info[1]].append(info[0])
        #print(labelled)
    return labelled