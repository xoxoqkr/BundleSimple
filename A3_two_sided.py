# -*- coding: utf-8 -*-

#from scipy.stats import poisson
import time
import operator
import itertools
from numba import jit
import re_A1_class
from A1_BasicFunc import distance, ActiveRiderCalculator, t_counter, counter2, BundleExpValueCalculator
from A2_Func import BundleConsist, BundleConsist2, GenBundleOrder, GenSingleOrder
import math
import numpy as np
import pandas as pd
import random
import copy


def CountActiveRider(riders, t, min_pr = 0, t_now = 0, option = 'w', point_return = False, print_option = True):
    """
    현대 시점에서 t 시점내에 주문을 선택할 확률이 min_pr보다 더 높은 라이더를 계산
    @param riders: RIDER CLASS DICT
    @param t: t시점
    @param min_pr: 최소 확률(주문 선택확률이 min_pr 보다는 높아야 함.)
    @return: 만족하는 라이더의 이름. LIST
    """
    names = []
    dists = []
    times = []
    for rider_name in riders:
        rider = riders[rider_name]
        #print('rider name ::',str(rider.name))
        if ActiveRiderCalculator(rider, t_now, option = option, print_option = print_option) == True :#and rider.select_pr(t) >= min_pr:
            names.append(rider_name)
            if point_return == True:
                dists.append(rider.CurrentLoc(rider.next_search_time2, tag = 'tr3'))
                """
                if len(rider.resource.users) > 0:
                    dists.append(rider.CurrentLoc(rider.next_search_time))
                else:
                    dists.append(rider.last_departure_loc)
                """
                times.append(rider.next_search_time2)
                #print('라이더 {} 마지막 위치 {} 마지막 시간 {} 다음 탐색 시간 {}'.format(rider_name, dists[-1], times[-1], rider.next_search_time2))
            else:
                print('False')
        else:
            #print('False2')
            pass
    if point_return == True:
        return names, dists, times
    else:
        return names


def BundleConsideredCustomers(target_order, platform, riders, customers, speed = 1, bundle_search_variant = True, d_thres_option = True, max_d_infos = [],
                              revise_type = 'None',stopping = 40, cut_info = [1000,1000], except_target_order = False):
    #todo : 0907 정정
    not_served_ct_name_cls = {}
    not_served_ct_names = [] #번들 구성에 고려될 수 있는 고객들
    not_served_ct_names_infos = []
    in_bundle_names = []
    for order_index in platform.platform:
        order = platform.platform[order_index]
        if len(order.customers) > 1 or order.picked == True:
            in_bundle_names += order.customers
    un_served_num = 0
    for customer_name in customers:
        if customers[customer_name].time_info[1] == None:
            un_served_num += 1
    store_para = 0.4
    loc_para = 1.5
    dec_weight = un_served_num / (50 * 50)
    store_min = 3
    loc_min = 5
    candi_num = 0
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
            candi_num += 1
            if d_thres_option == False:
                d_thres = 100
            else:
                d_thres = customer.p2
            dist = distance(target_order.store_loc[0],target_order.store_loc[1], customer.store_loc[0],customer.store_loc[1]) / speed #OD거리
            dist2 = distance(target_order.location[0],target_order.location[1], customer.location[0],customer.location[1]) / speed ##DD 거리
            #if target_order.name != customer.name and dist <= d_thres :
            in_max_d = False
            for d_info in max_d_infos:
                dist3 = distance(target_order.store_loc[0],target_order.store_loc[1], d_info[1][0],d_info[1][1])/ speed
                if dist3 < d_info[2]:
                    in_max_d = True
                    break
            if len(max_d_infos) == 0:
                in_max_d = True
            dist3 = distance(target_order.store_loc[0], target_order.store_loc[1], customers[customer.name].location[0],customers[customer.name].location[1]) / speed  ##OO 거리
            if revise_type == 'cut_info':
                if target_order.name != customer.name and dist3 <= (cut_info[0] / speed) and dist2 <= (cut_info[1] / speed) and in_max_d == True:
                    not_served_ct_names.append(customer_name)
                    not_served_ct_name_cls[customer_name] = customer
                    not_served_ct_names_infos.append([customer_name, dist, dist2,dist3])
            else:
                if target_order.name != customer.name and dist <= d_thres*store_para and dist2 <= d_thres*loc_para and in_max_d == True:
                    not_served_ct_names.append(customer_name)
                    not_served_ct_name_cls[customer_name] = customer
                    not_served_ct_names_infos.append([customer_name, dist, dist2,dist3])
            """
            elif revise_type == 'cut_info2':
                if target_order.name != customer.name and (dist3 <= (cut_info[0] / speed) or dist2 <= (cut_info[1] / speed)) and in_max_d == True:
                    not_served_ct_names.append(customer_name)
                    not_served_ct_name_cls[customer_name] = customer
                    not_served_ct_names_infos.append([customer_name, dist, dist2,dist3])
            """
            """
            #if target_order.name != customer.name and dist <= store_min and dist2 <= loc_min and in_max_d == True:
            if target_order.name != customer.name and dist <= cut_info[0]/speed and dist2 <= cut_info[1]/speed and in_max_d == True:
                not_served_ct_names.append(customer_name)
                not_served_ct_name_cls[customer_name] = customer
                not_served_ct_names_infos.append([customer_name, dist, dist2])
            """
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
    if revise_type == 'stopping':
        rev_not_served_ct_names = []
        not_served_ct_names_infos.sort(key=operator.itemgetter(2))
        for info in not_served_ct_names_infos[:min(len(not_served_ct_names_infos),stopping)]:
            rev_not_served_ct_names.append(info[0])
        """
        #pareto count dominance
        pareto_score = []
        for info1 in not_served_ct_names_infos:
            score = 0
            for info2 in not_served_ct_names_infos:
                if info1[0] != info2[1] and info1[0] <= info2[0] and info1[1] <= info2[1]:
                    score += 1
            pareto_score.append([info1[0],score])
        pareto_score.sort(key=operator.itemgetter(1), reverse = True)
        for info in pareto_score[:min(len(pareto_score),stopping)]:
            rev_not_served_ct_names.append(info[0])
        """
    elif revise_type == 'cut_info': #현재 상태에서는 의미X
        rev_not_served_ct_names = []
        not_served_ct_names_infos.sort(key=operator.itemgetter(2))
        count = 0
        for info in not_served_ct_names_infos:
            if count < stopping:
                rev_not_served_ct_names.append(info[0])
                count += 1
                if info[3] > cut_info[0] / speed or info[2] > cut_info[1] / speed:
                    break
    elif revise_type == 'cut_info2': #현재 상태에서는 의미X
        rev_not_served_ct_names = []
        not_served_ct_names_infos.sort(key=operator.itemgetter(2))
        count = 0
        for info in not_served_ct_names_infos:
            if ((info[3] < cut_info[0] / speed and info[2] < cut_info[1] / speed)) or count < stopping:
                rev_not_served_ct_names.append(info[0])
                count += 1
                if count > stopping:
                    break
    else:
        rev_not_served_ct_names = not_served_ct_names
    for ct_name in rev_not_served_ct_names:
        if ct_name in rider_on_hand + rider_finished:
            input('ERROR {} :: 고려 고객 {} 제외1 {} 제외 2 {}'.format(ct_name, not_served_ct_names, rider_on_hand, rider_finished))
        else:
            res[ct_name] = customers[ct_name]
    if except_target_order == False:
        res[target_order.name] = target_order
    try:
        f = open('candi_num_cal.txt','a')
        info = '{};{};{};{};\n'.format(int(riders[0].env.now),revise_type,candi_num,len(res))
        f.write(info)
        f.close()
    except:
        pass
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
            route_dist += distance(before[0],before[1], after[0],after[1])
        OD_pair_dist.append(route_dist)
    p2p_dist = 0
    for order_name in orders:
        p2p_dist += distance(orders[order_name].store_loc[0],orders[order_name].store_loc[1],orders[order_name].location[0],orders[order_name].location[1])
    return min(OD_pair_dist), p2p_dist

#todo: 번들 생성 관련자
def ConstructFeasibleBundle_TwoSided(target_order, orders, s, p2, thres = 0.05, speed = 1, bundle_permutation_option = False, uncertainty = False,
                                     platform_exp_error = 1, print_option = True, sort_index = 5, now_t = 0, XGBoostModel = None, search_type = 'enumerate',
                                     feasible_return = True):
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
        #print(orders[customer_name].time_info[1], orders[customer_name].cancel)
        if customer_name != target_order.name and orders[customer_name].time_info[1] == None and orders[customer_name].cancel == False:
            d.append(customer_name)
    #print(d,s)
    #input("확인2")
    new = 0
    M2_count = 0
    if len(d) == 0:
        #input('번들을 구성할 고객이 없음')
        pass
    if len(d) > s - 1:
        M = itertools.permutations(d, s - 1)
        b = []
        if print_option == True:
            print('대상 고객 {} ::고려 고객들 {}'.format(target_order.name, d[:min(len(d)-1,10)]))
        for m in M:
            #print('대상 seq :: {}'.format(m))
            q = list(m) + [target_order.name]
            subset_orders = []
            time_thres = 0 #3개의 경로를 연속으로 가는 것 보다는
            for name in q:
                subset_orders.append(orders[name])
                time_thres += orders[name].distance/speed
            #input('확인 1 {} : 확인2 {}'.format(subset_orders, time_thres))
            if search_type == 'enumerate':
                counter2('old1',len(subset_orders))
                if thres < 100:
                    tem_route_info = BundleConsist(subset_orders, orders, p2, speed = speed, bundle_permutation_option= bundle_permutation_option, time_thres= time_thres, uncertainty = uncertainty, platform_exp_error = platform_exp_error, feasible_return = feasible_return, now_t = now_t)
                    # ver0: feasible_routes.append([route, round(max(ftds),2), round(sum(ftds)/len(ftds),2), round(min(ftds),2), order_names, round(route_time,2)])
                else:
                    tem_route_info = BundleConsist2(subset_orders, orders, p2, speed = speed, bundle_permutation_option= bundle_permutation_option, time_thres= time_thres, uncertainty = uncertainty, platform_exp_error = platform_exp_error, feasible_return = feasible_return, now_t = now_t)
                    # ver1: [route, unsync_t[0], round(sum(ftds) / len(ftds), 2), unsync_t[1], order_names, round(route_time, 2),min(time_buffer), round(P2P_dist - route_time, 2)]
                M2_count += 1
            elif search_type == 'XGBoost':
                #dataset 구성
                tem_route_info = [] #작동하지 않는 기능
                pass
            else:
                input('ConstructFeasibleBundle_TwoSided ERROR')
            #print('계산{} :: {}'.format(q, tem_route_info))
            if len(tem_route_info) > 0:
                OD_pair_dist, p2p_dist = MIN_OD_pair(orders, q, s)
                for info in tem_route_info: #todo: 번들 점수 내는 부분
                    #info.append((info[5]/p2p_dist)/s)
                    #info.append((info[1] + info[3]+info[5])/ s) #todo:220105번들 점수 내는 과정
                    #info.append((info[3] + info[5]) / s)  # todo:220105번들 점수 내는 과정
                    info.append((info[5]) / s)
            b += tem_route_info
            new += 1
        #input('가능 번들 수 {} : 정보 d {} s {}'.format(len(b), d, s))
        comparable_b = []
        if len(b) > 0:
            #sort_index = len(tem_route_info[0])-1  # 5: route time, 6: s_b
            sort_index = 5
            #b.sort(key=operator.itemgetter(6))  # s_b 순으로 정렬  #target order를 포함하는 모든 번들에 대해서 s_b를 계산.
            #print('정렬정보',b[0], sort_index)
            b.sort(key=operator.itemgetter(sort_index))
            b_star = b[0][sort_index]
            ave = []
            for ele in b:
                ave.append(ele[sort_index])
                if (ele[sort_index] - b_star)/b_star <= thres: #percent loss 가 thres 보다 작아야 함.
                    comparable_b.append(ele)
            #print('평균 {}'.format(sum(ave)/len(ave)))
        f = open('부하정도.txt', 'a')
        f.write(
            'Enu T;{};고객이름;{};B크기;{};신규;{};후보 수;{};대상 조합;{}; \n'.format(now_t, target_order.name, s, new, len(d), M2_count))
        f.close()
        return comparable_b
    else:
        return []


@jit(nopython=True)
def TriangleArea(s, d1,d2,d3):
    return math.sqrt(s * (s - d1) * (s - d2) * (s - d3))


def XGBoost_Bundle_Construct_tem(target_order, orders, s):
    d = []
    for customer_name in orders:
        loc_dist = distance(target_order.location[0],target_order.location[1],orders[customer_name].location[0],orders[customer_name].location[1])
        store_dist = distance(target_order.store_loc[0], target_order.store_loc[1], orders[customer_name].store_loc[0],
                    orders[customer_name].store_loc[1])
        if customer_name != target_order.name and orders[customer_name].time_info[1] == None and orders[customer_name].cancel == False and loc_dist <= 20 and store_dist <= 15:
            d.append(customer_name)
    # 1 : M1의 데이터에 대해서 attribute 계산 후 dataframe으로 만들기
    if len(d) <= s-1:
        return [], np.array([])
    M2 = itertools.permutations(d, s - 1)
    res = []
    for m in M2:
        q = list(m) + [target_order.name]
        q = list(q)
        q.sort()
        res.append(q)
    return res

def XGBoost_Bundle_Construct(target_order, orders, s, p2, XGBmodel, now_t = 0, speed = 1 , bundle_permutation_option = False, uncertainty = False,thres = 1,
                             platform_exp_error = 1,  thres_label = 1, label_check = None, feasible_return = True, fix_start = True, cut_info = [2500,2500], belonged_cts = []):
    print('XGBoost_Bundle_Construct 시작')
    d = []
    success_OO = [0]
    success_DD = [0]
    for customer_name in orders:
        if customer_name != target_order.name and orders[customer_name].time_info[1] == None and orders[customer_name].cancel == False:
            d.append(customer_name)
    # 1 : M1의 데이터에 대해서 attribute 계산 후 dataframe으로 만들기
    #print(d)
    print('대상 ::',len(d))
    #input('XGBoost_Bundle_Construct')
    start_time_sec = time.time()
    if len(d) <= s-1:
        return [], np.array([]) ,[[],[]]
    M1 = []
    input_data = []
    M2 = itertools.permutations(d, s - 1)
    M2_count = 0
    customer_names = []
    for m in M2:
        q = list(m) + [target_order.name]
        customer_names.append(q)
        tem1 = []
        tem2 = []
        # OD
        distOD = []
        gen_t = []
        ser_t = []
        for name in q:
            ct = orders[name]
            tem1.append(ct)
            tem2.append(ct.name)
            #distOD.append(ct.p2) #p2는 이동 시간임
            distOD.append(distance(ct.store_loc[0],ct.store_loc[1], ct.location[0],ct.location[1], rider_count='xgboost'))
            gen_t.append(ct.time_info[0])
            ser_t.append(ct.time_info[7])
        M1.append(tem1)
        #continue
        eachother = itertools.combinations(q, 2)
        distS = [] ##DD거리
        distC = [] #OO거리
        break_para = False
        for info in eachother:
            ct1 = orders[info[0]]
            ct2 = orders[info[1]]
            val1 = distance(ct1.store_loc[0],ct1.store_loc[1], ct2.store_loc[0],ct2.store_loc[1], rider_count='xgboost')
            if val1 > cut_info[0]:
                break_para = True
                break
            val2 = distance(ct1.location[0],ct1.location[1], ct2.location[0],ct2.location[1], rider_count='xgboost')
            if val2 > cut_info[1]:
                break_para = True
                break
            distS.append(val1)
            distC.append(val2)
            """
            if val1 > 5 or val2 > 5:
                break_para = True
                break
            """
        if break_para == True:
            continue
        distOD.sort()
        distS.sort()
        distC.sort()
        gen_t.sort()
        ser_t.sort()
        tem2 += distOD + distC + distS + gen_t + ser_t
        ##0916 추가된 부분
        ## --------start------
        vectors = []
        for name in q:
            vectors += [orders[name].store_loc[0] - orders[name].location[0],
                        orders[name].store_loc[1] - orders[name].location[1]]
        if len(q) == 2:
            triangles = [0, 0]
        else:
            if min(distS) <= 0:
                v1 = 0.0
            else:
                s1 = sum(distS) / 2
                try:
                    v1 = float(TriangleArea(s1,distS[0],distS[1],distS[2]))
                    #v1 = float(np.sqrt(s1 * (s1 - distS[0]) * (s1 - distS[1]) * (s1 - distS[2])))
                except:
                    v1 = -1
                    #print('SS TRIA ; distS;', distS)
                    # input('distS;확인1')
                    pass
            if min(distC) <= 0:
                v2 = 0.0
            else:
                s2 = sum(distC) / 2
                try:
                    v2 = float(TriangleArea(s2,distC[0],distC[1],distC[2]))
                    #v2 = float(np.sqrt(s2 * (s2 - distC[0]) * (s2 - distC[1]) * (s2 - distC[2])))
                except:
                    v2 = -1
                    #print('CC TRIA ; distC;', distC)
                    # input('distC;확인1')
                    pass
            if type(v1) != float or type(v2) != float:
                #print(distC, distS)
                #print('확인2', v1, v2, type(v1), type(v2))
                # input('VVV;확인3')
                pass
            triangles = [v2,v1]

        tem2 += vectors + triangles
        ##0916 추가된 부분
        ## ------end------
        input_data.append(tem2)
        M2_count += 1
    new = 0
    if now_t - 5 <= target_order.time_info[0]:
        new  = 1
    input_data = np.array(input_data)
    org_df = pd.DataFrame(data=input_data)
    X_test = org_df.iloc[:,s:] #탐색 번들에 따라, 다른 index 시작 지점을 가짐.
    X_test_np = np.array(X_test)
    counter2('sess1',len(X_test_np))
    print('B',s,':입력 고객 묶음 수', len(X_test_np))
    #input('확인6789')
    end_time_sec = time.time()
    duration = end_time_sec - start_time_sec
    if s == 2:
        t_counter('test10', duration)
    else:
        t_counter('test11', duration)
    #print(input_data[:2])
    #print(X_test_np[:2])
    #input('test중')
    #2 : XGModel에 넣기
    #start_time_sec = datetime.now()
    start_time_sec = time.time()
    if len(X_test_np) > 0:
        tem_test = [[],[]]
        cutter = 100
        for tem_index in range(int(len(X_test_np)/cutter)+1):
            s_index = cutter * tem_index
            e_index = cutter * (tem_index + 1)
            tem_data = X_test_np[s_index: min(e_index, len(X_test_np))]
            print('작업 대상::',s_index,'~', min(e_index, len(X_test_np)))
            if len(tem_data) > 0:
                tem_pred_onx = XGBmodel.run(None, {"feature_input": tem_data.astype(np.float32)})  # Input must be a list of dictionaries or a single numpy array for input 'input'.
                #tem_pred_onx = XGBmodel.run(None, {"feature_input": tem_data.astype(np.float32)}) <- ORG
                print(tem_pred_onx[0])
                print(list(tem_pred_onx[0]))
                print('고객 묶음 수',len(tem_pred_onx[0]))
                #print(tem_pred_onx[1])
                print('cutter', sum(tem_pred_onx[0]))
                #input('확인4678')
                tem_test[0] += list(tem_pred_onx[0])
        pred_onx = tem_test
        """
        pred_onx = XGBmodel.run(None, {"feature_input": X_test_np.astype(np.float32)})  # Input must be a list of dictionaries or a single numpy array for input 'input'.
        pred_onx = [[], []]
        pass_ratio = 0.01
        for _ in range(len(X_test_np)):
            if random.random() < pass_ratio:
                pred_onx[0].append(1)
            else:
                pred_onx[0].append(0)
        """
        pass
    else:
        return [], [],[]
    #end_time_sec = datetime.now()
    end_time_sec = time.time()
    duration = end_time_sec - start_time_sec
    #duration = duration.seconds + duration.microseconds / 1000000
    t_counter('sess', duration)
    start_time_sec = time.time()
    #print("predict", pred_onx[0], type(pred_onx[0]))
    #print("predict_proba", pred_onx[1][:1])
    #input('test중2')
    #y_pred = XGBmodel.predict(X_test)
    #labeled_org_df = pd.merge(y_pred, org_df, left_index=True, right_index=True)
    #3 : label이 1인 것에 대해, 경로 만들고 실제 가능 여부 계산
    constructed_bundles = []
    labels = []
    labels_larger_1 = []
    count = 0
    count1 = 0
    rc_count = 0
    print('pred_onx 수 세기 ',sum(pred_onx[0]))
    for label in pred_onx[0]:
        labels.append(int(label))
        if 0 < label <= 100: #todo : 0916 label # thres_label
            rev_M1 = [M1[count]]  # [M1[count]] #todo 1110 에러 발생 원인
            rev_M1_names =[]
            print('BC계산 시도!!')
            for name in belonged_cts:
                rev_M1.append(M1[count][:s-1] + [orders[name]])
            for s_orders in rev_M1:
                tem_name = []
                for s_order in s_orders:
                    tem_name.append(s_order.name)
                rev_M1_names.append(tem_name)
            if len(rev_M1_names) > 1:
                #print('Target',target_order.name)
                #print('확장 됨',rev_M1_names)
                #input('rev_M1 확인')
                pass
            print('rev_M1',rev_M1)
            for info in rev_M1:
                #info = 이전의 M1[count]
                #if label >= thres_label:
                    #print('라벨',label)
                rc_count += 1
                print('가즈아!!')
                if thres < 100 :
                    print('1::',M1[count])
                    tem = BundleConsist(info, orders, p2, speed = speed,
                                         bundle_permutation_option = bundle_permutation_option, uncertainty = uncertainty, platform_exp_error =  platform_exp_error,
                                         feasible_return = True, now_t = now_t)
                else:
                    print('2::',M1[count])
                    print('orders',orders)
                    print('ct# :: store_loc :: ct_loc')
                    tem = BundleConsist2(info, orders, p2, speed = speed,
                                         bundle_permutation_option = bundle_permutation_option, uncertainty = uncertainty, platform_exp_error =  platform_exp_error,
                                         feasible_return = feasible_return, now_t = now_t, max_dist= 15, fix_start = fix_start) #max_dist= 15
                    #print('구성 된 라벨 1 ::', label)
                    #print(tem)
                    labels_larger_1.append(int(label))
                if len(tem) > 0:
                    #constructed_bundles.append(tem)
                    constructed_bundles += tem
                    #input('번들 생성')
                    if s == 3:
                        success_DD += list(X_test_np[count][3:6])
                        success_OO += list(X_test_np[count][6:9])
                        #print(success_DD)
                        #print(success_OO)
                #if count1 > 0.12*len(X_test_np):
                #    break
                count1 += 1
            count += 1
    f = open('부하정도.txt','a')
    f.write('XGB T;{};고객이름;{};B크기;{};신규;{};후보 수;{};대상 조합;{};RC;{};후보 고객 수;{};1인 경우:{} \n'.format(now_t, target_order.name,s, new,len(d),M2_count, rc_count, len(d), len(labels_larger_1)))
    f.close()

    if len(labels_larger_1) > 0 :
        print('계산된 label 있음',len(labels_larger_1), sum(labels_larger_1)/len(labels_larger_1))
    else:
        print('계산된 label 없음')
        #input('문제 발새앵')
    counter2('sess2', count1)
    label_check = np.append(label_check, pred_onx[0])
    end_time_sec = time.time()
    duration = end_time_sec - start_time_sec
    t_counter('test12', duration)
    #print('확인용1',labels)
    #print('확인용2',labels_larger_1)
    #label_check = np.concatenate((label_check, pred_onx[0]))
    #unique, counts = np.unique(pred_onx[0], return_counts=True)
    #print(str(dict(zip(unique, counts))))
    #print('1:{}; 2:{}; 3:{};4:{};'.format(pred_onx[0].count(1),pred_onx[0].count(2),pred_onx[0].count(3),pred_onx[0].count(4)))
    #input('숫자 확인')
    if sum(pred_onx[0]) > 0:
        #print(constructed_bundles)
        #input('확인2')
        #print('번들 발생함::',len(constructed_bundles))
        pass
    try:
        pass
    except:
        input('확인')
    add_info = [success_DD, success_OO]
    return constructed_bundles, np.array(labels), add_info


def XGBoost_Bundle_Construct_ORG(target_order, orders, s, p2, XGBmodel, now_t = 0, speed = 1 , bundle_permutation_option = False, uncertainty = False,thres = 1,
                             platform_exp_error = 1,  thres_label = 1, label_check = None, feasible_return = True, fix_start = True, cut_info = [2500,2500], belonged_cts = None):
    #print('run1')
    d = []
    success_OO = [0]
    success_DD = [0]
    for customer_name in orders:
        if customer_name != target_order.name and orders[customer_name].time_info[1] == None and orders[customer_name].cancel == False:
            d.append(customer_name)
    # 1 : M1의 데이터에 대해서 attribute 계산 후 dataframe으로 만들기
    #print(d)
    #input('XGBoost_Bundle_Construct')
    start_time_sec = time.time()
    if len(d) <= s-1:
        return [], np.array([]) ,[[],[]]

    M1 = []
    input_data = []
    M2 = itertools.permutations(d, s - 1)
    M2_count = 0
    customer_names = []

    for m in M2:
        q = list(m) + [target_order.name]
        customer_names.append(q)
        tem1 = []
        tem2 = []
        # OD
        distOD = []
        gen_t = []
        ser_t = []
        for name in q:
            ct = orders[name]
            tem1.append(ct)
            tem2.append(ct.name)
            #distOD.append(ct.p2) #p2는 이동 시간임
            distOD.append(distance(ct.store_loc[0],ct.store_loc[1], ct.location[0],ct.location[1], rider_count='xgboost'))
            gen_t.append(ct.time_info[0])
            ser_t.append(ct.time_info[7])
        M1.append(tem1)
        #continue
        eachother = itertools.combinations(q, 2)
        distS = [] ##DD거리
        distC = [] #OO거리
        break_para = False
        for info in eachother:
            ct1 = orders[info[0]]
            ct2 = orders[info[1]]
            val1 = distance(ct1.store_loc[0],ct1.store_loc[1], ct2.store_loc[0],ct2.store_loc[1], rider_count='xgboost')
            if val1 > cut_info[0]:
                break_para = True
                break
            val2 = distance(ct1.location[0],ct1.location[1], ct2.location[0],ct2.location[1], rider_count='xgboost')
            if val2 > cut_info[1]:
                break_para = True
                break
            distS.append(val1)
            distC.append(val2)
            """
            if val1 > 5 or val2 > 5:
                break_para = True
                break
            """
        if break_para == True:
            continue
        distOD.sort()
        distS.sort()
        distC.sort()
        gen_t.sort()
        ser_t.sort()
        tem2 += distOD + distC + distS + gen_t + ser_t
        ##0916 추가된 부분
        ## --------start------
        vectors = []
        for name in q:
            vectors += [orders[name].store_loc[0] - orders[name].location[0],
                        orders[name].store_loc[1] - orders[name].location[1]]
        if len(q) == 2:
            triangles = [0, 0]
        else:
            if min(distS) <= 0:
                v1 = 0.0
            else:
                s1 = sum(distS) / 2
                try:
                    v1 = float(TriangleArea(s1,distS[0],distS[1],distS[2]))
                    #v1 = float(np.sqrt(s1 * (s1 - distS[0]) * (s1 - distS[1]) * (s1 - distS[2])))
                except:
                    v1 = -1
                    #print('SS TRIA ; distS;', distS)
                    # input('distS;확인1')
                    pass
            if min(distC) <= 0:
                v2 = 0.0
            else:
                s2 = sum(distC) / 2
                try:
                    v2 = float(TriangleArea(s2,distC[0],distC[1],distC[2]))
                    #v2 = float(np.sqrt(s2 * (s2 - distC[0]) * (s2 - distC[1]) * (s2 - distC[2])))
                except:
                    v2 = -1
                    #print('CC TRIA ; distC;', distC)
                    # input('distC;확인1')
                    pass
            if type(v1) != float or type(v2) != float:
                #print(distC, distS)
                #print('확인2', v1, v2, type(v1), type(v2))
                # input('VVV;확인3')
                pass
            triangles = [v2,v1]

        tem2 += vectors + triangles
        ##0916 추가된 부분
        ## ------end------
        input_data.append(tem2)
        M2_count += 1
    new = 0
    if now_t - 5 <= target_order.time_info[0]:
        new  = 1
    input_data = np.array(input_data)
    org_df = pd.DataFrame(data=input_data)
    X_test = org_df.iloc[:,s:] #탐색 번들에 따라, 다른 index 시작 지점을 가짐.
    X_test_np = np.array(X_test)
    counter2('sess1',len(X_test_np))
    end_time_sec = time.time()
    duration = end_time_sec - start_time_sec
    if s == 2:
        t_counter('test10', duration)
    else:
        t_counter('test11', duration)
    #print(input_data[:2])
    #print(X_test_np[:2])
    #input('test중')
    #2 : XGModel에 넣기
    #start_time_sec = datetime.now()
    start_time_sec = time.time()
    if len(X_test_np) > 0:
        pred_onx = XGBmodel.run(None, {"feature_input": X_test_np.astype(np.float32)})  # Input must be a list of dictionaries or a single numpy array for input 'input'.
    else:
        return [], [],[]
    #end_time_sec = datetime.now()
    end_time_sec = time.time()
    duration = end_time_sec - start_time_sec
    #duration = duration.seconds + duration.microseconds / 1000000
    t_counter('sess', duration)
    start_time_sec = time.time()
    #print("predict", pred_onx[0], type(pred_onx[0]))
    #print("predict_proba", pred_onx[1][:1])
    #input('test중2')
    #y_pred = XGBmodel.predict(X_test)
    #labeled_org_df = pd.merge(y_pred, org_df, left_index=True, right_index=True)
    #3 : label이 1인 것에 대해, 경로 만들고 실제 가능 여부 계산
    constructed_bundles = []
    labels = []
    labels_larger_1 = []
    count = 0
    count1 = 0
    rc_count = 0
    for label in pred_onx[0]:
        labels.append(int(label))
        if 0 < label <= thres_label: #todo : 0916 label
            rc_count += 1
            if thres < 100 :
                print('1::',M1[count])
                tem = BundleConsist(M1[count], orders, p2, speed = speed,
                                     bundle_permutation_option = bundle_permutation_option, uncertainty = uncertainty, platform_exp_error =  platform_exp_error,
                                     feasible_return = True, now_t = now_t)
            else:
                #print('2::',M1[count])
                #print('orders',orders)
                #print('ct# :: store_loc :: ct_loc')
                tem = BundleConsist2(M1[count], orders, p2, speed = speed,
                                     bundle_permutation_option = bundle_permutation_option, uncertainty = uncertainty, platform_exp_error =  platform_exp_error,
                                     feasible_return = feasible_return, now_t = now_t, max_dist= 15, fix_start = fix_start) #max_dist= 15
                #print('구성 된 라벨 1 ::', label)
                #print(tem)
                labels_larger_1.append(int(label))
            if len(tem) > 0:
                #constructed_bundles.append(tem)
                constructed_bundles += tem
                #input('번들 생성')
                if s == 3:
                    success_DD += list(X_test_np[count][3:6])
                    success_OO += list(X_test_np[count][6:9])
                    #print(success_DD)
                    #print(success_OO)
            #if count1 > 0.12*len(X_test_np):
            #    break
            count1 += 1
        count += 1
    f = open('부하정도.txt','a')
    f.write('XGB T;{};고객이름;{};B크기;{};신규;{};후보 수;{};대상 조합;{};RC;{};후보 고객 수;{}; \n'.format(now_t, target_order.name,s, new,len(d),M2_count, rc_count, len(d)))
    f.close()
    """
    if len(labels_larger_1) > 0 :
        print('계산된 label 있음',len(labels_larger_1), sum(labels_larger_1)/len(labels_larger_1))
    else:
        print('계산된 label 없음')
    """
    counter2('sess2', count1)
    label_check = np.append(label_check, pred_onx[0])
    end_time_sec = time.time()
    duration = end_time_sec - start_time_sec
    t_counter('test12', duration)
    #print('확인용1',labels)
    #print('확인용2',labels_larger_1)
    #label_check = np.concatenate((label_check, pred_onx[0]))
    #unique, counts = np.unique(pred_onx[0], return_counts=True)
    #print(str(dict(zip(unique, counts))))
    #print('1:{}; 2:{}; 3:{};4:{};'.format(pred_onx[0].count(1),pred_onx[0].count(2),pred_onx[0].count(3),pred_onx[0].count(4)))
    #input('숫자 확인')
    if sum(pred_onx[0]) > 0:
        #print(constructed_bundles)
        #input('확인2')
        #print('번들 발생함::',len(constructed_bundles))
        pass
    try:
        pass
    except:
        input('확인')
    add_info = [success_DD, success_OO]
    return constructed_bundles, np.array(labels), add_info


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
        dist = distance(target.store_loc[0],target.store_loc[1], customer2.store_loc[0],customer2.store_loc[1])
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
    len_a = distance(p0[0],p0[1], p1[0],p1[1])
    for customer in C_T:
        p2 = customer.location
        len_b = distance(p0[0],p0[1], p2[0],p2[1])
        if len_b > len_a or len_a == 0 or len_b == 0: #3개의 점이 필요하기 때문
            continue
        len_c = distance(p1[0],p1[1], p2[0],p2[1])
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
        dist0 = distance(target.store_loc[0],target.store_loc[1], target.location[0],target.location[1])*w
        dist1 = distance(target.store_loc[0],target.store_loc[1], customer2.store_loc[0],customer2.store_loc[1])
        dist2 = distance(target.store_loc[0],target.store_loc[1], customer2.location[0],customer2.location[1])
        dist3 = distance(target.store_loc[0],target.store_loc[1], customer2.location[0],customer2.location[1])
        dist4 = distance(middle[0],middle[1], customer2.store_loc[0],customer2.store_loc[1])
        dist5 = distance(middle[0],middle[1], customer2.location[0],customer2.location[1])
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
        dist0 = distance(target.store_loc[0],target.store_loc[1], target.location[0],target.location[1])
        dist1 = distance(target.store_loc[0],target.store_loc[1], customer2.store_loc[0],customer2.store_loc[1])
        dist2 = distance(target.location[0],target.location[1], customer2.store_loc[0],customer2.store_loc[1])
        dist3 = distance(target.store_loc[0],target.store_loc[1], customer2.location[0],customer2.location[1])
        dist4 = distance(target.location[0],target.location[1], customer2.location[0],customer2.location[1])
        if dist1 + dist2 <= dist0 + delta and dist3 + dist4 <= dist0 +delta:
            res_C_T[customer2.name] = customers[customer2.name]
    return res_C_T


def pareto_ranking(datas, val_index1, val_index2, option_index = None):
    """
    datas에 대해 pareto dominance rank계산
    @param datas:
    @param val_index1:
    @param val_index2:
    @return:
    """
    if len(datas) == 1:
        if option_index == None:
            return [[0,0]]
        else:
            return [[0, 0,datas[0][option_index]]]
    res = []
    index1 = 0
    for data1 in datas:
        index2 = 0
        val = 0
        for data2 in datas:
            if index1 != index2:
                if data1[val_index1] > data2[val_index1] and  data1[val_index2] > data2[val_index2]:
                    val += 1
            index2 += 1
        if option_index != None:
            res.append([index1, val, data1[option_index]])
        else:
            res.append([index1, val])
        index1 += 1
    res.sort(key=operator.itemgetter(1))
    return res
def SingleCustomerBundleInsert(customer, customers, bundle_infos, rider_names, riders, rider_infos, p2 = 2):
    #1 customer가 운행 중인 라이더와 선택될 수 있는 위치 인지를 계산
    res = []
    close_rider_names = []
    index = 0
    for rider_name in rider_names:
        rider = riders[rider_name]
        rider_end_loc = rider.route[-1][2]
        dist = distance(customer.store_loc[0],customer.store_loc[0], rider_end_loc[0],rider_end_loc[1])
        if dist <= rider_infos[index]:
            close_rider_names.append(rider.name)
    # 2 기존 번들에 customer가 삽입 될 수 있는지를 계산
    insert_cost = []
    for bundle_info in bundle_infos:
        # change customer to bundle
        tem = SubF1_SingleCustomerBundleInsert(customer, customers, bundle_info, rider_names, riders, p2=p2)
        if len(tem) > 0:
            tem2 = [index] + tem
            insert_cost.append(tem2)
            index += 1
    #3 계산된 삽입 가능 위치들에 대해 sort

    if len(res) > 0:
        return res
    else:
        return []

def SubF1_SingleCustomerBundleInsert(t_customer, customers, input_bundle_info, rider_names, riders, p2 = 2, pr_off = True):
    """
    대상 고객(class)와 BundleConsist2의 bundle_info가 주어졌을 때, 대상 고객이 bundle_info에 삽입 될 수 있는지 여부를 계산
    @param t_customer:
    @param customers:
    @param bundle_info:
    @param rider_names:
    @param riders:
    @param p2:
    @return:
    """
    # tem = [route, unsync_t[0], round(sum(ftds) / len(ftds), 2), unsync_t[1], order_names, round(route_time, 2),min(time_buffer), round(P2P_dist - route_time, 2), line_dist, round(P2P_dist,4), distance(origin[0],origin[1], destination[0],destination[1])/speed]
    #print(input_bundle_info)
    bundle_info = input_bundle_info
    #input('error전에 확인')
    possible_combination = itertools.combinations(bundle_info[4], len(bundle_info[4]) - 1)
    ava_bundle_infos = []
    if pr_off == False:
        org_likely2choose = max(BundleExpValueCalculator([bundle_info], rider_names, riders, customers)) # todo 1110 : 연산량이 많은 것 같음 잠시 꺼두자.
    else:
        org_likely2choose = 1
    org_cost = bundle_info[7] #고객 진선 거리 합 - 경로 시간; 클 수록 절약한 시간이 증가함을 의미
    info_index = 0
    for part in possible_combination:
        ct_names = list(part) + [t_customer.name]
        cts = []
        for name in ct_names:
            cts.append(customers[name])
        tem_bundle_info = BundleConsist2(cts, customers, p2,feasible_return= True) #가능한 모든 번들만 반환; [번들1 정보, 번들2_정보,..., 번들n_정보]
        if len(tem_bundle_info) > 0:
            for b_info in tem_bundle_info:
                if pr_off == False:
                    rev_likely2choose = max(BundleExpValueCalculator([b_info], rider_names, riders, customers))
                else:
                    rev_likely2choose = 1
                rev_cost = b_info[7]
                if rev_cost < org_cost or org_likely2choose < rev_likely2choose : #현재는 2중에 1개만 개선이 되어도 번들에 추가.
                    ava_bundle_infos.append([info_index, b_info, rev_cost, rev_likely2choose, org_cost - rev_cost, rev_likely2choose - org_likely2choose])
                    info_index += 1
    if len(ava_bundle_infos) > 0:
        # sort 기준
        #ava_bundle_infos.sort(key=operator.itemgetter(4))
        #print('ranked_ava_bundle_infos',ava_bundle_infos)
        #input('파레호 확인1')
        ranked_ava_bundle_infos = pareto_ranking(ava_bundle_infos, -1, -2)
        #print('return',ava_bundle_infos[ranked_ava_bundle_infos[0][0]])
        #input('파레호 확인2')
        return ava_bundle_infos[ranked_ava_bundle_infos[0][0]]
    else:
        return []

def DynamicBundleConstruct(t_customer, customers, rider_names, riders, platform, now_t, p2=2, stopping = 15,
                           bundle_permutation_option=False,feasible_return=False, min_time_buffer=10, max_dist=15, sort_index=8, fix_start=False,
                           pr_off = True, check_info = None, dec_ratio = 0.5):
    """
    Dynamic 연산 수행 function
    @param t_customer: 새로 발생한 고객  class
    @param customers: 번들 생성에 고려할 고객들
    @param rider_names: 운행 중인 라이더 이름
    @param riders: 라이더 class dict
    @param platform: platform class
    @param now_t: 현재 시간
    @param p2: 라이더 p2
    @param stopping: BundleConsideredCustomers가 종료할 고객 순위
    @param bundle_permutation_option: BundleConsideredCustomers option
    @param feasible_return: BundleConsideredCustomers option
    @param min_time_buffer:  BundleConsideredCustomers option
    @param max_dist:  BundleConsideredCustomers option
    @param sort_index:  BundleConsideredCustomers option
    @param fix_start:  BundleConsideredCustomers option
    @param pr_off: 라이더가 주문을 선택할지 여부 True: 계산X ;False : 계산O
    @param check_info: print_확인용
    @param dec_ratio:  B3 구성에 연산량 증가를 줄위기 위한 장치 #todo 1111 : 수정이 필요한 부분..
    @return: type (int), bundle_info(list) <- BundleConsist2 의 output
    """
    try:
        rider_speed = riders[0].speed
    except:
        rider_speed = 1
    #1 기존 번들에 삽입 될 수 있을까?
    res1 =  []
    for task_index in platform.platform:
        task = platform.platform[task_index]
        if len(task.customers) > 1: #번들
            #print('task.old_info',task.old_info)
            s_time1 = time.time()
            tem = SubF1_SingleCustomerBundleInsert(t_customer, customers, task.old_info, rider_names, riders, p2=p2, pr_off = pr_off)
            e_time1 = time.time()
            try:
                check_info['SubF1_SingleCustomerBundleInsert'][0] += 1
                check_info['SubF1_SingleCustomerBundleInsert'][1] += (e_time1 - s_time1)
            except:
                pass
            if len(tem) > 0:
                #print('tem', tem)
                #input('tem확인1')
                tem.append(task.index)
                res1.append(tem)
            #tem 구조 :  [info_index, b_info, rev_cost, rev_likely2choose, org_cost - rev_cost, rev_likely2choose - org_likely2choose, task.index] #b_info는 BundleConsist2의 반환 값
    #2 새로운 번들로 구성될 수 있을까?
    considered_customers = BundleConsideredCustomers(t_customer, platform, riders, customers, speed=rider_speed, stopping=stopping, revise_type = 'stopping', except_target_order= True) #todo : 확인 할 것
    #2-1 B3 계산
    s_time2 = time.time()
    count = 0
    M2 = itertools.permutations(considered_customers, 2)
    res2_3 = []
    for subset in M2:
        subset_orders = [t_customer]
        for name in subset:
            subset_orders.append(customers[name])
        tem = BundleConsist2(subset_orders, customers, p2, speed=rider_speed, bundle_permutation_option=bundle_permutation_option,feasible_return=feasible_return,
                             now_t=now_t, min_time_buffer=min_time_buffer, max_dist=max_dist*dec_ratio, sort_index=sort_index, fix_start=fix_start)
        if tem != []:
            if pr_off == False:
                likely_to_choose = max(BundleExpValueCalculator(tem, rider_names, riders, customers))
            else:
                likely_to_choose = 1
            tem.append(likely_to_choose)
            res2_3.append(tem)
            #print('추가됨33', tem)
        count += 1
    B3_count = copy.deepcopy(count)
    e_time2 = time.time()
    try:
        check_info['B3'][0] += count
        check_info['B3'][1] += (e_time2 - s_time2)
    except:
        pass
    #2-2 B2 계산
    res2_2 = []
    s_time3 = time.time()
    M2 = itertools.permutations(considered_customers, 1)
    if len(list(M2)) != len(considered_customers) or B3_count < len(list(M2)):
        print('길이 확인!',B3_count, len(list(M2)), len(considered_customers))
    """
    """
    for subset in M2:
        subset_orders = [t_customer]
        for name in subset:
            subset_orders.append(customers[name])
        #tem = BundleConsist2(subset_orders, customers, p2, speed=rider_speed, bundle_permutation_option=bundle_permutation_option,feasible_return=feasible_return,
        #                     now_t=now_t, min_time_buffer=min_time_buffer, max_dist=max_dist, sort_index=sort_index, fix_start=fix_start)
        tem = []
        if tem != []:
            if pr_off == False:
                likely_to_choose = max(BundleExpValueCalculator(tem, rider_names, riders, customers))
            else:
                likely_to_choose = 1
            #print(tem, likely_to_choose)
            tem.append(likely_to_choose)
            res2_2.append(tem)
            #print('추가됨22', tem)
        count += 1

    e_time3 = time.time()
    try:
        check_info['B2'][0] += count
        check_info['B2'][1] += (e_time3 - s_time3)
    except:
        check_info['B2'][0] += 1
        check_info['B2'][1] += (e_time3 - s_time3)
    if len(considered_customers) > 4:
        #print(t_customer.name)
        #print('대상 고객 들', considered_customers)
        #print('수행 횟 수:', count)
        #input('Dynamic 번들')
        pass
    #bundle 중 선택
    if len(res1) + len(res2_2) + len(res2_3) > 0:
        #print('기존 번들 수정',res1)
        #print('신규 B2',res2_2)
        #print('신규 B3',res2_3)
        #input('번들 가능성 존재')
        pass
    res_rank = []
    index = 0
    for info in res1:
        print('1',info)
        res_rank.append([index,0,info[2],info[3],info[6]])
        index += 1
    index = 0
    for info in res2_2:
        #info = [기존 번들 info, likely_to_choose]
        #print('2',info)
        #input('확인223')
        res_rank.append([index,1,info[0][5],info[-1],-1])
        index += 1
    index = 0
    for info in res2_3:
        #print('3', info)
        res_rank.append([index,2,info[0][5],info[-1],-1])
        index += 1
    b_type = -1
    if len(res_rank) > 0:
        #print('res_rank',res_rank)
        res_rank = pareto_ranking(res_rank, 2, 3, option_index=1)
        dominant_res = res_rank[0]
        #print(dominant_res)
        #input('data 확인')
        if dominant_res[2] == 0:
            res_info = res1[dominant_res[0]]
            #print(res_info)
            #input('res_info')
            b_type = 0
            del platform.platform[res_info[6]] #수정될 task를 제거
        elif dominant_res[2] == 1:
            res_info = res2_2[dominant_res[0]]
            b_type = 1
        elif dominant_res[2] == 2:
            res_info = res2_3[dominant_res[0]]
            b_type = 2
        else:
            res_info = None
            b_type = -1
            input('rank error')
        #print(res_info)
        #print('반환 정보')
        return b_type, res_info
    else:
        return -1, []

def OrdergeneratorByCSVForStressTestDynamic(env, orders, stores, lamda, platform = None, p2_ratio = 1, rider_speed = 1, unit_fee = 110, fee_type = 'linear',
                                     output_data = None, cooktime_detail = None, cook_first = False, dynamic_infos = None, riders = None, pr_off = True, end_t = 90,
                                            dynamic_para = False, customer_pend = False, search_range_index = 15):
    """
    Generate customer order
    :param env: Simpy Env
    :param orders: Order
    :param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param stores: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param interval: 주문 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    """
    #yield env.timeout(5)
    save_txt = False
    t_info = {'CountActiveRider':[0,0],'DynamicBundleConstruct':[0,0],'SubF1_SingleCustomerBundleInsert':[0,0],'B2':[0,0],'B3':[0,0]}
    dist_distribution = np.random.poisson(20,7) #todo: 0915 거리 조절
    for count in range(1000000):
        if output_data == None:
            store_name = random.choice(range(len(stores)))
            store = stores[store_name]
            store_loc = store.location
            #req_dist = random.randint(5,20)*rider_speed
            req_dist = max(random.choice(dist_distribution), 7)
            angle = math.radians(random.randrange(0, 360))
            num = 0
            while num < 1000:
                customer_loc = [store_loc[0] + round(req_dist*math.cos(angle),4),store_loc[1] + round(req_dist*math.sin(angle),4) ]
                if customer_loc[0] <= 50 and customer_loc[1] <= 50 and store_loc != customer_loc:
                    break
                num += 1
            if num == 1000:
                customer_loc = [random.randint(0,50),random.randint(0,50)]
        else:
            if count > len(output_data) - 2:
                break
            store_name = output_data[count][3]
            store_loc = [output_data[count][4], output_data[count][5]]
            customer_loc = [output_data[count][1], output_data[count][2]]
        name = count
        if cooktime_detail != None:
            cook_time = np.random.choice(cooktime_detail[0], p = cooktime_detail[1]) # todo : 221101실험을 현실적으로 변경.
            p2_ratio2 = store.p2
        else:
            cook_time = 3
            p2_ratio2 = p2_ratio
        OD_dist = distance(store_loc[0],store_loc[1], customer_loc[0],customer_loc[1])
        p2 = (OD_dist / rider_speed) * p2_ratio2 # todo : 221101실험을 현실적으로 변경.
        cook_time_type = 0
        cooking_time = [7,1]
        #order = A1_Class.Customer(env, name, input_location, store=store_num, store_loc=store_loc, p2=p2,
        #                       cooking_time=cook_time, cook_info=[cook_time_type, cooking_time])
        order = re_A1_class.Customer(env, name, customer_loc, store=store_name, store_loc=store_loc, p2=p2,
                               cooking_time=cook_time, cook_info=[cook_time_type, cooking_time], platform = platform, unit_fee = unit_fee, fee_type = fee_type)
        #order.actual_cook_time = random.choice(stores[store_name].FRT)
        order.actual_cook_time = cook_time
        order.dp_cook_time = cook_time
        order.dp_cook_time = 5*(1 + order.actual_cook_time//5)
        order.cancel = True #주문이 생성되자 마자 수행될 수 있도록
        if cooktime_detail != None:
            order.temperature = store.temperature
            order.rest_type = store.rest_type
        else:
            order.temperature = 'T'
        if order.dp_cook_time >= 15 and cook_first == True:
            order.cooking_process = env.process(order.CookingFirst(env, order.actual_cook_time)) #todo : 15분 이상 음식은 미리 조리 시작
        order.cancel = customer_pend
        if customer_pend == False:
            if len(list(platform.platform.keys())) > 0:
                task_index = max(list(platform.platform.keys())) + 1
            else:
                task_index = 1
            o = GenSingleOrder(task_index, order)  # todo 1115 : 주문을 추가
            platform.platform[task_index] = o
        print('T {} 음식 {} 조리 확인/ 시간 {}'.format(int(env.now), order.name,order.actual_cook_time))
        orders[name] = order
        stores[store_name].received_orders.append(orders[name])
        interval = 1.0/lamda
        #todo : 0317 지연되는 조건 생각할 것.
        if dynamic_para == True and dynamic_infos != None and riders != None:
            platform_interval = 5
            s_time = time.time()
            active_rider_names, d_infos, time_data = CountActiveRider(riders, platform_interval, min_pr=0.05, t_now=env.now,option='w', point_return=True, print_option=False)
            e_time = time.time()
            t_info['CountActiveRider'][0] += 1
            t_info['CountActiveRider'][1] += (e_time - s_time)
            #파라메터 정의
            p2 = dynamic_infos[0]
            bundle_permutation_option = dynamic_infos[1]
            feasible_return = dynamic_infos[2]
            min_time_buffer = dynamic_infos[3]
            max_dist = dynamic_infos[4]
            sort_index = dynamic_infos[5]
            fix_start = dynamic_infos[6]
            s_time = time.time()
            #이 고객이 active rider와 가까이 있는지를 확인해 볼 것
            tem_count = 0
            search_index = search_range_index  # 이전에는 30
            BundleCloseRider = {}
            max_ds = []
            for p1 in d_infos:
                tem = []
                # BundleCloseRider[active_rider_names[tem_count]] = []
                for order_name in orders:
                    order = orders[order_name]
                    if order.time_info[1] == None and order.cancel == False:
                        tem.append(
                            [order.name, distance(p1[0], p1[1], order.store_loc[0], order.store_loc[1])])
                        if tem_count == 0:
                            BundleCloseRider[order.name] = []
                tem.sort(key=operator.itemgetter(1))
                try:
                    max_d = tem[min(search_index, len(tem) - 1)][1]  # max_d : 라이더로 부터 search_index 까지 멀리 떨어진 지점의 이동시간
                    for dist_info in tem[:min(search_index, len(tem) - 1)]:
                        BundleCloseRider[dist_info[0]].append(active_rider_names[tem_count])
                except:
                    max_d = 10
                    pass
                max_ds.append([active_rider_names[tem_count],max_d])
                tem_count += 1
            dynamic_run = False
            for index_1 in range(d_infos):
                if distance(d_infos[index_1][0], d_infos[index_1][1], order.store_loc[0], order.store_loc[1]) <= max_ds[index_1]:
                    dynamic_run = True
                    break
            if dynamic_run == True:
                b_type, new_bundle_info1 = DynamicBundleConstruct(orders[name], orders, active_rider_names, riders, platform, env.now, p2=p2, stopping=15,
                                       bundle_permutation_option=bundle_permutation_option, feasible_return=feasible_return, min_time_buffer=min_time_buffer,
                                       max_dist=max_dist, sort_index=sort_index, fix_start=fix_start, pr_off= pr_off, check_info= t_info)
                e_time = time.time()
                t_info['DynamicBundleConstruct'][0] += 1
                t_info['DynamicBundleConstruct'][1] += (e_time - s_time)
                if b_type == 0:
                    new_bundle_info = new_bundle_info1[1]
                elif b_type == 1 or b_type == 2:
                    new_bundle_info = new_bundle_info1[0]
                else:
                    new_bundle_info = []
                #플랫폼에 order 추가
                if len(new_bundle_info) > 0:
                    # 겹치는 번들 삭제
                    #print(new_bundle_info)
                    #input("new_bundle_info")
                    del_task_indexs = []
                    for task in platform.platform:
                        check_task = platform.platform[task]
                        if len(check_task.customers) > 1:
                            for ct_name in check_task.customers:
                                if ct_name in new_bundle_info[4]:
                                    del_task_indexs.append(check_task.index)
                                    break
                    for del_index in del_task_indexs:
                        print('기존 B 삭제', platform.platform[del_index].customers)
                        del platform.platform[del_index]
                    #새로운 번들 추가
                    if len(list(platform.platform.keys())) > 0:
                        task_index = max(list(platform.platform.keys())) + 1
                    else:
                        task_index = 1
                    print(b_type, new_bundle_info)
                    #input('new_bundle_info')
                    o = GenBundleOrder(task_index, new_bundle_info, orders, env.now, add_fee=0)
                    o.old_info = new_bundle_info
                    #todo 1115: exp rider 계산 추가 Start -> 부하가 매우 크게 발생할 것임.
                    #active_rider_names, d_infos, times = CountActiveRider(riders, interval, min_pr=0.05, t_now=env.now,option='w', point_return=True, print_option=False)
                    # d_infos : 라이더가 주문을 선택할 지점 = 라이더가 주문을 완료하는 지점
                    tem_riders = []
                    for ct_name in new_bundle_info[4]:
                        try:
                            tem_riders += BundleCloseRider[ct_name]
                        except:
                            print(ct_name)
                            print(list(BundleCloseRider.keys()))
                            #input('왜 없나?')
                    tem_riders = list(set(tem_riders))
                    o.exp_riders = tem_riders
                    # todo 1115: exp rider 계산 추가 End
                    platform.platform[task_index] = o
                    task_index += 1
                    #input('번들 추가됨')
        if interval > 0:
            yield env.timeout(interval)
        else:
            print('현재 T :{} / 마지막 고객 {} 생성'.format(int(env.now), name))
            pass
        if env.now > end_t - 5*(1.0/lamda) and save_txt == False:
            f = open('E:/python_백업/py_charm/BundleSimple/log_dynamic.txt','a')
            vals = []
            for key in t_info:
                vals += t_info[key]
            f.write('CountActiveRider;t;DynamicBundleConstruct;t;SubF1_SingleCustomerBundleInsert;t;B2;t;B3;t; \n')
            f.write('{};{};{};{};{};{};{};{};{};{}; \n'.format(vals[0],vals[1],vals[2],vals[3],vals[4],vals[5],vals[6],vals[7],vals[8],vals[9]))
            f.close()
            save_txt = True