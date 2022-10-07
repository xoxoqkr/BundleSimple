# -*- coding: utf-8 -*-

#from scipy.stats import poisson
import time
import operator
import itertools
from numba import jit
from A1_BasicFunc import distance, ActiveRiderCalculator, t_counter, counter2
from A2_Func import BundleConsist, BundleConsist2
import math
import numpy as np
import pandas as pd




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
        print('rider name ::',str(rider.name))
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
                print('라이더 {} 마지막 위치 {} 마지막 시간 {} 다음 탐색 시간 {}'.format(rider_name, dists[-1], times[-1], rider.next_search_time2))
            else:
                print('False')
        else:
            print('False2')
    if point_return == True:
        return names, dists, times
    else:
        return names


def BundleConsideredCustomers(target_order, platform, riders, customers, speed = 1, bundle_search_variant = True, d_thres_option = True, max_d_infos = [], stopping = 0):
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
                d_thres = 100
            else:
                d_thres = customer.p2
            dist = distance(target_order.store_loc[0],target_order.store_loc[1], customer.store_loc[0],customer.store_loc[1]) / speed
            dist2 = distance(target_order.location[0],target_order.location[1], customer.location[0],customer.location[1]) / speed
            #if target_order.name != customer.name and dist <= d_thres :
            in_max_d = False
            for d_info in max_d_infos:
                dist3 = distance(target_order.store_loc[0],target_order.store_loc[1], d_info[1][0],d_info[1][1])/ speed
                if dist3 < d_info[2]:
                    in_max_d = True
                    break
            if len(max_d_infos) == 0:
                in_max_d = True
            if target_order.name != customer.name and dist <= d_thres*store_para and dist2 <= d_thres*loc_para and in_max_d == True:
            #if target_order.name != customer.name and dist <= store_min and dist2 <= loc_min and in_max_d == True:
                not_served_ct_names.append(customer_name)
                not_served_ct_name_cls[customer_name] = customer
                not_served_ct_names_infos.append([customer_name, dist, dist2])
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
    if stopping > 0:
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
    else:
        rev_not_served_ct_names = not_served_ct_names
    for ct_name in rev_not_served_ct_names:
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
        if customer_name != target_order.name and orders[customer_name].time_info[1] == None and orders[customer_name].cancel == False:
            d.append(customer_name)
    #print(d,s)
    #input("확인2")
    new = 0
    M2_count = 0
    if len(d) > s - 1:
        M = itertools.permutations(d, s - 1)
        b = []
        if print_option == True:
            print('대상 고객 {} ::고려 고객들 {}'.format(target_order.name, d))
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


#@jit(nopython=True)
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
                             platform_exp_error = 1,  thres_label = 1, label_check = None, feasible_return = True, fix_start = True, cut_info = [2500,2500]):
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
        #if label >= thres_label:
            #print('라벨',label)
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