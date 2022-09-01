# -*- coding: utf-8 -*-
import numpy as np

import A1_BasicFunc as Basic
from scipy.stats import poisson
import simpy
import operator
import itertools
import random
import time
import numpy
import matplotlib.pyplot as plt

class Order(object):
    def __init__(self, order_name, customer_names, route, order_type, fee = 0, parameter_info = [0,0,0]):
        self.index = order_name
        self.customers = customer_names
        self.route = route
        self.picked = False
        self.type = order_type #1:단주문, 2:B2, 3:B3
        self.average_ftd = None
        self.fee = fee
        self.parameter_info = parameter_info
        self.old_info = None
        self.gen_t = 0


class Rider(object):
    def __init__(self, env, i, platform, customers, stores, start_time = 0, speed = 1, capacity = 3, end_t = 120, p2= 2, bound = 5, freedom = True,
                 order_select_type = 'simple', wait_para = False, uncertainty = False, exp_error = 1, platform_recommend = False, p_ij = [0.5,0.3,0.2],
                 bundle_construct = False, lamda = 5, ite = 1):
        self.name = i
        self.env = env
        self.gen_time = int(env.now)
        self.resource = simpy.Resource(env, capacity=1)
        self.visited_route = [[-1, -1, [25, 25], int(env.now)]]
        self.speed = speed
        self.route = []
        self.bundles_infos = []
        self.run_process = None
        self.capacity = capacity
        self.onhand = []
        self.picked_orders = []
        self.end_t = env.now + end_t
        self.last_departure_loc = [25, 25]
        self.container = []
        self.served = []
        self.p2 = p2
        self.start_time = start_time
        self.max_order_num = capacity #max_order_num
        self.bound = bound
        self.idle_time = 0
        self.candidates = []
        self.b_select = 0
        self.income = 0
        self.wait_para = wait_para
        self.store_wait = 0
        self.num_bundle_customer = 0
        self.bundle_store_wait = [] # 번들에 속한 주문들에 의해 발생한 대기 시간
        self.single_store_wait = [] # 일반 주문에 의해 발생한 대기 시간
        self.onhand_order_indexs = []
        self.decision_moment = []
        self.exp_error = exp_error
        self.search_lamda = lamda # random.randint(4,7)
        self.exp_wage = 0
        self.freedom = freedom
        self.order_select_type = order_select_type
        self.uncertainty = uncertainty
        self.next_select_t = int(env.now)
        self.next_search_time = 0 #다음에 주문을 선택할 시점
        self.last_pick_time = 0 #저번에 주문을 선택한 시점
        self.platform_recommend = platform_recommend
        self.check_t = 0.5
        self.empty_serach_count = 0
        self.p_j = p_ij
        self.bundle_construct = bundle_construct
        self.order_select_time = []
        self.pages_history = []
        self.Rand = numpy.random.RandomState(seed=i + ite)
        self.bundle_count = []
        self.snapshots = []
        self.next_search_time2 = env.now
        self.onhand_bundle = []
        self.rider_select_print_fig = False
        self.rider_wait2 = []
        env.process(self.RunProcess(env, platform, customers, stores, self.p2, freedom= freedom, order_select_type = order_select_type, uncertainty = uncertainty))
        env.process(self.TaskSearch(env, platform, customers, p2=self.p2, order_select_type=order_select_type, uncertainty=uncertainty))

    def RiderMoving(self, env, time, info = '가게'):
        """
        라이더가 움직이는 시간의 env.time의 generator를 반환
        :param env: simpy.env
        :param time: 라이더가 움직이는 시간
        """
        yield env.timeout(time)
        print('현재1 T:{} 라이더{} {} 도착'.format(int(env.now),self.name,info))


    def RunProcess(self, env, platform, customers, stores,p2 = 0, wait_time = 2, freedom = True, order_select_type = 'simple', uncertainty = False):
        """
        라이더의 행동 과정을 정의.
        1)주문 선택
        2)선택할만 한 주문이 없는 경우 대기(wait time)
        @param env:
        @param platform:
        @param customers:
        @param stores:
        @param p2:
        @param wait_time:
        """
        while int(env.now) < self.end_t:
            if len(self.route) > 0:
                node_info = self.route[0]
                #print('T: {} 라이더 :{} 노드 정보 {} 경로 {}'.format(int(env.now),self.name, node_info,self.route))
                order = customers[node_info[0]]
                store_name = order.store
                move_t = Basic.distance(self.last_departure_loc, node_info[2], rider_count= True) / self.speed
                self.next_search_time = env.now + move_t
                print('라이더 {}/ 현재 시간 {} /다음 선택 시간 {}/ OnHandOrder{}/ 최대 주문 수{}'.format(self.name, env.now, self.next_search_time, len(self.onhand), self.capacity))
                if len(self.route) == 1:
                    exp_idle_t = round(env.now + move_t + order.service_time,1) + 0.1
                    self.next_search_time = min(exp_idle_t, self.next_search_time)
                with self.resource.request() as req:
                    print('T: {} 노드 {} 시작'.format(int(env.now), node_info))
                    yield req  # users에 들어간 이후에 작동
                    req.loc = node_info[2]
                    print('T: {} 라이더 : {} 노드 {} 이동 시작 예상 시간{}'.format(int(env.now), self.name, node_info, move_t))
                    if node_info[1] == 0: #가게인 경우
                        exp_store_arrive = env.now + move_t
                        move_t += order.time_info[6]
                        if order.type == 'single_order':
                            pool = numpy.random.normal(order.cook_info[1][0], order.cook_info[1][1] * self.exp_error, 1000)
                            order.rider_exp_cook_time = random.choice(pool)
                            exp_cook_time = order.rider_exp_cook_time
                        else:  # 'bundle'
                            exp_cook_time = order.platform_exp_cook_time
                        if exp_cook_time == None:
                            print(order.leave)
                            print(order.time_info)
                            input('멈춤')
                        wait_at_store, food_wait, manual_cook_time = WaitTimeCal1(exp_store_arrive, order.time_info[1], exp_cook_time, order.cook_time,move_t = move_t)
                        self.store_wait += wait_at_store
                        order.rider_wait = wait_at_store
                        order.food_wait = food_wait
                        if order.inbundle == True:
                            self.bundle_store_wait.append(wait_at_store)
                        else:
                            self.single_store_wait.append(wait_at_store)
                        cal_cook_time = max(0.001, (order.time_info[1] + order.actual_cook_time) - env.now)
                        expPickUpT = env.now + move_t
                        #yield env.process(stores[store_name].Cook(env, order, order.cook_info[0], manual_cook_time = cal_cook_time)) & env.process(self.RiderMoving(env, move_t))
                        yield order.cooking_process & env.process(self.RiderMoving(env, move_t))
                        if expPickUpT < env.now:
                            self.rider_wait2.append(env.now -expPickUpT)
                            order.rider_wait3 = env.now - expPickUpT
                        elif expPickUpT == env.now:
                            order.food_wait2.append(expPickUpT - env.now)
                            order.food_wait3 = env.now - (order.cook_start_time + order.actual_cook_time)
                        else: #완전 일치.ㅎㅎ
                            print('현재T{};음식 조리 시작{};음식 조리시간{};음식조리 완료 시간{} expPickUpT{}'.format(env.now,order.cook_start_time, order.actual_cook_time,order.cook_start_time + order.actual_cook_time,expPickUpT))
                            order.rider_wait3 = 0
                            order.food_wait3 = 0
                            input('확인')
                        print('T:{} 라이더{} 고객{}을 위해 가게 {} 도착'.format(int(env.now), self.name, customers[node_info[0]].name,customers[node_info[0]].store))
                        self.container.append(node_info[0])
                        print('라이더 {} 음식{} 적재'.format(self.name, node_info[0]))
                        order.time_info[2] = env.now
                        order.time_info[8] = exp_store_arrive
                        order.who_serve.append([self.name, int(env.now),0])
                    else:#고객인 경우
                        yield env.process(self.RiderMoving(env, move_t, info = '고객'))
                        print('T: {} 라이더 {} 고객 {} 도착 서비스 종료'.format(int(env.now),self.name, node_info[0]))
                        order.time_info[3] = env.now
                        real_flt = round(order.time_info[3] - order.time_info[2],4)
                        exp_flt = round(Basic.distance(order.location, order.store_loc)/self.speed,4)
                        yield env.timeout(order.time_info[7]) #todo: 고객의 시간 발생.
                        order.time_info[4] = env.now
                        if real_flt < exp_flt:
                            input('차이 {} ;고객 {};가게위치 {};고객 위치{};realFlt:{};expFlt:{}'.format(real_flt - exp_flt, order.name, order.store_loc, order.location, real_flt, exp_flt))
                        order.who_serve.append([self.name, int(env.now),1])
                        try:
                            self.container.remove(node_info[0])
                            self.onhand.remove(node_info[0])
                            self.served.append(node_info[0])
                            self.income += order.fee
                            if node_info in self.onhand_bundle:
                                self.onhand_bundle.remove(node_info[0])
                        except:
                            print('방문 경로 {}/ 현재 지점 {} / 노드 정보 {}'.format(self.visited_route, self.route[0], node_info))
                            input('라이더 {} / 현재 컨테이너::{}/들고 있는 주문::{}/대상 주문::{}'.format(self.name, self.container,self.onhand,node_info[0]))
                        #todo: order를 완료한 경우 order를 self.picked_orders에서 제거해야함.
                        for order_info in self.picked_orders:
                            done = True
                            for customer_name in order_info[1]:
                                if customer_name not in self.served:
                                    done = False
                                    break
                            if done == True:
                                self.picked_orders.remove(order_info)
                        if self.visited_route[-1][0] != node_info[0]: #이전 노드가 아니라는 뜻
                            check_indexs = list(range(len(self.visited_route)))
                            check_indexs.reverse()
                            for index in check_indexs:
                                if self.visited_route[index][0] == node_info[0] and self.visited_route[index][1] == 0:
                                    order.rider_bundle = [True, env.now - self.visited_route[index][3]]
                                    break
                            if order.rider_bundle == [None, None]:
                                print(node_info)
                                print(self.visited_route)
                                input('에러 발생')
                    self.last_departure_loc = self.route[0][2]
                    self.visited_route.append(self.route[0])
                    self.visited_route[-1][3] = int(env.now)
                    #input('방문 경로 확인 {}'.format(self.visited_route))
                    del self.route[0]
                    print('남은 경로 {}'.format(self.route))
            else:
                self.empty_serach_count += 1
                self.next_search_time2 = round(env.now,4) - 0.001
                self.OrderSelect_module(self.env, platform, customers, p2 = self.p2)
                yield env.timeout(self.check_t)


    def OrderSelect_module(self,env, platform, customers, p2=0, order_select_type='simple', uncertainty=False, score_type = 'simple'):
        if self.bundle_construct == True:  # FT/ TT
            # 주문 고르기 #번들 생성 가능하도록
            order_info, self_bundle = self.OrderSelect(platform, customers, p2=p2, uncertainty=uncertainty)
        elif len(self.onhand) == 0:  # FF/ TF
            # 플랫폼에 있는 주문만 고르기
            order_info, self_bundle = self.OrderSelect(platform, customers, p2=p2, uncertainty=uncertainty)
        else:
            order_info = None
        # 선택된 번들 반영 부분
        empty_t = 1000
        if order_info != None:
            platform.platform[order_info[0]].picked = True
            if self.rider_select_print_fig == True:
                input('주문 선택')
                selected_task = platform.platform[order_info[0]]
                x1 = []
                y1 = []
                x2 = []
                y2 = []
                # 1 현재 위치 그리기
                plt.scatter(self.last_departure_loc[0], self.last_departure_loc[1], marker='*', color='b',
                            label='current_loc')
                # 2 선택한 번들 위치 그리기
                # 3 확인
                for index in range(1, len(order_info[1])):
                    start = order_info[1][index - 1][2]
                    end = [order_info[1][index][2][0] - order_info[1][index - 1][2][0],
                           order_info[1][index][2][1] - order_info[1][index - 1][2][1]]
                    plt.arrow(start[0], start[1], end[0], end[1], width=0.2, length_includes_head=True)
                for ct_name in order_info[5]:
                    x1.append(customers[ct_name].store_loc[0])
                    y1.append(customers[ct_name].store_loc[1])
                    x2.append(customers[ct_name].location[0])
                    y2.append(customers[ct_name].location[1])
                # if bundle_infos != None:
                #    bundle_infos.append([len(o.customers),])
                plt.scatter(x1, y1, marker='o', color='k', label='store')
                plt.scatter(x2, y2, marker='x', color='m', label='customer')
                plt.legend()
                plt.axis([0, 50, 0, 50])
                title = 'RiderBundle {} ;Rider {};T {}'.format(self.bundle_construct, self.name, round(env.now, 2))
                plt.title(title)
                # plt.savefig(title + '.png', dpi=1000)
                plt.show()
                input('라이더 선택 확인2')
                plt.close()
                # target_customer, res_C_T, now_t = 0, titleinfo = 'None'
            # print('F:TaskSearch/라이더#:{}/order_info:{}'.format(self.name, order_info))
            self.OrderPick(env, order_info, order_info[1], customers, env.now, route_revise_option=self.bundle_construct,
                           self_bundle=self_bundle)  # 라우트 결정 후
            if order_info[8] == 'platform' and len(order_info[5]) > 1:
                self.b_select += 1
                self.num_bundle_customer += len(order_info[5])
            elif order_info[8] == 'rider':
                for customer_name in platform.platform[order_info[0]].customers:
                    customers[customer_name].rider_bundle_t = env.now
            Basic.UpdatePlatformByOrderSelection(platform, order_info[0])  # 만약 개별 주문 선택이 있다면, 해당 주문이 선택된 번들을 제거.


    def TaskSearch(self, env, platform, customers, p2=0, order_select_type='simple', uncertainty=False, score_type = 'simple'):
        while int(env.now) < self.end_t:
            onhand_slack = self.max_order_num - len(self.onhand)
            if env.now >= self.next_search_time2 and (onhand_slack > 0 or len(self.onhand_bundle) == 0):
                self.OrderSelect_module(self.env, platform, customers, p2=p2)
                """
                if self.bundle_construct == True: #FT/ TT
                    #주문 고르기 #번들 생성 가능하도록
                    order_info, self_bundle = self.OrderSelect(platform, customers, p2=p2, uncertainty=uncertainty)
                elif len(self.onhand) == 0: # FF/ TF
                    #플랫폼에 있는 주문만 고르기
                    order_info, self_bundle = self.OrderSelect(platform, customers, p2=p2, uncertainty=uncertainty)
                else:
                    order_info = None
                #선택된 번들 반영 부분
                empty_t = 1000
                if order_info != None:
                    platform.platform[order_info[0]].picked = True
                    if self.rider_select_print_fig == True:
                        input('주문 선택')
                        selected_task = platform.platform[order_info[0]]
                        x1 = []
                        y1 = []
                        x2 = []
                        y2 = []
                        # 1 현재 위치 그리기
                        plt.scatter(self.last_departure_loc[0], self.last_departure_loc[1], marker='*', color='b', label='current_loc')
                        #2 선택한 번들 위치 그리기
                        #3 확인
                        for index in range(1, len(order_info[1])):
                            start = order_info[1][index - 1][2]
                            end = [order_info[1][index][2][0] - order_info[1][index - 1][2][0],
                                   order_info[1][index][2][1] - order_info[1][index - 1][2][1]]
                            plt.arrow(start[0], start[1], end[0], end[1], width=0.2, length_includes_head=True)
                        for ct_name in order_info[5]:
                            x1.append(customers[ct_name].store_loc[0])
                            y1.append(customers[ct_name].store_loc[1])
                            x2.append(customers[ct_name].location[0])
                            y2.append(customers[ct_name].location[1])
                        # if bundle_infos != None:
                        #    bundle_infos.append([len(o.customers),])
                        plt.scatter(x1, y1, marker='o', color='k', label='store')
                        plt.scatter(x2, y2, marker='x', color='m', label='customer')
                        plt.legend()
                        plt.axis([0, 50, 0, 50])
                        title = 'RiderBundle {} ;Rider {};T {}'.format(self.bundle_construct,self.name, round(env.now, 2))
                        plt.title(title)
                        #plt.savefig(title + '.png', dpi=1000)
                        plt.show()
                        input('라이더 선택 확인2')
                        plt.close()
                        #target_customer, res_C_T, now_t = 0, titleinfo = 'None'
                    #print('F:TaskSearch/라이더#:{}/order_info:{}'.format(self.name, order_info))
                    self.OrderPick(order_info, order_info[1], customers, env.now,route_revise_option=self.bundle_construct, self_bundle = self_bundle)  # 라우트 결정 후
                    if order_info[8] == 'platform' and len(order_info[5]) > 1:
                        self.b_select += 1
                        self.num_bundle_customer += len(order_info[5])
                    Basic.UpdatePlatformByOrderSelection(platform,order_info[0])  # 만약 개별 주문 선택이 있다면, 해당 주문이 선택된 번들을 제거.
                """
                #종류에 따른 다음 선택 시간 결정 :todo 라이더의 효율에 영향을 미침.
                if self.bundle_construct == True:
                    pr = 1/(onhand_slack + 1)
                    if pr < random.random():
                        next = self.Rand.poisson(self.search_lamda)/(onhand_slack + 1)
                    else:
                        next = self.Rand.poisson(self.search_lamda)
                elif len(self.onhand) < self.max_order_num:
                    next = self.check_t #order_end_time
                else:
                    next = self.Rand.poisson(self.search_lamda)
                print('다음 시간 {}'.format(next))
                self.next_search_time2 += next
                self.order_select_time.append(env.now)
                yield env.timeout(next)
            else:
                yield env.timeout(self.check_t) #시간을 흘려 보내기 위한 장치


    def OrderSelect(self, platform, customers, p2, uncertainty, l = 4):
        #Step 1 : 고려 대상인 고객 계산
        scores = []
        bound_order_names = []
        bundle_task_names = []
        time_check = []
        for index in platform.platform:
            task = platform.platform[index]
            duplicated_para = False
            for name in task.customers:
                if len(customers[name].who_picked) > 0:
                    duplicated_para = True
            if duplicated_para == True: #Step 1-1 : 이미 다른 라이더가 선택한 고객의 경우 고려하지 않음
                continue
            elif task.picked == False and len(task.customers) + len(self.onhand) <= self.max_order_num: #Step 1-2 : 아직 선택되지 않은 task에 대해 라이더의 현재 위치와의 거리 계산
                #dist = Basic.distance(self.last_departure_loc,task.route[0][2]) / self.speed  # 자신의 현재 위치와 order의 시작점(가게) 사이의 거리.
                dist = Basic.distance(self.CurrentLoc(self.env.now),task.route[0][2],rider_count= True)  # 자신의 현재 위치와 order의 시작점(가게) 사이의 거리.
                if len(task.customers) > 1:
                    bundle_task_names.append([task.index, dist])
                bound_order_names.append([task.index, dist])
            else: #Step 1-3 : 에러 출력
                print('F:OrderSelect-E1/task{}/선택 정보 {}/ 주문 채택시 onhand주문수:{} > {}:maxonhand'.format(task.index, task.picked, len(task.customers) + len(self.onhand) ,self.max_order_num))

        bound_order_names.sort(key=operator.itemgetter(1))
        #Step 2 : 라이더가 확인할 페이지를 결정
        rv = float(self.Rand.random(size=1)) #Step 2 - 1 : 확인할 페이지 확률 변수 rv 생성
        page = 1
        pages = list(range(len(self.p_j)))
        for index in pages:
            if rv < sum(self.p_j[:index+1]):
                page = index + 1
                self.pages_history.append(rv)
                break
        considered_tasks = bound_order_names[:page*l] #Step 2 - 2 : 라이더가 확인할 주문 목록
        nearest_bundle = 0 #Step 2 - 3 : 정보 저장 부분
        out_count = 0
        for outer_task in bound_order_names[page*l:]:
            task = platform.platform[outer_task[0]]
            if len(task.customers) > 1:
                nearest_bundle = page*l + out_count
                print('nearest_bundle:',nearest_bundle)
                break
            out_count += 1
        #Step 3 : 라이더가 각 task의 점수를 계산
        for task_info in considered_tasks:
            tem_time_check = []
            rider_bundle = False
            #3-1: FF나 TF인 경우
            task = platform.platform[task_info[0]]
            print('task 정보/ 고객 {}/ 현재 경로 {}'.format(task.customers, self.route))
            mv_time = 0
            if len(self.route) > 0:
                rev_route = [self.route[-1]]
                print('경로 존재 {}'.format(rev_route))
            else:
                rev_route = [self.visited_route[-1]]
                print('경로 X :{}'.format(rev_route))
            rev_route += task.route
            for node_index in range(1, len(rev_route)):
                print('bf {} -> {} af/ T:{}'.format(rev_route[node_index - 1][2], rev_route[node_index][2],Basic.distance(rev_route[node_index - 1][2], rev_route[node_index][2],rider_count= True) / self.speed))
                mv_time += Basic.distance(rev_route[node_index - 1][2], rev_route[node_index][2],rider_count= True) / self.speed
                if rev_route[node_index][1] == 0:
                    mv_time += customers[rev_route[node_index][0]].time_info[6]
                    tem_time_check.append(round(self.env.now - customers[rev_route[node_index][0]].time_info[0],4))
                else:
                    try:
                        mv_time += customers[rev_route[node_index][0]].time_info[7]
                    except:
                        pass
            WagePerMin = round(task.fee / mv_time, 4)  # 분당 이익
            if type(task.route) == tuple:
                task.route = list(task.route)
            #if len(task.route) > 2:
            #    WagePerMin += 10000
            scores.append([task.index,task.route, None, None, None, task.customers, None,WagePerMin,'platform'])
            print('시간 정보1  {} '.format(mv_time))
            print('시간 계산 경로1 {}'.format(rev_route))
            print('단건:: 라이더#{}/경로 {} /리스트 확인 {}'.format(self.name, self.route, scores[-1]))
            # 3-2: TT나 FT 인 경우
            if self.bundle_construct == True and len(task.customers) + len(self.onhand) <= self.max_order_num: # 3-2-1: 선택 주문TT나 FT인 경우
                best_route_info = self.ShortestRoute2(task, customers, p2=p2,uncertainty=uncertainty)  # task가 산입될 수 있는 가장 좋은 경로
                # best_route_info = [rev_route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time]
                if len(best_route_info) > 0:
                    if len(self.route) > 0:
                        bf = self.route[-1][2]
                        af = best_route_info[0][0][2]
                        org_route_t = Basic.distance(bf, af,rider_count= True)/ self.speed
                    else:
                        org_route_t = Basic.distance(self.visited_route[-1][2],best_route_info[0][0][2],rider_count= True)/ self.speed
                        print('af {} -> {} bf'.format(self.visited_route[-1][2],best_route_info[0][0][2]))
                    if len(best_route_info) < 5:
                        input('best_route_info {} '.format(best_route_info))
                    print('경로 정보 2{}'.format(best_route_info))
                    print('시간 정보2  {} : {} : {}'.format(best_route_info[5], org_route_t, best_route_info[5]+ org_route_t))

                    benefit = round(task.fee / (best_route_info[5] +org_route_t),4)  # 이익 / 운행 시간
                    benefit2 = round(task.fee / best_route_info[6],4)
                    print('inc_t1:{}/inc_t2:{}'.format(best_route_info[5] +org_route_t, best_route_info[6]))
                    scores.append([task.index] + best_route_info[:6] + [benefit2] + ['rider'])
                    if WagePerMin < benefit2:
                        rider_bundle = True
                    print('번들:: 라이더#{}/경로 {} /리스트 확인 {}'.format(self.name, self.route, scores[-1]))
            if sum(tem_time_check) > 0:
                time_check.append([scores[-1][7]] + [customers[rev_route[node_index][0]].store_loc]+ [round(numpy.mean(tem_time_check),4)] + [tem_time_check]+ [rev_route[0][2]] )
                if len(tem_time_check) > 1:
                    time_check[-1].insert(1,'B')
                elif rider_bundle == True:
                    time_check[-1].insert(1, 'R')
                else:
                    time_check[-1].insert(1, 'S')
                #input('확인{}'.format(time_check[-1]))
            else:
                time_check.append([scores[-1][7]] + ['N',[-1,-1],0] + [tem_time_check,[-1,-1]])
        scores.sort(key=operator.itemgetter(7), reverse=True)
        time_check.sort(key=operator.itemgetter(0), reverse=True)
        ave_wait = []
        #print(time_check)
        #input('timecheck')
        for info in time_check:
            try:
                ave_wait += info[3]
            except:
                print(info)
                #input('timecheck')
                pass
        if sum(ave_wait) > 0:
            print('T;{};라이더;{};현위치;{};평균대기;{};선택;{}정보;{}'.format(int(self.env.now), self.name,time_check[0][5] ,round(numpy.mean(ave_wait),4), time_check[0][3] ,time_check))
        else:
            #input('T;{};라이더;{};평균대기;{};선택;{};정보;{}'.format(int(self.env.now), self.name, 0, None, time_check))
            pass
        # Step 4 : task 선택
        if len(scores) > 0:
            snapshot_info = self.SnapShotSaver(scores, page,l)
            nearest_bundle_page = int(nearest_bundle / l) + 1
            self.snapshots.append(snapshot_info + [nearest_bundle, nearest_bundle_page])
            #[라이더 이름, 시간, 확인 페이지, 번들 수, 번들 최대 가치, 단건 주문 최대 가치, 가장 가까운 주문 정렬 순서, 가장 가까운 주문 정렬 순서 페이지]
            print('스냅샷 {}'.format(self.snapshots[-1]))
            print('T{} /라이더 {}/확인 페이지{}/번들 수{}/번들 최대 가치{}/단건 주문 최대 가치{}/타입{}'.format(round(self.env.now,4), self.snapshots[-1][0],
                                                                                self.snapshots[-1][2],self.snapshots[-1][4],self.snapshots[-1][5],
                                                                                self.snapshots[-1][6],self.snapshots[-1][7]))
            print('가장 가까운 번들{}/번들 페이지:{}'.format(self.snapshots[-1][8],self.snapshots[-1][9]))
            print('현재 경로 {}'.format(self.route))
            print('점수들 {} '.format(scores[:3]))
            bundle_print = []
            for score in scores:
                if len(score[1]) > 2:
                    bundle_print.append(score)
            print('번들 점수들 {}'.format(bundle_print))
            return scores[0], scores[0][8]
        else:
            self.snapshots.append([round(self.env.now,4),None])
            return None, None


    def ShortestRoute2(self, order, customers, now_t = 0, p2 = 0, M = 1000, uncertainty = False):
        """
        order를 수행할 수 있는 가장 짧은 경로를 계산 후, 해당 경로의 feasible 여/부를 계산
        반환 값 [경로, 최대 FLT, 평균 FLT, 최소FLT, 경로 내 고객 이름, 경로 운행 시간]
        *Note : 선택하는 주문에 추가적인 조건이 걸리는 경우 feasiblity에 추가적인 조건을 삽입할 수 있음.
        @param order: task
        @param customers: 발생한 고객들 {[KY]customer name : [Value]class customer, ...}
        @param now_t: 현재 시간
        @param p2: 허용 Food Lead Time의 최대 값
        @param M: 가게와 고객을 구분하는 임의의 큰 수
        @return: 최단 경로 정보 -> [경로, 최대 FLT, 평균 FLT, 최소FLT, 경로 내 고객 이름, 경로 운행 시간]
        """
        order_names = []  # 가게 이름?
        store_names = []
        order_customers = []
        heading_node = []
        already_served_customer_names = []
        # 1: 기존 주문 주문 추가
        if len(self.route) > 0:
            if self.route[0][1] == 1:
                heading_node = [self.route[0][0]]
                already_served_customer_names = [self.route[0]]
            else:
                heading_node = [self.route[0][0] + M]
                already_served_customer_names = []
        prior_route = []
        for node_info in self.route[1:]: #현재 향하고 있는 노드는 도착해야함.
            if node_info[1] == 1:
                order_names.append(node_info[0])
                order_customers.append(customers[node_info[0]])
                prior_route.append(node_info[0])
            else:
                store_names.append(node_info[0] + M)
                prior_route.append(node_info[0] + M)
        org_route_time =  Basic.RouteTime(order_customers, prior_route, speed=self.speed, M=M, uncertainty=uncertainty, error = self.exp_error, now_t = self.env.now)
        #2:새로운 주문 추가
        for customer_name in order.customers:
            order_names.append(customer_name)
            store_names.append(customer_name + M)
            order_customers.append(customers[customer_name])
        candi = order_names + store_names
        subset = itertools.permutations(candi, len(candi))  # todo: permutations 사용으로 연산량 부하 지점
        feasible_subset = []
        #3 : 가능 여부 계산
        for route_part in subset:
            route = heading_node + list(route_part)
            sequence_feasiblity = True
            feasible_routes = []
            for order_name in order_names:  # order_name + M : store name ;
                if order_name + M in route:
                    if route.index(order_name + M) < route.index(order_name):
                        pass
                    else:
                        sequence_feasiblity = False
                        break
            if sequence_feasiblity == True:
                ftd_feasiblity, ftds = Basic.FLT_Calculate(order_customers, customers, route, p2,
                                                           except_names=already_served_customer_names, M=M,
                                                           speed=self.speed, now_t=now_t, uncertainty=uncertainty,
                                                           exp_error=self.exp_error)
                print('가능?{}/ 초과시간{}'.format(ftd_feasiblity, ftds))
                if ftd_feasiblity == True:
                    # print('ftds',ftds)
                    # input('멈춤5')
                    #route_time = Basic.RouteTime(order_customers, route, speed=speed, M=M)
                    print('시간 계산 경로 2 {}'.format(route_part))
                    route_time = Basic.RouteTime(order_customers, list(route_part), speed=self.speed, M=M, uncertainty=uncertainty, error = self.exp_error, now_t = self.env.now)
                    #feasible_routes.append([route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time])
                    #route_time = Basic.RouteTime(order_customers, list(route_part), speed=speed, M=M)
                    rev_route = []
                    for node in route:
                        if node < M:
                            name = node
                            info = [name, 1, customers[name].location, 0]
                        else:
                            name = node - M
                            info = [name, 0, customers[name].store_loc, 0]
                        rev_route.append(info)
                    try:
                        feasible_routes.append([rev_route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time, route_time - org_route_time])
                    except:
                        #input('대상 경로 {} 고객들 {} '.format(rev_route, order_names))
                        print('대상 경로 {} 고객들 {} '.format(rev_route, order_names))

                    #input('기존 경로 중 {} 제외 경로 {} -> 추가될 경로 {}'.format(route,prior_route,rev_route))
            if len(feasible_routes) > 0:
                feasible_routes.sort(key=operator.itemgetter(5)) #가장 짧은 거리의 경로 선택.
                feasible_subset.append(feasible_routes[0])
        if len(feasible_subset) > 0:
            feasible_subset.sort(key=operator.itemgetter(5))
            #print('선택 된 정보 {} / 경로 길이 {}'.format(feasible_subset[0][0], feasible_subset[0][5]))
            #input('확인9')
            return feasible_subset[0]
        else:
            return []


    def OrderPick(self, env, order_info, route, customers, now_t, route_revise_option = 'simple', self_bundle = None):
        """
        수행한 order에 대한 경로를 차량 경로self.route에 반영하고, onhand에 해당 주문을 추가.
        @param order: class order
        @param route: 수정될 경로
        @param customers: 발생한 고객들 {[KY]customer name : [Value]class customer, ...}
        @param now_t: 현재 시간
        """
        names = order_info[5]
        if len(names) > 1:
            self.bundles_infos.append(route)
            self.bundle_count.append(len(names))
            self.onhand_bundle = names
        for name in names:
            customer = customers[name]
            customer.time_info[1] = now_t
            customer.who_picked.append([self.name, now_t,self_bundle,'single'])
            if 0< customer.dp_cook_time < 15:
                customer.cooking_process = env.process(customer.CookingFirst(env, customer.actual_cook_time))
            if len(names) > 1:
                customer.inbundle = True
                customer.type = 'bundle'
                customer.who_picked[-1][3] = 'bundle'
            #print('주문 {}의 고객 {} 가게 위치{} 고객 위치{}'.format(order.index, name, customers[name].store_loc, customers[name].location))
        #print('선택된 주문의 고객들 {} / 추가 경로{}'.format(names, route))
        if route[0][1] != 0:
            #input('삽입 경로에 문제 발생:: 삽입경로 {}'.format(route))
            pass
        if route_revise_option == 'simple': # 기존 경로 종료 지점에서 가장 가까운 고객을 선택하기 때문에, 현재 경로에 새롭게 추가되는 고객을 추가해야 함.
            self.route += route
        elif order_info[8] == 'platform':
            self.route += route
        elif order_info[8] == 'rider':# Shortest path를 이용해 고객을 선택하기 때문에, 갱신된 경로를 현재 경로에 새롭게 추가되는 고객을 추가해야 함.
            self.route = route
        else:
            input('ERROR {}'.format(order_info))
        self.onhand += names
        self.picked_orders.append([order_info[0], names])
        print('라이더 {} 수정후 경로 {}/ 보유 고객 {}/ 추가된 고객 {}'.format(self.name, self.route, self.onhand, names))


    def SnapShotSaver(self, infos, page, l=4):
        # best_route_info = [task.index, rev_route, max(ftds), sum(ftds) / len(ftds), min(ftds), order_names, route_time, benefit]
        now_t = round(self.env.now,4)
        bundle_exist = 0
        single_values = [0]
        bundle_values = [0]
        for info in infos:
            if len(info[5]) > 1:
                bundle_values.append(info[7])
                bundle_exist += 1
            else:
                single_values.append(info[7])
        max_bundle_value = max(bundle_values)
        max_single_value = max(single_values)
        if max_bundle_value == 0:
            type = 1 #번들이 존재X
        elif max_bundle_value > max_single_value:
            type = 2 #번들 존재 & 가치 충분
        else:
            type = 3 #번들 존재 & 가치 부족
        res = [self.name, now_t, page,l, len(bundle_values) - 1, max_bundle_value, max_single_value, type]
        #[라이더 이름, 시간, 확인 페이지, 페이지당 주문수, 페이지 내 번들 수, 번들 최대 가치, 단건 주문 최대 가치, type]
        return res


    def CurrentLoc(self, t_now):
        """
        현재의 위치를 물어 보는 함수.
        @return:
        """
        nodeA = self.last_departure_loc
        try:
            nodeB = self.resource.users[0].loc
        except:
            #print(' T {} 출발 위치 에러 ; 마지막 노드 {}'.format(t_now, self.last_departure_loc))
            nodeB = self.last_departure_loc
        if nodeA == nodeB:
            return nodeA
        else:
            t = t_now - self.visited_route[-1][3] # nodeA출발 후 경과한 시간.
            ratio = t / Basic.distance(nodeA, nodeB)
            x_inc = (nodeB[0] - nodeA[0])*ratio
            y_inc = (nodeB[1] - nodeA[1])*ratio
            return [nodeA[0] + x_inc, nodeA[1] + y_inc]


    def select_pr(self, t):
        # t분 이내에 주문을 선택을 할 확률을 반환
        mu = (1.0/self.search_lamda)*t
        x = 0
        return round(1 - poisson.pmf(x, mu),4) # 주문을 1번이상 수행할 확률


class Store(object):
    """
    Store can received the order.
    Store has capacity. The order exceed the capacity must be wait.
    """
    def __init__(self, env, platform, name, loc = (25,25), order_ready_time = 7, capacity = 6, slack = 2, print_para = True):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.location = loc
        self.order_ready_time = order_ready_time
        self.resource = simpy.Resource(env, capacity = capacity)
        self.slack = slack #자신의 조리 중이 queue가 꽉 차더라도, 추가로 주문을 넣을 수 있는 주문의 수
        self.received_orders = []
        self.wait_orders = []
        self.ready_order = []
        self.loaded_order = []
        self.capacity = capacity
        self.FRT = [0]
        env.process(self.StoreRunner(env, platform, capacity = capacity, print_para= print_para))


    def StoreRunner(self, env, platform, capacity, open_time = 1, close_time = 900, print_para = True):
        """
        Store order cooking process
        :param env: simpy Env
        :param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
        :param capacity: 발생한 고객들 {[KY]customer name : [Value]class customer, ...}
        :param open_time: store open time
        :param close_time: store close time
        """
        #input('가게 주문 채택')
        #yield env.timeout(open_time)
        now_time = round(env.now, 1)
        #input('가게 주문 채택0')
        while now_time < close_time:
            now_time = round(env.now,1)
            #받은 주문을 플랫폼에 올리기
            #print('값 확인',len(self.resource.users) , len(self.wait_orders), capacity , self.slack,len(self.received_orders))
            if len(self.resource.users) + len(self.resource.put_queue) < capacity + self.slack:  # 플랫폼에 자신이 생각하는 여유 만큼을 게시
            #if len(self.resource.users) + len(self.wait_orders) < capacity + self.slack: #플랫폼에 자신이 생각하는 여유 만큼을 게시
                #self.received_orders.append()
                #print('접수된 고객 수', len(self.received_orders))
                #input('가게 주문 채택1')
                #slack = min(capacity + self.slack - len(self.resource.users), len(self.received_orders))
                slack = capacity + self.slack - len(self.resource.users)
                #print('가게:',self.name,'/ 잔여 용량:', slack,'/대기 중 고객 수:',len(self.received_orders))
                received_orders_num = len(self.received_orders)
                platform_exist_order = []
                for index in platform.platform:
                    try:
                        platform_exist_order += platform.platform[index].customers
                    except:
                        print(' 에러 확인 용', platform.platform, index,platform.platform[index].customers)
                #print('플랫폼에 있는 주문 {}'.format(platform_exist_order))
                if received_orders_num > 0:
                    for count in range(min(slack,received_orders_num)):
                        order = self.received_orders[0] #앞에서 부터 플랫폼에 주문 올리기
                        route = [order.name, 0, order.store_loc, 0], [order.name, 1, order.location,0]
                        if len(list(platform.platform.keys())) > 0:
                            order_index = max(list(platform.platform.keys())) + 1
                        else:
                            order_index = 1
                        o = Order(order_index, [order.name],route,'single', fee = order.fee)
                        #print('주문 정보',o.index, o.customers, o.route, o.type)
                        if o.customers[0] not in platform_exist_order:
                            #platform[order_index] = o
                            platform.platform[order_index] = o
                            print('T : {} 가게 {} 고객 {} 주문 인덱스 {}에 추가'.format(env.now, self.name, o.customers, o.index))
                            print('플랫폼 ID{}'.format(id(platform)))
                        #platform.append(o)
                        #print('T : {} 가게 {} 고객 {} 주문 인덱스 {}에 추가'.format(env.now, self.name, o.customers, o.index))
                        if print_para == True:
                            print('현재T:', int(env.now), '/가게', self.name, '/주문', order.name, '플랫폼에 접수/조리대 여유:',capacity - len(self.resource.users),'/조리 중',len(self.resource.users))
                        self.wait_orders.append(order)
                        self.received_orders.remove(order)
                        #input('가게 주문 채택2')
            else: #이미 가게의 능력 최대로 조리 중. 잠시 주문을 막는다(block)
                #print("가게", self.name, '/',"여유 X", len(self.resource.users),'/주문대기중',len(self.received_orders))
                pass
            #만약 현재 조리 큐가 꽉차는 경우에는 주문을 더이상 처리하지 X
            yield env.timeout(0.1)
        #print("T",int(env.now),"접수 된 주문", self.received_orders)


    def Cook(self, env, customer, cooking_time_type = 'fixed', manual_cook_time = None):
        """
        Occupy the store capacity and cook the order
        :param env: simpy Env
        :param customer: class customer
        :param cooking_time_type: option
        """
        #print('현재 사용중', len(self.resource.users))
        with self.resource.request() as req:
            yield req #resource를 점유 해야 함.
            now_time = round(env.now , 1)
            req.info = [customer.name, now_time]
            if cooking_time_type == 'fixed':
                cooking_time = self.order_ready_time
            elif cooking_time_type == 'random':
                cooking_time = random.randrange(1,self.order_ready_time)
            elif cooking_time_type == 'uncertainty':
                cooking_time = random.choice(self.FRT)
            else:
                cooking_time = 0.001
            #customer.actual_cook_time = cooking_time
            print('T :{} 가게 {}, {} 분 후 주문 {} 조리 완료'.format(int(env.now),self.name,cooking_time,customer.name))
            if manual_cook_time == None:
                yield env.timeout(cooking_time)
            else:
                yield env.timeout(manual_cook_time)
            #print(self.resource.users)
            print('T :{} 가게 {} 주문 {} 완료'.format(int(env.now),self.name,customer.name))
            customer.food_ready = True
            customer.ready_time = env.now
            self.ready_order.append(customer)


class Customer(object):
    def __init__(self, env, name, input_location, store = 0, store_loc = (25, 25),end_time = 60, ready_time=0, service_time=0,
                 fee = 2500, p2 = 15, cooking_time = (2,5), cook_info = (None, None), platform = None, unit_fee = 110, fee_type = 'linear'):
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.time_info = [round(env.now, 2), None, None, None, None, end_time, ready_time, service_time, None]
        # [0 :발생시간, 1: 차량에 할당 시간, 2:차량에 실린 시간, 3:목적지 도착 시간,
        # 4:고객이 받은 시간, 5: 보장 배송 시간, 6:가게 출발 시간),7: 고객에게 서비스 하는 시간, 8: 가게 도착 시간]
        self.location = input_location
        self.store_loc = store_loc
        self.store = store
        self.type = 'single_order'
        self.min_FLT = p2 #Basic.distance(input_location, store_loc) #todo: 고객이 기대하는 FLT 시간.
        self.fee = fee + 110*Basic.distance(input_location, store_loc) #원래 150원
        if fee_type == 'linear':
            self.fee = fee + unit_fee * max(0 , Basic.distance(input_location, store_loc)-10)  # 원래 150원
        else:
            self.fee = fee + unit_fee * Basic.distance(input_location, store_loc)
        self.ready_time = None #가게에서 음식이 조리 완료된 시점
        self.who_serve = []
        self.distance = Basic.distance(input_location, store_loc)
        self.p2 = p2
        self.cook_time = cooking_time
        self.inbundle = False
        self.rider_wait = 0
        self.in_bundle_time = None
        self.cook_info = cook_info
        self.exp_info = [None,None,None]
        self.rider_exp_cook_time = None
        self.platform_exp_cook_time = 1
        self.food_wait = None
        self.service_time = service_time
        self.priority_weight = 1
        self.cancel = False
        self.rider_bundle = [None, None]
        self.who_picked = []
        self.in_bundle_t = 0
        self.rider_bundle_t = 0
        self.cook_start_time = 0
        self.cook_finish_time = 0
        self.actual_cook_time = 0
        self.dp_cook_time = 0
        self.food_wait2 = []
        env.process(self.CustomerLeave(env, platform))
        self.cooking_process = None
        self.rider_wait3 = None
        self.food_wait3 = None


    def CustomerLeave(self, env, platform):
        yield env.timeout(self.time_info[5])
        if self.time_info[1] == None:
            delete_list = []
            for task_index in platform.platform:
                if self.name in platform.platform[task_index].customers:
                    delete_list.append(task_index)
            for delete_index in delete_list:
                del platform.platform[delete_index]
            self.cancel = True
        else:
            pass


    def CookingFirst(self,env,time):
        print('음식 {} 선 조리 시작/ T{}'.format(self.name,int(env.now)))
        self.cook_start_time = env.now
        yield env.timeout(time)
        self.cook_finish_time = env.now
        print('음식 {} 조리 완료/ T{}'.format(self.name, int(env.now)))
        #input('조리 경과 확인')


class Platform_pool(object):
    def __init__(self):
        self.platform = {}
        self.info = []
        self.p = 1


class scenario(object):
    def __init__(self, name, p1 = True, search_option= False,  scoring_type = 'myopic',  unserved_bundle_order_break = True, bundle_selection_type = 'greedy', considered_customer_type = 'new'):
        self.name = name
        self.platform_work = p1
        self.res = []
        self.bundle_search_option = search_option
        self.store_dir = None
        self.customer_dir = None
        self.rider_dir = None
        self.scoring_type = scoring_type
        self.unserved_order_break = unserved_bundle_order_break# True면 기존에 있는 번들 고객도 고려, False면 번들에 없는 고객만 고려
        self.bundle_selection_type = bundle_selection_type
        self.considered_customer_type = considered_customer_type
        self.platform_recommend = False
        self.rider_bundle_construct = False
        self.obj_type = 'simple_max_s'
        self.snapshots = []
        self.search_type = 'heuristic'
        self.durations = []
        self.bundle_snapshots = {'size': [],'length':[],'od':[]}
        self.mix_ratio = None
        self.countf = [0,0,0,0,0]
        self.countt = [0, 0, 0, 0, 0]

def WaitTimeCal1(exp_store_arrive_t, assign_t, exp_cook_time, cook_time, move_t = 0):
    exp_food_ready_t = assign_t + exp_cook_time
    actual_food_ready_time = assign_t + cook_time
    if exp_store_arrive_t > exp_food_ready_t: #조리 시간에 여유가 있다면, 미이 주문을 실시
        #wait_at_store = 0
        food_wait = exp_store_arrive_t - exp_food_ready_t
        manual_cook_time = 0.001  # 거의 존재하지 않는다는 의미.
        if actual_food_ready_time > exp_store_arrive_t:
            wait_at_store = actual_food_ready_time - exp_store_arrive_t
            food_wait = 0
            manual_cook_time = wait_at_store + move_t
        elif exp_store_arrive_t >= actual_food_ready_time >= exp_food_ready_t:
            wait_at_store =0
            food_wait = exp_store_arrive_t - actual_food_ready_time
            manual_cook_time = 0.001
        elif exp_food_ready_t >= actual_food_ready_time:
            wait_at_store = 0
            food_wait = exp_store_arrive_t - actual_food_ready_time
            manual_cook_time = 0.001
        else:
            input('시간 관계 문제 발생1 ::actual_food_ready_time {}:: exp_store_arrive_t {} :: exp_food_ready_t{} '.format(actual_food_ready_time, exp_store_arrive_t, exp_food_ready_t))
    else:#조리 시간에 여유가 없다면, 가게에서 대기
        wait_at_store = exp_food_ready_t - exp_store_arrive_t
        food_wait = 0
        manual_cook_time = move_t + wait_at_store
        if actual_food_ready_time > exp_food_ready_t:
            wait_at_store = actual_food_ready_time - exp_store_arrive_t
            food_wait = 0
            manual_cook_time = wait_at_store + move_t
        elif exp_food_ready_t >= actual_food_ready_time >= exp_store_arrive_t:
            wait_at_store = actual_food_ready_time - exp_store_arrive_t
            food_wait = 0
            manual_cook_time = wait_at_store + move_t
        elif exp_store_arrive_t >= actual_food_ready_time:
            wait_at_store = 0
            food_wait = exp_store_arrive_t - actual_food_ready_time
            manual_cook_time = 0.001
        else:
            input('시간 관계 문제 발생2 ::actual_food_ready_time {}:: exp_store_arrive_t {} :: exp_food_ready_t{} '.format(actual_food_ready_time, exp_store_arrive_t, exp_food_ready_t))
    return wait_at_store, food_wait, manual_cook_time
