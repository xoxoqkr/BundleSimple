def OrdergeneratorByCSVForStressTest(env, orders, stores, lamda, platform = None, p2_ratio = 1, rider_speed = 1, unit_fee = 110, fee_type = 'linear',
                                     output_data = None, cooktime_detail = None, cook_first = False, customer_pend = False, manual_cook_time = 7):
    """
    Generate customer order
    :param env: Simpy Env
    :param orders: Order
    :param platform: 플랫폼에 올라온 주문들 {[KY]order index : [Value]class order, ...}
    :param stores: 플랫폼에 올라온 가게들 {[KY]store name : [Value]class store, ...}
    :param interval: 주문 생성 간격
    :param runtime: 시뮬레이션 동작 시간
    """
    dist_distribution = numpy.random.poisson(20,7) #todo: 0915 거리 조절
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
            cook_time = numpy.random.choice(cooktime_detail[0], p = cooktime_detail[1]) # todo : 221101실험을 현실적으로 변경.
            p2_ratio2 = store.p2
        else:
            cook_time = manual_cook_time
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
        #order.dp_cook_time = cook_time
        order.dp_cook_time = 5*(1 + order.actual_cook_time//5)
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
            platform_exp_error = 1
            pool = numpy.random.normal(order.cook_info[1][0], order.cook_info[1][1] * platform_exp_error, 1000)
            order.platform_exp_cook_time = random.choice(pool)
            route = [[order.name, 0, order.store_loc, 0], [order.name, 1, order.location, 0]]
            o = re_A1_class.Order(task_index, [order.name], route, 'single', fee=order.fee, parameter_info=None)
            platform.platform[task_index] = o
        print('T {} 음식 {} 조리 확인/ 시간 {}'.format(int(env.now), order.name,order.actual_cook_time))
        orders[name] = order
        stores[store_name].received_orders.append(orders[name])
        interval = 1.0/lamda
        if interval > 0:
            yield env.timeout(interval)
        else:
            print('현재 T :{} / 마지막 고객 {} 생성'.format(int(env.now), name))
            pass