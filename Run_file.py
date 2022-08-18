# -*- coding: utf-8 -*-
#from Simulator_v3 import exp_range
heuristic_type = 'XGBoost'
exp_range = [0,2,3,4]*2  #[0,2,3,4] * 2
rider_list = [10,12]#[5,6,7,8,9]
start_heads = ['1221_기본거리O_거리할증110원','1221_기본거리X_거리할증110원']
service_time_diff = True #True인 경우, 고객의 service_time이 1~5분 사이에 분포.
unit_fee = 110
head_index = 0
for fee_type in ['step','linear']:
    f3 = open("결과저장1209.txt", 'a')
    f3.write(start_heads[head_index] + '\n')
    f3.close()

    for rider_num in rider_list:
        #exec(open('Simulator_v3.py', encoding='UTF8').read(),globals().update(instance_type='Instance_random', ellipse_w=10, heuristic_theta=10,heuristic_r1=10, heuristic_type='enumerate', rider_num=rider_num, mix_ratios=None, exp_range = exp_range, unit_fee = unit_fee,fee_type = fee_type, service_time_diff = service_time_diff))
        exec(open('Simulator_v3.py', encoding='UTF8').read(),globals().update(instance_type='Instance_cluster', ellipse_w=10, heuristic_theta=10, heuristic_r1=10,heuristic_type=heuristic_type, rider_num=rider_num, mix_ratios=None, exp_range = exp_range, unit_fee = unit_fee,fee_type = fee_type, service_time_diff = service_time_diff))
        #input('결과 확인')
    """
    for heuristic_theta in [110,130,150]:
        for heuristic_r1 in [8,11]:
            for rider_num in rider_list:
                exec(open('Simulator_v3.py', encoding='UTF8').read(), globals().update(instance_type='Instance_random',ellipse_w = 10, heuristic_theta = heuristic_theta, heuristic_r1= heuristic_r1, heuristic_type = 'heuristic', rider_num = rider_num, mix_ratios = None, exp_range = exp_range, unit_fee = unit_fee,fee_type = fee_type, service_time_diff = service_time_diff))
                exec(open('Simulator_v3.py', encoding='UTF8').read(), globals().update(instance_type='Instance_cluster', ellipse_w=10, heuristic_theta=heuristic_theta,heuristic_r1=heuristic_r1,heuristic_type = 'heuristic', rider_num = rider_num, mix_ratios = None, exp_range = exp_range, unit_fee = unit_fee,fee_type = fee_type, service_time_diff = service_time_diff))
                #input('종료')


    scenario_indexs = []
    for ellipse_w in [1.5,1.7,2]: #w를 조정
        for rider_num in rider_list:
            exec(open('Simulator_v3.py', encoding='UTF8').read(), globals().update(instance_type='Instance_random',ellipse_w = ellipse_w, heuristic_theta = 100, heuristic_r1= 10,heuristic_type = 'ellipse', rider_num = rider_num,mix_ratios = None, exp_range = exp_range, unit_fee = unit_fee,fee_type = fee_type, service_time_diff = service_time_diff))
            exec(open('Simulator_v3.py', encoding='UTF8').read(), globals().update(instance_type='Instance_cluster',ellipse_w = ellipse_w, heuristic_theta = 100, heuristic_r1= 10,heuristic_type = 'ellipse', rider_num = rider_num, mix_ratios = None, exp_range = exp_range, unit_fee = unit_fee,fee_type = fee_type, service_time_diff = service_time_diff))
            #exec(open('Simulator_v3.py', encoding='UTF8').read(),globals().update(instance_type='Instance_random', ellipse_w=ellipse_w, heuristic_theta=100,heuristic_r1=10, heuristic_type='ellipse', rider_num=rider_num,mix_ratios=[0.7, 0.8, 0.9]))
            #input('사이클 종료')

    
    for ellipse_delta in [5,7,9]: #w를 조정
        for rider_num in rider_list:
            exec(open('Simulator_v3.py', encoding='UTF8').read(), globals().update(instance_type='Instance_random',ellipse_w = ellipse_delta, heuristic_theta = 100, heuristic_r1= 10,heuristic_type = 'ellipse_MJ', rider_num = rider_num,mix_ratios = None, exp_range = exp_range, unit_fee = unit_fee,fee_type = fee_type, service_time_diff = service_time_diff))
            exec(open('Simulator_v3.py', encoding='UTF8').read(), globals().update(instance_type='Instance_cluster',ellipse_w = ellipse_delta, heuristic_theta = 100, heuristic_r1= 10,heuristic_type = 'ellipse_MJ', rider_num = rider_num, mix_ratios = None, exp_range = exp_range, unit_fee = unit_fee,fee_type = fee_type, service_time_diff = service_time_diff))
            #exec(open('Simulator_v3.py', encoding='UTF8').read(),globals().update(instance_type='Instance_random', ellipse_w=ellipse_w, heuristic_theta=100,heuristic_r1=10, heuristic_type='ellipse', rider_num=rider_num,mix_ratios=[0.7, 0.8, 0.9]))
            #input('사이클 종료')
    """
    head_index += 1