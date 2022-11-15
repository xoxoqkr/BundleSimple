# -*- coding: utf-8 -*-
##'Instance_random','Instance_cluster'
import time
#고객 주문이 발생하자 마자 바로 서비스 받을 수 있는 상태가 되는 경우 <- 이번 실험 상태
#고객들의 주문이 발생 후 cancel == True인 상태에서 다음 interval이 되면 canceal = False가 되는 상황
run_time = 120
customer_pend_options = [True, False]
dir = "E:/python_백업/py_charm/BundleSimple/"
basic_infos = [[False,False],[False,True],[True,False],[True,True]] #P2P,Dynamic,Static,Hybrid,
basic_infos = [[False,True],[True,True]] #
infos = []

for info in basic_infos:
    for customer_pend in customer_pend_options:
        infos.append(info + [customer_pend])
print(infos)
input('info 확인')
for info in infos:
    for _ in range(1):
        s_t = time.time()
        exec(open(dir + 'Simulator_v3.py', encoding='UTF8').read(),
             globals().update(run_time=run_time, platform_recommend_input=info[0], dynamic_env=info[1],
                              customer_pend=info[2]))
        f = open(dir + 'report_test.txt', 'a')
        e_t = time.time()
        # f.write('success; duration;{};t_now;{} \n'.format(e_t - s_t, time.strftime('%Y-%m-%d %I:%M:%S %p', e_t)))
        f.write('success ;{};{}; \n'.format(info[0], info[1]))
        f.close()
        print('success')
        """
        try:
            exec(open(dir+'Simulator_v3.py', encoding='UTF8').read(),globals().update(run_time= run_time, platform_recommend_input= info[0],dynamic_env= info[1], customer_pend = info[2]))
            f = open(dir +'report_test.txt','a')
            e_t = time.time()
            #f.write('success; duration;{};t_now;{} \n'.format(e_t - s_t, time.strftime('%Y-%m-%d %I:%M:%S %p', e_t)))
            f.write('success ;{};{}; \n'.format(info[0], info[1]))
            f.close()
            print('success')
        except:
            f = open(dir +'report_test.txt','a')
            e_t = time.time()
            #f.write('Fail; duration;{};t_now;{} \n'.format(e_t - s_t, time.strftime('%Y-%m-%d %I:%M:%S %p', e_t)))
            f.write('Fail \n')
            f.close()
            print('error')
        """
        f = open(dir + 'report_test.txt', 'a')
        e_t = time.time()
        f.write('RunTime;{}; \n'.format(e_t - s_t))
        f.close()