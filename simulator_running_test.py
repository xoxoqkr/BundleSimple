# -*- coding: utf-8 -*-
##'Instance_random','Instance_cluster'
import time

from Simulator_v3 import dynamic_env

run_time = 30
dir = "E:/python_백업/py_charm/BundleSimple/"
infos = [[True,False],[False,True],[True,True], [False,False]] #Static, Dynamic, Hybrid, P2P
for _ in range(1):
    for info in infos:
        s_t = time.time()
        try:
            exec(open(dir+'Simulator_v3.py', encoding='UTF8').read(),globals().update(run_time= run_time, platform_recommend_input= info[0],dynamic_env= info[1]))
            f = open(dir +'report_test.txt','a')
            e_t = time.time()
            #f.write('success; duration;{};t_now;{} \n'.format(e_t - s_t, time.strftime('%Y-%m-%d %I:%M:%S %p', e_t)))
            f.write('success \n')
            f.close()
            print('success')
        except:
            f = open(dir +'report_test.txt','a')
            e_t = time.time()
            #f.write('Fail; duration;{};t_now;{} \n'.format(e_t - s_t, time.strftime('%Y-%m-%d %I:%M:%S %p', e_t)))
            f.write('Fail \n')
            f.close()
            print('error')
        f = open(dir + 'report_test.txt', 'a')
        e_t = time.time()
        f.write('RunTime;{}; \n'.format(e_t - s_t))
        f.close()