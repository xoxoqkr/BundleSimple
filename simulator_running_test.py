# -*- coding: utf-8 -*-
##'Instance_random','Instance_cluster'
import time

for _ in range(100):
    s_t = time.time()
    try:
        exec(open('Simulator_v3.py', encoding='UTF8').read())
        f = open('report_test.txt','a')
        e_t = time.time()
        f.write('success; duration;{};t_now;{} \n'.format(e_t - s_t, time.strftime('%Y-%m-%d %I:%M:%S %p', e_t)))
        f.close()
    except:
        f = open('report_test.txt','a')
        e_t = time.time()
        f.write('Fail; duration;{};t_now;{} \n'.format(e_t - s_t, time.strftime('%Y-%m-%d %I:%M:%S %p', e_t)))
        f.close()
