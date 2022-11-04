# -*- coding: utf-8 -*-
##'Instance_random','Instance_cluster'
import time

for instance_type in ['Instance_random','Instance_cluster']:
    for gen_B_size in [3]:
        for _ in range(10):
            try:
                exec(open('Simulator_for_GXBoost.py', encoding='UTF8').read(),globals().update(gen_B_size = gen_B_size, instance_type = instance_type))
            except:
                f = open('error_log.txt', 'a')
                tm = time.localtime(time.time())
                tm_str = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
                f.write('error_occurred;T; {}  \n'.format(tm_str))
                f.close()
                pass
        #input('test')