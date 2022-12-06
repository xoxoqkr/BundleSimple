# -*- coding: utf-8 -*-
##'Instance_random','Instance_cluster'
import time
test_run_time = 10
for instance_type in ['송파구','동작구']:
    for gen_B_size in [3,2]:
        for _ in range(1):
            order_dir = 'C:/Users/박태준/jupyter_notebook_base/data/' + instance_type + '/1206_ORG' + str(_) + '.txt'
            exec(open('Simulator_for_GXBoost2.py', encoding='UTF8').read(),
                 globals().update(gen_B_size=gen_B_size, instance_type=instance_type, order_dir=order_dir,
                                  test_run_time=test_run_time))
            """
            try:
                #exec(open('Simulator_for_GXBoost.py', encoding='UTF8').read(),globals().update(gen_B_size = gen_B_size, instance_type = instance_type))
                exec(open('Simulator_for_GXBoost2.py', encoding='UTF8').read(),globals().update(gen_B_size=gen_B_size, instance_type=instance_type, order_dir = order_dir, test_run_time = test_run_time))
            except:
                f = open('error_log.txt', 'a')
                tm = time.localtime(time.time())
                tm_str = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
                f.write('error_occurred;T; {}  \n'.format(tm_str))
                f.close()
                pass
            """
        #input('test')