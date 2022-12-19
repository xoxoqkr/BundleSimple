# -*- coding: utf-8 -*-
##'Instance_random','Instance_cluster'
import time
test_run_time = 50
for instance_type in ['동작구']:
    if instance_type == '송파구':
        speed = 1.95
    elif instance_type == '동작구':
        speed = 2.1
    else:
        print('error')
    for gen_B_size in [3,2]:
        for _ in range(0,5):
            try:
                order_dir = 'C:/Users/박태준/jupyter_notebook_base/data/' + instance_type + '/R_1206_NOR' + str(_) + '.txt'
                exec(open('Simulator_for_GXBoost2.py', encoding='UTF8').read(),
                     globals().update(gen_B_size=gen_B_size, instance_type=instance_type, order_dir=order_dir,
                                      test_run_time=test_run_time, input_speed = speed))
            except:
                f = open('error_log.txt', 'a')
                tm = time.localtime(time.time())
                tm_str = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
                f.write('error_occurred;T; {}  \n'.format(tm_str))
                f.close()
                pass
        #input('test')