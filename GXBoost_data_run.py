# -*- coding: utf-8 -*-
##'Instance_random','Instance_cluster'

for instance_type in ['Instance_random','Instance_cluster']:
    for gen_B_size in [2, 3]:
        for _ in range(100):
            exec(open('Simulator_for_GXBoost.py', encoding='UTF8').read(),globals().update(gen_B_size = gen_B_size, instance_type = instance_type))