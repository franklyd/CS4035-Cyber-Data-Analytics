# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 22:17:50 2017

@author: xps
"""

# FREQUENT algorithm

stream = ['g','g','b','g','b','y','bk','b','g','b','g','b']

def reservoir_sampling(stream, k,debug=False):
    
    '''
    stream: unknown stream, type is list 
    k-1 memory space
    
    '''
    frequent_dict = {}
    for s in stream:
        if s in frequent_dict.keys():
            frequent_dict[s] += 1
        elif len(frequent_dict) < k-1:
            frequent_dict[s] = 1
        else:
            frequent_dict = {k:v-1 for k,v in frequent_dict.items() if v>1}
    if debug:        
        print frequent_dict
    
    return frequent_dict

#reservoir_sampling(stream,3,True)

frequent_dict = reservoir_sampling(ip_list,1000)

for ip_pair in sorted_ip_list[0:10]:
    ip = ip_pair[0]
    true_count = ip_pair[1]
    try:
        estimate_count = frequent_dict[ip]
    except:
        estimate_count = 0
    print "ip: %s, True count: %d, estimate count: %d" %(ip, true_count, estimate_count)