# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:47:47 2017

@author: xps
"""
from mmh3 import hash
import numpy as np
import math

class CountMinSketch:
    
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.sketch_array = np.zeros((hash_count,size),dtype=np.int)
        
    def add(self, string):
        for seed in xrange(self.hash_count):
            result = hash(string, seed) % self.size
            self.sketch_array[seed][result]+=1
            
    def estimate(self, string):
        minimum = 1000000
        for seed in xrange(self.hash_count):
            result = hash(string, seed) % self.size
            minimum = min(minimum,self.sketch_array[seed][result])
        return minimum
        
epsilon = 0.0001
delta = 0.1

w = int(2 / epsilon)  # num of columns
d = int(np.log(1/delta)) #number of row, hash count

cm = CountMinSketch(w, d)

for item in ip_list:
    cm.add(str(item))

errors = 0
for ip_pair in sorted_ip_list[0:10]:
    ip = ip_pair[0]
    true_count = ip_pair[1]
    try:
        estimate_count = cm.estimate(str(ip))
    except:
        estimate_count = 0
    error = 1.0*abs(true_count-estimate_count)/len(ip_list)
    errors += error
    print "ip:%s, True: %d, estimate: %d error: %f" %(ip, true_count, estimate_count,error )
print "average top ten error is: ",errors/10

    
    