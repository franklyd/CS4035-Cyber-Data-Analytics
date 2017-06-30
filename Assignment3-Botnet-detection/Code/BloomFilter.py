# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 00:19:19 2017
Bloom filter implementation
@author: Yadong Li
"""

from mmh3 import hash
from bitarray import bitarray
import numpy as np
import math

class BloomFilter:
    
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_values = bitarray(size)
        self.hash_values.setall(0)
        self.seeds = range(hash_count)
        
    def add(self, element):
        for seed in self.seeds:
            result = hash(element, seed) % self.size
            self.hash_values[result] = 1
        return self.hash_values
            
    def lookup(self, element):
        for seed in self.seeds:
            result = hash(element, seed) % self.size
            if self.hash_values[result] == 0:
                return False
        return True

ip_list = []
f = open('ip_data.txt','r')
for line in f:
    ip_list.append(line[0:-1])
f.close()
print "ip data have been loaded!"

# Training/testing split
training_size = len(ip_list) / 2

training = ip_list[0:training_size]
testing = ip_list[training_size:]

training_set = set(training)
testing_label = [ip in training_set for ip in testing]

# create bf
bit_size = 1000000
#bit_size = 100000
#bit_size = 10000
p = np.power(math.e,-((np.log(2))**2) * bit_size / training_size)
print "Theorical false positive rate:",p
#bit_size = - training_size * np.log(p) / ((np.log(2))**2)

bit_size = int(bit_size)
print "bit size is:%d" %bit_size

hash_count = np.log(2) *  bit_size / training_size
hash_count = int(np.ceil(hash_count))
print "number of hashes to use:%d" %hash_count

bf = BloomFilter(bit_size,hash_count)

# add training data
for ip in training:
    bf.add(ip)

# for testing:
#for i in range(len(testing)):
#    ip = np.random.choice(testing,1)[0]
#    predict = bf.lookup(ip)
#    truth = ip in training
#    if truth != predict:
#        print 'prediction:', predict,'  truth:', truth

predict =[]
for ip in testing:
    predict.append(bf.lookup(ip))


fp = 1.0* sum([1 for i in range(len(testing_label))
 if predict[i]==True and testing_label[i]==False])/len(testing)

print "False positive rate is:", fp

fn = 1.0* sum([1 for i in range(len(testing_label))
 if predict[i]==False and testing_label[i]==True])/len(testing)
