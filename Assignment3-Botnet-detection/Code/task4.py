# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:58:31 2017

@author: xps
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

file_name = 'capture20110817.binetflow'
L = 10 # taking top L n-grams into account
w = 40 #length of sliding window
n = 7 # the number of n-gram


df = pd.read_csv(file_name,sep=',')

# Assign simple labels
def assgin_label(label):
    if label.find('Normal')!=-1:
        return 0
    elif label.find('Botnet')!=-1:
        return 1 
    else:
        return -1
df['label'] = df['Label'].apply(lambda x: assgin_label(x))

# imbalanced: 184987 botnet, 29893 normal
new_df = df[df['label']!=-1]
data = new_df[['Dur','Dir','Proto','TotPkts','TotBytes']]
name_dir = df.Dir.unique()
name_dir_dict = {name:i for i,name in enumerate(name_dir)}
name_proto = df.Proto.unique()
name_proto_dict = {name:i for i,name in enumerate(name_proto)}
##  Obtaining Timed Events ##
def change_Dir(x):
    return name_dir_dict[x]
def change_Proto(x):
    return name_proto_dict[x]
#### TASK 4.##### We studied scenario 9. https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-50/
data['Dir'] = data['Dir'].apply(lambda x: change_Dir(x))
data['Proto'] = data['Proto'].apply(lambda x: change_Proto(x))
data['timestamp'] = new_df.StartTime
data['IP'] = new_df.SrcAddr

from mylibrary import elbow
# plot the ElBOW of each categorical feature to determine the number of cluster
elbow(data['Dur'])
elbow(data['TotPkts'])
elbow(data['TotBytes'])
#from the figure, we can determine the number of cluster ：4,3,4


def get_percentile(p_list,name):
    split_list = []
    for p in p_list:
        split_list.append(np.percentile(new_df[name],p))
    return split_list

def assign_value(x,split_list):
    for i,s in enumerate(split_list):
        if x<s:
            return i
    return len(split_list)

split_list_Dur = get_percentile([25,50,75],'Dur')
data['Dur'] = new_df['Dur'].apply(lambda x: assign_value(x,split_list_Dur))
    
split_list_pkt = get_percentile([33,80],'TotPkts')
data['TotPkts'] = new_df['TotPkts'].apply(lambda x: assign_value(x,split_list_pkt))

split_list_byte = get_percentile([25,50,75],'TotBytes')
data['TotBytes'] = new_df['TotBytes'].apply(lambda x: assign_value(x,split_list_byte))
     
m_list = [data[name].nunique() for name in data.columns[0:5]]


     
def coding(x):
    code = 0
    spacesize = m_list[0]*m_list[1]*m_list[2]*m_list[3]*m_list[4]
    for i in range (0,4):
        code = code + (x[i]) * spacesize / m_list[i]
        spacesize = spacesize / m_list[i]
    return code
data['code'] = data.apply(lambda x: coding(x),axis=1)        
# One-Hot Encoding for Proto
#data = pd.get_dummies(data)
data['timestamp'] = pd.to_datetime(data['timestamp'],format='%Y-%m-%d %H:%M:%S')


def extract_state(host_data,width=20):
    time1 = host_data['timestamp']
    difference_list = []
    for i in range(len(host_data)):
        if i == 0:
            diff = 0
        else:
            diff = time1.iloc[i]-time1.iloc[i-1]
            diff = np.ceil(diff.value/1e6)
        difference_list.append(diff)
    host_data['time'] = difference_list
            
    ## sliding windows ##
    state_list = []
    for i in range(len(host_data)):
        j = i
        state_list.append([])
        temp_list = [host_data['code'].iloc[j]]
        time_sum = 0
        while True:
            try:
                time_sum += difference_list[j+1]
            except:
                break
            j += 1
            if time_sum<=width:
                temp_list.append(host_data['code'].iloc[j])
            else:
                break
        if len(temp_list)>=3:
            state_list[i] = temp_list
    print 'finished: ',len(state_list)
    name = 'w%d_state' %40 
    host_data[name] = state_list
    return host_data
#Infected1 was used to model the fingerprint of infected host. 
#Nomarl1 was used to model the fingerprint of normal host
#These two fingerprint makes up the ground truth
#we used 40ms sliding window to obtain sequential data

infected1 = data[data['IP'] == '147.32.84.165']
infected1 = extract_state(infected1,width=w)

normal1 = data[data['IP'] == '147.32.84.164']
normal1 = extract_state(normal1,width=w)
#all the other labeled host was used as tesing data (https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-50/).
#we have 9 infected hosts and 5 normal hosts. 14 in total.
infected2 = data[data['IP'] == '147.32.84.191']
infected2 = extract_state(infected2,width=w)

infected3 = data[data['IP'] == '147.32.84.192']
infected3 = extract_state(infected3,width=w)

infected4 = data[data['IP'] == '147.32.84.193']
infected4 = extract_state(infected4,width=w)

infected5 = data[data['IP'] == '147.32.84.204']
infected5 = extract_state(infected5,width=w)

infected6 = data[data['IP'] == '147.32.84.205']
infected6 = extract_state(infected6,width=w)

infected7 = data[data['IP'] == '147.32.84.206']
infected7 = extract_state(infected7,width=w)

infected8 = data[data['IP'] == '147.32.84.207']
infected8 = extract_state(infected8,width=w)

infected9 = data[data['IP'] == '147.32.84.208']
infected9 = extract_state(infected9,width=w)

infected10 = data[data['IP'] == '147.32.84.209']
infected10 = extract_state(infected10,width=w)

normal2 = data[data['IP'] == '147.32.84.170']
normal2 = extract_state(normal2,width=w)

normal3 = data[data['IP'] == '147.32.84.134']
normal3 = extract_state(normal3,width=w)

normal4 = data[data['IP'] == '147.32.87.36']
normal4 = extract_state(normal4,width=w)

normal5 = data[data['IP'] == '147.32.80.9']
normal5 = extract_state(normal5,width=w)


# delete the null data
state_infected1 = [l for l in infected1['w40_state'] if len(l)>0]
state_normal1 = [l for l in normal1['w40_state'] if len(l)>0]

state_infected2 = [l for l in infected2['w40_state'] if len(l)>0]
state_infected3 = [l for l in infected3['w40_state'] if len(l)>0]
state_infected4 = [l for l in infected4['w40_state'] if len(l)>0]
state_infected5 = [l for l in infected5['w40_state'] if len(l)>0]
state_infected6 = [l for l in infected6['w40_state'] if len(l)>0]
state_infected7 = [l for l in infected7['w40_state'] if len(l)>0]
state_infected8 = [l for l in infected8['w40_state'] if len(l)>0]
state_infected9 = [l for l in infected9['w40_state'] if len(l)>0]
state_infected10 = [l for l in infected10['w40_state'] if len(l)>0]
state_normal2 = [l for l in normal2['w40_state'] if len(l)>0]
state_normal3 = [l for l in normal3['w40_state'] if len(l)>0]
state_normal4 = [l for l in normal4['w40_state'] if len(l)>0]
state_normal5 = [l for l in normal5['w40_state'] if len(l)>0]

## Sequential modal: n-grams ##

def find_ngrams(x):
    temp = []
    for i in range(len(x)):
        for j in range (len(x[i])-n+1):
            temp.append(x[i][j:j+n])    
    return temp
 
grams3_normal1 = find_ngrams(state_normal1)
grams3_infected1 = find_ngrams(state_infected1)

grams3_infected2 = find_ngrams(state_infected2)
grams3_infected3 = find_ngrams(state_infected3)
grams3_infected4 = find_ngrams(state_infected4)
grams3_infected5 = find_ngrams(state_infected5)
grams3_infected6 = find_ngrams(state_infected6)
grams3_infected7 = find_ngrams(state_infected7)
grams3_infected8 = find_ngrams(state_infected8)
grams3_infected9 = find_ngrams(state_infected9)
grams3_infected10 = find_ngrams(state_infected10)
grams3_normal2 = find_ngrams(state_normal2)
grams3_normal3 = find_ngrams(state_normal3)
grams3_normal4 = find_ngrams(state_normal4)
grams3_normal5 = find_ngrams(state_normal5)


def sort_ngrams(grams3_normals):
    ngram_dict = {}
    for gram in grams3_normals :
        grams = str(gram)[1:-1]
        if grams in ngram_dict:
            ngram_dict[grams] += 1
        else:
            ngram_dict[grams] = 1 
    sorted_ngrams = sorted(ngram_dict.items(),key = lambda x:x[1], reverse = True )
    sortedgrams_normed = [ (list[0], 1.0*list[1]/len(grams3_normals)) for list in sorted_ngrams]
    return sortedgrams_normed

fingerprint_normal1 = sort_ngrams(grams3_normal1)
fingerprint_infected1 = sort_ngrams(grams3_infected1)

fingerprint_normal = fingerprint_normal1
fingerprint_infected = fingerprint_infected1

fingerprint_infected2 = sort_ngrams(grams3_infected2)
fingerprint_infected3 = sort_ngrams(grams3_infected3)
fingerprint_infected4 = sort_ngrams(grams3_infected4)
fingerprint_infected5 = sort_ngrams(grams3_infected5)
fingerprint_infected6 = sort_ngrams(grams3_infected6)
fingerprint_infected7 = sort_ngrams(grams3_infected7)
fingerprint_infected8 = sort_ngrams(grams3_infected8)
fingerprint_infected9 = sort_ngrams(grams3_infected9)
fingerprint_infected10 = sort_ngrams(grams3_infected10)

fingerprint_normal2 = sort_ngrams(grams3_normal2)
fingerprint_normal3 = sort_ngrams(grams3_normal3)
fingerprint_normal4 = sort_ngrams(grams3_normal4)
fingerprint_normal5 = sort_ngrams(grams3_normal5)


def distance(x,y):
    x = np.array(x)
    y = np.array(y)
    dis = sum((np.divide((x-y),(x+y)/2))**2)
    return dis

def fingerprint_matching(x,y,L):
    x = x[0:L]
    freq_x = [pair[1] for pair in x]
    y = {pair[0]:pair[1] for pair in y}
    fre_y = []
    for i in range(L):
        key = x[i][0]
        if key in y:
            fre_y.append(y[key])
        else:
            fre_y.append(0)  
    dis = distance(freq_x,fre_y)
    return dis
##calculate the nearest neighbour of the 14 testing hosts.
fmatch_test= np.zeros((13,2))
fmatch_test[0][0] = fingerprint_matching(fingerprint_infected1,fingerprint_infected2,L)                
fmatch_test[0][1] = fingerprint_matching(fingerprint_normal1,fingerprint_infected2,L) 
fmatch_test[1][0] = fingerprint_matching(fingerprint_infected1,fingerprint_infected3,L)                
fmatch_test[1][1] = fingerprint_matching(fingerprint_normal1,fingerprint_infected3,L) 
fmatch_test[2][0] = fingerprint_matching(fingerprint_infected1,fingerprint_infected4,L)                
fmatch_test[2][1] = fingerprint_matching(fingerprint_normal1,fingerprint_infected4,L) 
fmatch_test[3][0] = fingerprint_matching(fingerprint_infected1,fingerprint_infected5,L)                
fmatch_test[3][1] = fingerprint_matching(fingerprint_normal1,fingerprint_infected5,L) 
fmatch_test[4][0] = fingerprint_matching(fingerprint_infected1,fingerprint_infected6,L)                
fmatch_test[4][1] = fingerprint_matching(fingerprint_normal1,fingerprint_infected6,L) 
fmatch_test[5][0] = fingerprint_matching(fingerprint_infected1,fingerprint_infected7,L)                
fmatch_test[5][1] = fingerprint_matching(fingerprint_normal1,fingerprint_infected7,L) 
fmatch_test[6][0] = fingerprint_matching(fingerprint_infected1,fingerprint_infected8,L)                
fmatch_test[6][1] = fingerprint_matching(fingerprint_normal1,fingerprint_infected8,L) 
fmatch_test[7][0] = fingerprint_matching(fingerprint_infected1,fingerprint_infected9,L)                
fmatch_test[7][1] = fingerprint_matching(fingerprint_normal1,fingerprint_infected9,L) 
fmatch_test[8][0] = fingerprint_matching(fingerprint_infected1,fingerprint_infected10,L)                
fmatch_test[8][1] = fingerprint_matching(fingerprint_normal1,fingerprint_infected10,L) 
fmatch_test[9][0] = fingerprint_matching(fingerprint_infected1,fingerprint_normal2,L)                
fmatch_test[9][1] = fingerprint_matching(fingerprint_normal1,fingerprint_normal2,L) 
fmatch_test[10][0] = fingerprint_matching(fingerprint_infected1,fingerprint_normal3,L)                
fmatch_test[10][1] = fingerprint_matching(fingerprint_normal1,fingerprint_normal3,L) 
fmatch_test[11][0] = fingerprint_matching(fingerprint_infected1,fingerprint_normal4,L)                
fmatch_test[11][1] = fingerprint_matching(fingerprint_normal1,fingerprint_normal4,L) 
fmatch_test[12][0] = fingerprint_matching(fingerprint_infected1,fingerprint_normal5,L)                
fmatch_test[12][1] = fingerprint_matching(fingerprint_normal1,fingerprint_normal5,L) 


test_label = np.zeros(13)
for i in range(13):
        if fmatch_test[i][0] <= fmatch_test[i][1]:
            test_label[i] = 1
        else:
            test_label[i] = 0

true_label = [1,1,1,1,1,1,1,1,1,0,0,0,0,]
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(13):
    if test_label[i] == 1 and true_label[i] == 1:
        tp += 1
    elif test_label[i] == 0 and true_label[i] == 1:
        fn += 1
        print "%d is netbot, but not detected!!!!!" %i
    elif test_label[i] == 0 and true_label[i] == 0:
        tn += 1
    else:
        fp += 1
        print "%d is benign, but detected as botnet!!!!!" %i

print 'precision:', float(tp)/(tp+fp)
print 'recall', float(tp)/(tp+fn) 





