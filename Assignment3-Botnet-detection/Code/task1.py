import glob

#file_names = glob.glob('F:\TU Delft\*.csv')
#ip_list = []
#
#for f_name in file_names:
#    f = open(f_name,'r')
#    for line in f:
#        ip_list.append(line.split('$$')[1])
#    f.close()
#    print "Finished %s !" %f_name
#
## Save data to a local file
#f = open('ip_data.txt','a')
#for ip in ip_list:
#    f.write(ip+'\n')
#f.close()

# Load local data
ip_list = []
f = open('ip_data.txt','r')
for line in f:
    ip_list.append(line[0:-1])
f.close()
print "ip data have been loaded!"

# Count number of unique ip address
ip_set = set(ip_list)
ip_unique_count = len(ip_set)
print "Unique count is: %d" %ip_unique_count 

# Count the 10 most frequent
#sorted_ip = []
#for ip in ip_set:
#    ip_pair = [ip, ip_list.count(ip)]
#    sorted_ip.append(ip_pair)
#sorted_ip = sorted(sorted_ip, key=lambda pair: pair[1], reverse = True)

# more efficient way?
sorted_ip_dict = {}
for i, ip in enumerate(ip_list):
    if ip in sorted_ip_dict:
        sorted_ip_dict[ip] += 1
    else:
        sorted_ip_dict[ip] = 1
    if i%100000 == 0:
        print "Finished %d!" %i
sorted_ip_list = sorted(sorted_ip_dict.items(), key=lambda pair: pair[1], reverse = True)



# print out top ten 
for i in range(10):
    ip = sorted_ip_list[i][0]
    count = sorted_ip_list[i][1]
    print "ip: %s, True count: %d" %(ip, count)
