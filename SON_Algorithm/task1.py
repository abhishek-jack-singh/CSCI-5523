###################################
## importing necessary libraries ##
###################################

from itertools import combinations, chain
import pyspark
import argparse
import json
import time

#####################################
# all global variables located here #
#####################################

parser = argparse.ArgumentParser(description='A2T1')
parser.add_argument('--input_file', type=str, default='data/small1.csv', help='the input file')
parser.add_argument('--output_file', type=str, default='data/a2t1.json', help='the output file contains your answers')
parser.add_argument('--s', type=int, default=5, help='support')
parser.add_argument('--c', type=int, default=1, help='case we want to evaluate')

args = parser.parse_args()

####################################
#### initializing Spark Context ####
####################################

if __name__ == '__main__':
    
    sc_conf = pyspark.SparkConf()\
        .setAppName('task1')\
        .setMaster('local[*]')\
        .set('spark.driver.memory', '8g')\
        .set('spark.executor.memory', '4g')
    
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

###############################
#### Initializing Text RDD ####
###############################

# config for case 1 and case 2
if args.c == 1:
    k,v = 0,1
elif args.c == 2:
    k,v = 1,0

# let's time the execution
start_time = time.time()

# importing data and getting rid of the header
dataRDD = sc.textFile(args.input_file)
header = dataRDD.first()

baskets = dataRDD.filter(lambda item: item != header)\
.map(lambda basket: basket.split(','))\
.map(lambda x: (x[k],x[v]))\
.groupByKey()\
.map(lambda x: list(set(x[1]))).persist()

# getting total number of baskets
# required later for chunk support calculation
items_per_partition = baskets.glom().map(len).collect()
total_baskets = sum(items_per_partition)

########################################################################################################
# In the SON Algorithm, during the 1st Pass (phase 1) we try to get the probable frequent items
# During the 2nd Pass (phase 2) we check the above shortlisted candidates for actually frequent items
########################################################################################################

####################################
############# Phase 1 ##############
####################################

def phase_1(variable):
    singles = {}
    frequent_singles = []
    baskets = list(variable)
    chunk_support = (len(baskets)/total_baskets)*args.s

    # generating and counting singles
    for basket in baskets:
        for item in basket:
            if item not in list(singles.keys()):
                singles.update({item:1})
            else:
                singles[item]+=1
    
    # filtering singles
    for key, value in singles.items():
        if value >= chunk_support:
            frequent_singles.append([key])
    
    candidates = {1: frequent_singles}

    level = 2

    # generating all candidates and checking them
    while len(candidates[level-1]) > 0:
        candidates.update({level:[]})
        frequent_items = set(chain(*candidates[level-1]))
        
        #Making candidates.     
        for basket in baskets:
            if len(basket) >= level:
                basket_items = set(basket).intersection(frequent_items)                
                combos = [list(x) for x in combinations(sorted(basket_items), level)]
                for combo in combos:                
                    if combo not in candidates[level]:
                        flag = True

                        #checking if frequent in one level below
                        for subset in [list(x) for x in combinations(combo, level-1)]:
                            flag *= subset in candidates[level-1]
                            if not flag:
                                break
                        if flag:
                            candidates[level].append(combo)

        #Checking if candidates meet chunk support.
        for candidate in list(candidates[level]):
            c = 0
            for basket in baskets:
                if set(candidate).issubset(set(basket)):
                    c += 1
                if c >= chunk_support:
                    break
            if c < chunk_support:
                candidates[level].remove(candidate)
        
        level+=1
    return candidates.values()

########################################################################################################

####################################
############# Phase 2 ##############
####################################

def phase_2(variables):
    baskets = list(variables)
    actually_frequent = {}
    for basket in baskets:
        for candidate in clean_candidates:
            if set(candidate).issubset(set(basket)):
                if tuple(candidate) not in list(actually_frequent.keys()):
                    actually_frequent.update({tuple(candidate):1})
                else:
                    actually_frequent[tuple(candidate)]+=1
    return list(actually_frequent.items())

########################################################################################################

# Now that the functions are defined, let's implement it using map and reduce

# applying phase 1
phase_1_map = baskets.mapPartitions(phase_1).flatMap(list).map(lambda x: (tuple(x),1))
phase_1_reduce = phase_1_map.reduceByKey(lambda a, b: a + b).collect()

# pre-process the data before phase 2
clean_candidates = []
for x in phase_1_reduce:
    if len(x[0]) == 1:
        clean_candidates.append([x[0][0]])
    else:
        clean_candidates.append(list(x[0]))

# preparing candidates for output
grouped_candidates = {}
for x in clean_candidates:
    if len(x) not in list(grouped_candidates.keys()):
        grouped_candidates.update({len(x):[x]})
    else:
        grouped_candidates[len(x)].append(x)

for x,y in grouped_candidates.items():
    grouped_candidates[x] = sorted(y)

candidates_json = list(grouped_candidates.values())

# applying phase 2
phase_2_map = baskets.mapPartitions(phase_2)
phase_2_reduce = phase_2_map.reduceByKey(lambda x, y: x + y)\
.filter(lambda x: x[1] >= args.s)\
.map(lambda x: x[0])\
.collect()

# preparing frequents for output
clean_frequent = []
for x in phase_2_reduce:
    if len(x) == 1:
        clean_frequent.append([x[0]])
    else:
        clean_frequent.append(list(x))


grouped_frequent = {}
for x in clean_frequent:
    if len(x) not in list(grouped_frequent.keys()):
        grouped_frequent.update({len(x):[x]})
    else:
        grouped_frequent[len(x)].append(x)

for x,y in grouped_frequent.items():
    grouped_frequent[x] = sorted(y)

frequent_json = list(grouped_frequent.values())

########################################################################################################

data ={
"Candidates": candidates_json,
"Frequent Itemsets": frequent_json,
"Runtime": time.time() - start_time
}

with open(args.output_file, 'w') as outputfile:
    json.dump(data, outputfile)


