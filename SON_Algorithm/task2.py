###################################
## importing necessary libraries ##
###################################

from pyspark import SparkContext
from itertools import combinations, chain
import pyspark
import argparse
import json
import time

####################################
# all global variables locate here #
####################################

parser = argparse.ArgumentParser(description='A2T2')
parser.add_argument('--input_file', type=str, default='data/small1.csv', help='the input file')
parser.add_argument('--output_file', type=str, default='data/a2t2.json', help='the output file contains your answers')
parser.add_argument('--s', type=int, default=10, help='support')
parser.add_argument('--k', type=int, default=10, help='reviews threshold')

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


# Task 2 file creation
# review_rdd = sc.textFile('review.json')
# business_rdd = sc.textFile('business.json')

# new_review_rdd = review_rdd.map(lambda review: (json.loads(review)["business_id"], json.loads(review)["user_id"]))

# new_business_rdd = business_rdd.map(lambda business: (json.loads(business)["business_id"], json.loads(business)["state"]))\
# .filter(lambda x: x[1] == "NV")


# joined_rdd = new_review_rdd.join(new_business_rdd).map(lambda x: (x[1][0], x[0])).collect()

# import csv

# with open('task2_input.csv', 'w', newline='') as file:
#     writer = csv.writer(file)

#     for line in joined_rdd:
#         writer.writerow(line)

###############################
#### Initializing Text RDD ####
###############################

start_time = time.time()

#Input Data
A = sc.textFile(args.input_file).distinct()\
.map(lambda x : x.split(","))

B = A.map(lambda user: (user[0],1))\
.reduceByKey(lambda a, b: a + b)\
.filter(lambda x: x[1] > args.k)\
.distinct()

dataRDD = B.join(A)\
.map(lambda x: (x[0], x[1][1]))\
.distinct()

baskets = dataRDD.groupByKey()\
.map(lambda x: list(set(x[1]))).persist()

items_per_partition = baskets.glom().map(len).collect()
total_baskets = sum(items_per_partition)

########################################################################################################

######################
###### Phase 1 #######
######################

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

######################
###### Phase 2 #######
######################

def phase_2(variables):
    baskets = list(variables)
    frequent_actual = {}
    for basket in baskets:
        for candidate in clean_candidates:
            if set(candidate).issubset(set(basket)):
                if tuple(candidate) not in list(frequent_actual.keys()):
                    frequent_actual.update({tuple(candidate):1})
                else:
                    frequent_actual[tuple(candidate)]+=1
    return list(frequent_actual.items())

########################################################################################################

phase_1_map = baskets.mapPartitions(phase_1).flatMap(list).map(lambda x: (tuple(x),1))
phase_1_reduce = phase_1_map.reduceByKey(lambda a, b: a + b).collect()

clean_candidates = []
for x in phase_1_reduce:
    if len(x[0]) == 1:
        clean_candidates.append([x[0][0]])
    else:
        clean_candidates.append(list(x[0]))

grouped_candidates = {}
for x in clean_candidates:
    if len(x) not in list(grouped_candidates.keys()):
        grouped_candidates.update({len(x):[x]})
    else:
        grouped_candidates[len(x)].append(x)

for x,y in grouped_candidates.items():
    grouped_candidates[x] = sorted(y)

candidates_json = list(grouped_candidates.values())

phase_2_map = baskets.mapPartitions(phase_2)
phase_2_reduce = phase_2_map.reduceByKey(lambda x, y: x + y)\
.filter(lambda x: x[1] >= args.s)\
.map(lambda x: x[0])\
.collect()

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


