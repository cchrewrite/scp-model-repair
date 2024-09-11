import sys
import Bmch
import os
import time
import Bgenlib
import random
import RepSimpLib
import SemLearnLib
from nnet.nnetlib import *
from Cartlib import *
from NBayes import *
from SKCART import *
import numpy
import logging
import pickle
import time

# ==================================================

# Comparing the quality between two B models.
# Usage: python [this script] [evaluation result 1] [evaluation result 2] [comparison result]

# =================================================

#e.g., python src/python/b_model_evaluation_comparison.py TOSEM_Experiments/result_PE15_NDMI_IV100_10/repaired_SKCART_ADD4_NDMI_IV100_SD1001/source.eval TOSEM_Experiments/result_PE15_NDMI_IV100_10/repaired_SKCART_ADD4_NDMI_IV100_SD1001/result.eval TOSEM_Experiments/result_PE15_NDMI_IV100_10/repaired_SKCART_ADD4_NDMI_IV100_SD1001/comparison.eval


if len(sys.argv) != 4:
    print "Error: The number of input parameters should be 4."
    print "Usage: python [this script] [evaluation result 1] [evaluation result 2] [comparison result]"
    exit(1)

f1 = sys.argv[1]
f2 = sys.argv[2]
resf = sys.argv[3]

# Compute the relative improvement from x to y.
# x and y are values
# t --- type of improvement
# t == "imp0": normal relative improvement
# t == "imp1": relative improvement of the distance to 1.0 
def relative_improvement(x,y,t):
    if x < 0 or y < 0:
        print "Error: both x and y should be >= 0."
        return "None"
    if t == "imp0":
        if x - 0.0 < pow(10,-100) and y - 0.0 < pow(10,-100):
            return 0.0
        else: 
            return (y - x) * 1.0 / max(x,y)
    if t == "imp1":
        if x > 1.0 or y > 1.0:
            print "Error: both x and y should be <= 1.0."
            return "None"
        if 1.0 - x < pow(10,-100) and 1.0 - y < pow(10,-100):
            return 0.0
        else:
            return (y - x) * 1.0 / max(1.0-x,1.0-y)

def read_summary_file(fname):
    f = open(fname,"r")
    EV = []
    for x in f.readlines():
        if "**** SUMMARY ****" in x:
            continue
        y = x.split(": ")
        y[1] = eval(y[1])
        EV.append(y)
    return EV

EV1 = read_summary_file(f1)
EV2 = read_summary_file(f2)


CMP = ["******** COMPARISON ********"]
for x in EV1:
    if x[0] == "Modularity of each operation":
        continue
    for y in EV2:
        if y[0] == "Modularity of each operation":
            continue
        if x[0] == y[0] and x[0] in ["Total_Model_Checking_CPU_Time (s)","Peak_Memory_Usage (MB)","Capacity",]:
            ipv = relative_improvement(x[1],y[1],"imp0")
            z = [x[0],x[1],y[1],ipv]
            CMP.append(z)
        elif x[0] == y[0]:
            #ipv = relative_improvement(x[1],y[1],"imp1")
            ipv = y[1] - x[1]
            z = [x[0],x[1],y[1],ipv]
            CMP.append(z)


for x in EV1:
    flag = False
    for y in EV2:
        if x[0] == y[0]:
            flag = True
            break
    if flag == False:
        CMP.append("%s is missing in evaluation result 2."%x[0])
for x in EV2:
    flag = False
    for y in EV1:
        if x[0] == y[0]:
            flag = True
            break
    if flag == False:
        CMP.append("%s is missing in evaluation result 1."%x[0])
    
f = open(resf,"w")
for x in CMP:
    print x
    f.write(str(x))
    f.write("\n")
f.close()
