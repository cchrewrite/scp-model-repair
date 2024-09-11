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

# Comparing our repair refactoring algorithm with Joshua el al's tool.
# Usage: python [this script] [B model] [configuration file] [result folder]

# =================================================

# e.g. python src/python/joshua_et_al_compare.py TOSEM_Experiments/joshua_et_al TOSEM_Experiments/joshua_et_al_comparison/scheduler_final.mch TOSEM_Experiments/joshua_et_al_comparison/config TOSEM_Experiments/joshua_et_al_comparison/result

start_time = time.time()

sd = 777

if len(sys.argv) != 4:
    print "Error: The number of input parameters should be 4."
    print "python [this script] [B model] [configuration file] [result folder]"
    exit(1)


MS = sys.argv[1] + ""
MA = sys.argv[1] + ""
conffile = sys.argv[2]
resdir = sys.argv[3]

print "B Model:", MS
print "Configuration File:", conffile
print "Result Folder:", resdir


print "Data Preparation..."

cmd = "mkdir %s"%resdir
os.system(cmd)

s = resdir + "/MS.mch"
cmd = "cp %s %s"%(MS,s)
os.system(cmd)
MS = s

fn = resdir + "/MS_pp.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,MS)
os.system(oscmd)
MS = fn

with open(MS) as mchf:
    mch = mchf.readlines()
mch = [x.strip() for x in mch]


s = resdir + "/MA.mch"
cmd = "cp %s %s"%(MA,s)
os.system(cmd)
MA = s

fn = resdir + "/MA_pp.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,MA)
os.system(oscmd)
MA = fn

with open(MA) as mchf:
    mchA = mchf.readlines()
mchA = [x.strip() for x in mchA]

print "Computing the state space of MS..."

TS = resdir + "/TS.txt"
max_initialisations = Bmch.read_config(conffile,"max_initialisations","int")
max_operations = Bmch.read_config(conffile,"max_operations","int")
bscope = Bmch.generate_training_set_condition(mch)
oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MS,max_initialisations,max_operations,bscope,TS)
os.system(oscmd)

sg = Bgenlib.BStateGraphForNN()
sg.ReadStateGraph(TS)
TS = sg.GetTransList()
VList = sg.GetVbleList()

TA = []

print "Making test data..."

FTL = [["del",6],["new",8],["set_active",8],["set_ready",9],["active_to_waiting",11],["ready_to_active",10]]

TST = []
TAT = []

for FT in FTL:
    op = FT[0]
    nt = FT[1]
    U = []
    for x in TS:
        if x[1] == op:
            U.append(x)

    random.shuffle(U)
    TST.append(U[0])
    TAT = TAT + U[0:nt+1]

for x in TST: print x
raw_input("Press Enter to Continue...")
for x in TAT: print x
raw_input("Press Enter to Continue...")

TS = TST
TA = TAT

RIAll = Bmch.list_difference(TA,TS)

simpdir = resdir + "/repair_simplification/"
RISimp = []
RIF = []
TRes = []

for FT in FTL:
    op = FT[0]
    RI = []
    for x in RIAll:
        if x[1] == op:
            RI.append(x)

    print "Synthesising conditions and substitutions for \"%s\" operation..."%op
    t1 = time.time()
    TAE = TA + []
    W = RepSimpLib.CFGInsertionSimplification(RI,TAE,VList,conffile,simpdir)
    RISimp = RISimp + W
    RIF.append(W)
    t2 = time.time()
    TRes.append([op,t2-t1])
    print "End. Time consumption: %.3f (s).\n"%(t2-t1)
    raw_input("Press Enter to Continue...")

print "====== Time Consumptions ======"
for x in TRes: print x

print "====== End of Evaluation ======"


