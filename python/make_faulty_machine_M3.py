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

# ==================================================

# A Maker of Faulty Machine.
# It makes a faulty B machines by randomly removing current states.
# Input - source abstract machine, number of goal states, seed
# Output - 
# Usage: python [this script] [source abstract machine] [number of goal states] [reachability number] [seed] [result folder]

# =================================================

if len(sys.argv) != 6:
    print "Error: The number of input parameters should be 5."
    print "Usage: python [this script] [source abstract machine] [number of goal states] [reachability depth] [seed] [result folder]"
    exit(1)



M = sys.argv[1]
N = sys.argv[2]
RN = sys.argv[3]
SD = sys.argv[4]
resdir = sys.argv[5]

print "Making a faulty abstract machine..."

print "Source Abstract Machine:", M
print "Number of Goal States:", N
print "Reachability Number:", RN
print "Seed:", SD
print "Result Folder:", resdir

print "Data Preparation..."

N = int(N)
SD = int(SD)
RN = int(RN)
random.seed(SD)

cmd = "mkdir %s"%resdir
os.system(cmd)

s = resdir + "/source.mch"
cmd = "cp %s %s"%(M,s)
os.system(cmd)
M = s

fn = resdir + "/M.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,M)
os.system(oscmd)
M = fn

print "====== D = StateDiagram(M) ======"

with open(M) as mchf:
    mch = mchf.readlines()
mch = [x.strip() for x in mch]

D = resdir + "/D.txt"
max_initialisations = 65536 #Bmch.read_config(conffile,"max_initialisations","int")
max_operations = 65536 #Bmch.read_config(conffile,"max_operations","int")
bscope = Bmch.generate_training_set_condition(mch)
oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(M,max_initialisations,max_operations,bscope,D)
os.system(oscmd)

sg = Bgenlib.BStateGraphForNN()
sg.ReadStateGraph(D)
D = sg.GetTransList()
DS = D + []

#for x in D: print x
#x = raw_input("ppp")

print "====== LGS = StateRandomSelection(D,N) ======"

S = RepSimpLib.extract_all_states(D)
LGS = Bmch.random_selection(S,N)
for x in LGS: print x

print "====== LMT = [] ======"
LMT = []

"""
 for G in LGS:
    LDT = DeletedTransition(DS,PD)
    LMT = LMT + LDT
    DS = DS - LDT
MF = Update(MS,LMT)
return MF, LGS, LMT
"""

print "====== for G in LGS: ======"

LLL = len(D)
for G in LGS:
    print "====== LDT = DeletedTransitions(D,G,PD) ======"
    print "====== D = D - LDT ======"
    X = Bmch.delete_a_number_of_transitions(D,G,RN)
    D = X[0]
    LDT = X[1]

    print "====== LMT = LMT + LDT ======"
    LMT = Bmch.list_union(LMT,LDT)


print "LMT"
LMT.sort()
for X in LMT: print X
print len(LMT)

print "====== MF = Update(M,LMT) ======"
VList = sg.GetVbleList()
MF = mch + []
LC = Bmch.split_M_changes_by_operations(LMT)

for P in LC:
    ope = P[0]
    LMTX = P[1]
    MF = Bmch.apply_M_change(MF,ope,LMTX,VList)

fn = resdir + "/MF.mch"
Bmch.print_mch_to_file(MF,fn)
MF = fn

fn = resdir + "/result.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,MF)
os.system(oscmd)
MF = fn


print "====== ANS <-- ConvertChangesToAnswer(LMT) ======"

ANS = ["BEGIN_MISSING_TRANSITIONS"]
ANS.append("FORMAT: [ < Pre-state > , < Operation > , < Post-state > ]")
ANSD = []
for x in LMT:
    y = [x[0],x[1],x[2]]
    ANSD.append(y)
ANSD.sort()
ANS = ANS + ANSD
ANS.append("END_MISSING_TRANSITIONS")

print len(LMT)
#raw_input("sdsf")

fn = resdir + "/answer.txt"
f = open(fn,"w")
for x in ANS:
    f.write(str(x))
    f.write("\n")
f.close()
ANS = fn


fn = resdir + "/goal.txt"
f = open(fn,"w")
for x in LGS:
    N = 0
    for Y in DS:
        if Y[2] == x:
            N = N + 1
    f.write(str([x,N]))
    f.write("\n")
f.close()
LGS = fn


s = resdir + "/source.mch"
cmd = "cp %s %s"%(M,s)
os.system(cmd)
M = s

print "Source machine is %s."%M
print "Changed machine has been written to %s."%MF
print "Goal has been written to %s."%LGS
print "Answer has been written to %s."%ANS

print "Done."

