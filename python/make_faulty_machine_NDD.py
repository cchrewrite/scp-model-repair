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
# It makes a faulty B machine by randomly removing existing states. Then a set of goals indicating the removed states and with noisy states are specified.
# Input - source abstract machine, percentage of removed states, seed
# Output - result abstract machine (result.mch) and evaluation data (eval.data)
# Usage: python [this script] [source abstract machine] [percentage of removed states] [seed] [result folder]

# =================================================

if len(sys.argv) != 5:
    print "Error: The number of input parameters should be 5."
    print "Usage: python [this script] [source abstract machine] [percentage of removed states] [seed] [result folder]"
    exit(1)



M = sys.argv[1]
PN = sys.argv[2]
SD = sys.argv[3]
resdir = sys.argv[4]


print "Making a faulty abstract machine..."

print "Source Abstract Machine:", M
print "Percentage of Removed States:", PN
print "Seed:", SD
print "Result Folder:", resdir

print "Data Preparation..."

PN = int(PN) * 1.0 / 100
SD = int(SD)
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

print "====== D <-- StateDiagram(M) ======"

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

print "====== S <-- AllStates(D) ======"

S = RepSimpLib.extract_all_states(D)

N = int(len(D) * PN)

print "====== Preparing Evaluation Data... ======"

ED = ["BEGIN_DESIRED_STATE_TRANSITIONS"]
ED.append("FORMAT: [ < Pre-state > , < Operation > , < Post-state > ]")
DST = D + []
DST.sort()
ED = ED + DST
ED.append("END_DESIRED_STATE_TRANSITIONS")

print "====== LGS,LMT = RandomRemoval(D,N) ======"

LGS = []
LMT = []

ST = S + []
random.shuffle(ST)

for q in ST:
    LGS.append(q)
    for x in D:
        if x[2] == q:
            LMT.append(x)
    if len(LMT) >= N:
        break

LMT.sort()

print "====== MF = Update(M,LMT) ======"
VList = sg.GetVbleList()
MF = mch + []

for i in xrange(len(LMT)):
    LMT[i] = LMT[i] + ["isolation"]

epid = "mkdt0"

RD = []
for x in LMT:
    y = Bmch.atomic_modification_or_deletion_to_conditional_substitution(x,VList,epid)
    RD.append(y)


FL = []
for x in DS:
    if not(x[1] in FL):
        FL.append(x[1])

for ope in FL:
    RL = []
    for x in RD:
        if x[0] == ope:
            RL.append([x[1],x[2]])
    if RL != []:
        MF = Bmch.apply_modifications_and_deletions(MF,ope,RL,VList,epid)

fn = resdir + "/MF.mch"
Bmch.print_mch_to_file(MF,fn)
MF = fn

fn = resdir + "/result.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,MF)
os.system(oscmd)
MF = fn

# Generate a set of noisy goal predicates.
S = RepSimpLib.extract_all_states(DS)
SM = S + []
for i in xrange(len(SM) * 10):
    X = []
    for j in xrange(len(S[0])):
        SX = SM[int(random.random() * len(SM))]
        X.append(SX[j])
    SM.append(X)
SM = Bmch.list_difference(SM,S)

LGP = []
for x in LGS:
    GP = [x]
    if SM == []:
        print "Warning: not able to find noisy states."
    else:
        for i in xrange(int(10 * random.random())):
            y = SM[int(random.random() * len(SM))]
            GP.append(y)
    GP = Bmch.convert_states_to_conditions(GP,VList)
    LGP.append(GP)


ED.append("BEGIN_DELETED_STATES")
for x in LGS:
    ED.append(x)
ED.append("END_DELETED_STATES")

ED.append("BEGIN_DELETED_STATE_TRANSITIONS")
for x in LMT:
    ED.append([x[0],x[1],x[2]])
ED.append("END_DELETED_STATE_TRANSITIONS")


fn = resdir + "/goal_predicates.txt"
f = open(fn,"w")

ED.append("BEGIN_GOALS")
for i in xrange(len(LGS)):
    x = LGS[i]
    p = LGP[i]
    RN = 0
    for Y in DS:
        if Y[2] == x:
            RN = RN + 1
    ED.append(str([p,RN]))
    f.write(str([p,RN]))
    f.write("\n")
ED.append("END_GOALS")
f.close()
LGP = fn

ED.append("BEGIN_GENERAL_INFO")
ED.append(["Number of States in the Source Machine",len(S)])
ED.append(["Number of State Transitions in the Source Machine",len(DS)])
ED.append(["Number of Deleted States",len(LGS)])
ED.append(["Number of Deleted State Transitions",len(LMT)])
ED.append("END_GENERAL_INFO")


s = resdir + "/source.mch"
cmd = "cp %s %s"%(M,s)
os.system(cmd)
M = s

fn = resdir + "/eval.data"
f = open(fn,"w")
for x in ED:
    f.write(str(x))
    f.write("\n")
f.close()
ED = fn

print "Source machine is %s."%M
print "Changed machine has been written to %s."%MF
print "Goal predicates have been written to %s."%LGP
print "Evaluation data have been written to %s."%ED

print "Done."

