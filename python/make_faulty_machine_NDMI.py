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
# It makes a faulty B machine by randomly modifying current transitions to faulty transitions / inserting faulty transitions (randomly decided with a probability of 0.5 vs 0.5). The faulty transitions must violate invariants, and can be either deterministic or non-deterministic.
# Input - source abstract machine, percentage of faulty transitions, seed
# Output - result abstract machine (result.mch) and evaluation data (eval.data)
# Usage: python [this script] [source abstract machine] [percentage of faulty transitions] [seed] [result folder]

# =================================================

if len(sys.argv) != 5:
    print "Error: The number of input parameters should be 5."
    print "Usage: python [this script] [source abstract machine] [percentage of faulty transitions] [seed] [result folder]"
    exit(1)



M = sys.argv[1]
PN = sys.argv[2]
SD = sys.argv[3]
resdir = sys.argv[4]

print "Making a faulty abstract machine..."

print "Source Abstract Machine:", M
print "Percentage of Faulty Transitions:", PN
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

#for x in D: print x
#x = raw_input("ppp")

print "====== S <-- AllStates(D) ======"

S = RepSimpLib.extract_all_states(D)

print "Machine:", sys.argv[1]
print "# States:", len(S)
print "# State Transitions:", len(D)
raw_input("safsad")

N = int(len(D) * PN)

print "====== Printing Evaluation Data... ======"

ED = ["BEGIN_DESIRED_STATE_TRANSITIONS"]
ED.append("FORMAT: [ < Pre-state > , < Operation > , < Post-state > ]")
DST = D + []
DST.sort()
ED = ED + DST
ED.append("END_DESIRED_STATE_TRANSITIONS")


print "====== Z <-- GenerateProhibitedStates(S,N,SD) ======"

# SD has been set before.
Z = Bmch.generate_prohibited_states(S,N)

#x = raw_input("ppp")

print "====== PSI <-- ConvertProhibitedStatesToInvariants(Z) ======"
VList = sg.GetVbleList()
PSI = Bmch.convert_prohibited_states_to_invariants(Z,VList)

print "====== MF <-- AddInvariants(M,PSI) ======"

MF = Bmch.add_invariant_to_machine(mch,PSI)

print "====== RD <-- D ======"
#RD = sg.FindDeterministicTransitions(D)
RD = D + []

print "Number of Prohibited States:",len(Z)
print "Number of Transitions:",len(RD)

print "====== RD <-- Shuffle(RD,SD) ======"

# SD has been set before.
random.shuffle(RD)

print "====== RD <-- FirstNItems(RD,N) ======"

if N > len(RD):
    print "Error: N is greater than the number of transitions in the original state diagram!"
    RaiseAnError()
RD = RD[0:N]

#for x in RD: print x
#x = raw_input("pppppp")

print "====== LM <-- RandomlyGenerateModifications(RD,Z,SD) ======"

print "====== LI <-- RandomlyGenerateInsertions(RD,Z,SD) ======"

FL = []
for x in D:
    if not(x[1] in FL):
        FL.append(x[1])

LM = []
LI = []
#for x in RD:
    #if random.random() >= 0.5:
for k in xrange(len(RD)):
    x = RD[k]
    if k < len(RD) / 2:
       i = int(random.random() * len(Z))
       y = x + [Z[i]]
       LM.append(y)
    else:
       i = int(random.random() * len(Z))
       j = int(random.random() * len(FL))
       y = [x[2],FL[j],Z[i]]
       LI.append(y)


print "====== MF <-- ApplyChangesToMachine(MF,LM,LI) ======"

epid = "mkdt0"

RM = []
for x in LM:
    y = Bmch.atomic_modification_or_deletion_to_conditional_substitution(x,VList,epid)
    RM.append(y)

for ope in FL:
    RL = []
    for x in RM:
        if x[0] == ope:
            RL.append([x[1],x[2]])
    if RL != []:
        MF = Bmch.apply_modifications_and_deletions(MF,ope,RL,VList,epid)

  


RI = RepSimpLib.AtomicReachabilityRepair(LI,VList)
for x in RI:
    ope = x[0]
    rep = x[1]
    MF = Bmch.apply_insertions(MF,ope,rep)

fn = resdir + "/MF.mch"
Bmch.print_mch_to_file(MF,fn)
MF = fn

fn = resdir + "/result.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,MF)
os.system(oscmd)
MF = fn

s = resdir + "/source.mch"
cmd = "cp %s %s"%(M,s)
os.system(cmd)
M = s


ED.append("BEGIN_INSERTED_STATE_TRANSITIONS")
for x in LI:
    ED.append(x)
ED.append("END_INSERTED_STATE_TRANSITIONS")

ED.append("BEGIN_MODIFIED_STATE_TRANSITIONS")
for x in LM:
    ED.append(x)
ED.append("END_MODIFIED_STATE_TRANSITIONS")

ED.append("BEGIN_GENERAL_INFO")
ED.append(["Number of States in the Source Machine",len(S)])
ED.append(["Number of State Transitions in the Source Machine",len(D)])
ED.append(["Number of Inserted State Transitions",len(LI)])
ED.append(["Number of Modified State Transitions",len(LM)])
ED.append("END_GENERAL_INFO")


fn = resdir + "/eval.data"
f = open(fn,"w")
for x in ED:
    f.write(str(x))
    f.write("\n")
f.close()
ED = fn




print "Source machine is %s."%M
print "Changed machine has been written to %s."%MF
print "Evaluation data have been written to %s."%ED

print "Done."

