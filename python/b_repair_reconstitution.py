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

# Reconstituting atomic repairs to compound repairs in B models.
# Usage: python [this script] [source B model] [target B model with atomic repairs] [configuration file] [result folder]

# =================================================

# e.g. python src/python/b_repair_reconstitution.py TOSEM_Experiments/result_PE15_NDMI_IV10P_10/repaired_SKCART_ADD4_NDMI_IV10P_SD1001/source.mch TOSEM_Experiments/result_PE15_NDMI_IV10P_10/repaired_SKCART_ADD4_NDMI_IV10P_SD1001/result.mch TOSEM_Experiments/PE15_NDMI_IV10P_10/config/SKCART_config TOSEM_Experiments/result_PE15_NDMI_IV10P_10/repaired_SKCART_ADD4_NDMI_IV10P_SD1001/recon



start_time = time.time()

sd = 777

if len(sys.argv) != 5:
    print "Error: The number of input parameters should be 5."
    print "python [this script] [source B model] [target B model with atomic repairs] [configuration file] [result folder]"
    exit(1)


MS = sys.argv[1]
MA = sys.argv[2]
conffile = sys.argv[3]
resdir = sys.argv[4]

print "Source B Model:", MS
print "Target B Model with Atomic Repairs:", MA
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

print "Computing the state space of MA..."

TA = resdir + "/TA.txt"
max_initialisations = Bmch.read_config(conffile,"max_initialisations","int")
max_operations = Bmch.read_config(conffile,"max_operations","int")
bscopeA = Bmch.generate_training_set_condition(mchA)
oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MA,max_initialisations,max_operations,bscopeA,TA)
os.system(oscmd)

sgA = Bgenlib.BStateGraphForNN()
sgA.ReadStateGraph(TA)
TA = sgA.GetTransList()

print "Now Building MSA..."

SPA = RepSimpLib.extract_all_pre_states(TA)
initA = RepSimpLib.initialise_vble_by_examples(VList,SPA)
with open(MS) as mchf:
    mch1 = mchf.readlines()
mch1 = [x.strip() for x in mch1]
mchSA = Bmch.replace_initialisation(mch1,initA)

fn = "%s/MSA.mch"%resdir
f = open(fn,"w")
for x in mchSA:
    f.write(x)
    f.write("\n")
f.close()
MSA = fn

fn = resdir + "/MSA_pp.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,MSA)
os.system(oscmd)
MSA = fn

with open(MSA) as mchf:
    mchSA = mchf.readlines()
mchSA = [x.strip() for x in mchSA]

TSA = resdir + "/TSA.txt"
max_initialisationsSA = 10000 * Bmch.read_config(conffile,"max_initialisations","int")
max_operationsSA = Bmch.read_config(conffile,"max_operations","int")
bscopeSA = Bmch.generate_training_set_condition(mchSA)
oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MSA,max_initialisationsSA,max_operationsSA,bscopeSA,TSA)
os.system(oscmd)

sgSA = Bgenlib.BStateGraphForNN()
sgSA.ReadStateGraph(TSA)
TSA = sgSA.GetTransList()

# LM - list of atomic modifications
# LI - list of atomic insertions

LM = []
LI = []

print "Computing insertions and modifications..."

print len(TSA),len(TS),len(TA),len(Bmch.list_intersect(TS,TA))

TSS = TS + []

# get insertions and modifications
while True:

    # TAX - Transitions in TA and not in TS
    TAX = Bmch.list_difference(TA,TS)

    # if TS contains all transitions in TA, then break
    if TAX == []:
        break

    # TSX - Transitions in TS and not in TA
    TSX = Bmch.list_difference(TS,TA)

    # SS - States in TS
    SS = []
    for x in TS:
        SS.append(x[0])
        SS.append(x[2])
    SS = Bmch.list_union(SS,[])

    # Get insertions and modifications
    LIP = []
    LMP = []
    TN = [] # New state transitions in T
    TR = [] # Removed state transitions in T
    for x in TAX:
        flag = "NO"
        yt = "NONE"

        # modification
        for y in TSX:
            if y in TR: # y is already modified
                continue
            if x[0] == y[0] and x[1] == y[1]:
                flag = "MOD"
                yt = y
                break
        if flag == "MOD":
            LMP.append([yt[0],yt[1],yt[2],x[2]])
            TN.append(x)
            TR.append(yt)
            continue

        # insertion
        if x[0] in SS:
            flag = "INS"
            LIP.append(x)
            TN.append(x)
            continue

    LM = Bmch.list_union(LM,LMP)
    LI = Bmch.list_union(LI,LIP)

    # Update
    TCand = Bmch.list_difference(TSA,TS)
    while True:
        flag = False

        SNPost = [] # New post states
        for x in TN:
            SNPost.append(x[2])
        SNPost = Bmch.list_union(SNPost,[])
        for x in TCand:
            if x[0] in SNPost and not(x in TN):
                TN.append(x)
                flag = True
        if flag == False:
            break
    TS = Bmch.list_difference(TS,TR)
    TS = Bmch.list_union(TS,TN)






SType = sg.GetSetTypeFromTransList(TS)


print "Computing deletions"
LD = Bmch.list_difference(TS,TA)

print len(LM),len(LI),len(LD)


print "Repairing MS..."

# Try....
print "TA - TS"
for x in Bmch.list_difference(TA,TSS):
    print x
print "TS - TA"
for x in Bmch.list_difference(TSS,TA):
    print x
print "LM"
for x in LM:
    print x
print "PPPPPPPP"

# End try..


RMAll = LM
RIAll = LI
RDAll = LD

simpdir = resdir + "/repair_simplification/"
RISimp = []
RMSimp = []
RDSimp = []

"""
RMAll = RMAll[0:10]
RDAll = RDAll[0:10]
TA = TA[0:100] + RIAll
"""

if RIAll != []:
    print "Simplifying insertions..."
    TAE = TA + []
    RISimp = RepSimpLib.CFGInsertionSimplification(RIAll,TAE,VList,conffile,simpdir)

    for x in RISimp:
        print x

epid = "v%d"%(int(time.time()) % 10000)

VPF = ["_pre_%s"%epid,""]

if RMAll != []:
    print "Simplifying modifications..."
    TAE = Bmch.list_union(TA,RDAll)
    T1 = []
    T2 = []
    for x in RMAll:
        T1.append([x[0],x[1],x[2]])
        T2.append([x[0],x[1],x[3]])
    TAE = Bmch.list_difference(TAE,T2)
    TAE = Bmch.list_union(TAE,T1)
    RMSimp = RepSimpLib.CFGModificationSimplification(RMAll,TAE,VList,VPF,conffile,simpdir)

    for x in RMSimp:
        print x

if RDAll != []:
    print "Simplifying deletions..."
    TAE = Bmch.list_union(TA,RDAll)
    T1 = []
    T2 = []
    for x in RMAll:
        T1.append([x[0],x[1],x[2]])
        T2.append([x[0],x[1],x[3]])
    TAE = Bmch.list_difference(TAE,T2)
    TAE = Bmch.list_union(TAE,T1)
    RDSimp = RepSimpLib.CFGDeletionSimplification(RDAll,TAE,VList,VPF,conffile,simpdir)

    for x in RDSimp:
        print x

print "Applying simplified repairs..."
NM = 0
ND = 0
RMDSimp = []
RFL = []
for x in RMSimp + RDSimp:
    if not(x[0] in RFL):
        RFL.append(x[0])
for ope in RFL:
    CS = []
    for x in RMSimp:
        if x[0] == ope:
            CS = CS + x[1]
            NM = NM + len(x[1])
    for x in RDSimp:
        if x[0] == ope:
            CS = CS + x[1]
            ND = ND + len(x[1])
    RMDSimp.append([ope,CS])


 
# Apply compound repairs
# modifications and deletions
for R in RMDSimp:
    ope = R[0]
    RL = R[1]
    mch = Bmch.apply_modifications_and_deletions(mch,ope,RL,VList,epid)

# insertions
NI = 0
for R in RISimp:
    ope = R[0]
    rep = R[1]
    NI = NI + len(rep)
    mch = Bmch.apply_insertions(mch,ope,rep)

MC = resdir + "/MC.mch"
f = open(MC,"w")
for x in mch:
    f.write(x)
    f.write("\n")
f.close()

fn = resdir + "/result.mch"
cmd = "./../ProB/probcli -pp %s %s"%(fn,MC)
os.system(cmd)
MC = fn


cmd = "python src/python/state_graph_comparison.py %s %s %s/comparison/"%(MA,MC,resdir)
os.system(cmd)

LMA = Bmch.count_words(MA,resdir)
LMC = Bmch.count_words(MC,resdir)

print LMA,LMC

print ("*************** Reconstitution Done! ****************")

end_time = time.time()
elapsed_time = end_time - start_time

summary_text = []

summary_text.append("*************** SUMMARY ****************")

summary_text.append("Elapsed Time (s): %f."%elapsed_time)

summary_text.append("Length of the Original Machine: %s"%LMA)

summary_text.append("Length of the Resulting Machine: %s"%LMC)


summary_text.append("Number of Atomic Insertions: %s"%len(LI))
summary_text.append("Number of Compound Insertions: %s"%NI)

summary_text.append("Number of Atomic Modifications: %s"%len(LM))
summary_text.append("Number of Compound Modifications: %s"%NM)

summary_text.append("Number of Atomic Deletions: %s"%len(LD))
summary_text.append("Number of Compound Deletions: %s"%ND)

for x in summary_text: print x

resfile = resdir + "/RESULT"
f = open(resfile,"w")
for x in summary_text:
    f.write(x)
    f.write("\n")
f.close()



