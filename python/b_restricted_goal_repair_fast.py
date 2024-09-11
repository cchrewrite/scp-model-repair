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

# The B Reachability Repair Approach
# Usage: python [this script] [source abstract machine] [goal predicate file] [semantics model] [oracle (answer) file] [configuration file] [result folder]

# =================================================

start_time = time.time()

if len(sys.argv) != 7:
    print "Error: The number of input parameters should be 6."
    print "Usage: python [this script] [source abstract machine] [goal predicate file] [semantics model] [oracle (answer) file] [configuration file] [result folder]"
    exit(1)



MS = sys.argv[1]
FGP = sys.argv[2]
W = sys.argv[3]
G = sys.argv[4]
conffile = sys.argv[5]
resdir = sys.argv[6]

print "Source Abstract Machine:", MS
print "Goal Predicate File:", FGP
print "Semantics Model:", W
print "Oracle File:", G
print "Configuration File:", conffile
print "Result Folder:", resdir

print "Data Preparation..."

cmd = "mkdir %s"%resdir
os.system(cmd)

s = resdir + "/source.mch"
cmd = "cp %s %s"%(MS,s)
os.system(cmd)
MS = s

s = resdir + "/oracle.txt"
cmd = "cp %s %s"%(G,s)
os.system(cmd)
G = s

s = resdir + "/config"
cmd = "cp %s %s"%(conffile,s)
os.system(cmd)
conffile = s

"""
fn = resdir + "/M_pp.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,M)
os.system(oscmd)
M = fn
"""

print "Reading Oracle File..."

G = Bmch.read_oracle_file(G)
LMT = None
if "MISSING_TRANSITIONS" in G[0]:
    i = G[0].index("MISSING_TRANSITIONS")
    LMT = G[1][i]


L = []
f = open(FGP,"r")
for x in f.readlines():
    L.append(eval(x))

# LGP - list of goal predicates
# LGN - list of goal numbers
LGP = []
LGN = []
for X in L:
    LGP.append(X[0])
    LGN.append(X[1])
print LGP
print LGN


print "====== DS <-- StateDiagram(MS) ======"

fn = resdir + "/MS_pp.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,MS)
os.system(oscmd)
MS = fn

with open(MS) as mchf:
    mch = mchf.readlines()
mch = [x.strip() for x in mch]

DS = resdir + "/DS.txt"
max_initialisations = Bmch.read_config(conffile,"max_initialisations","int")
max_operations = Bmch.read_config(conffile,"max_operations","int")
bscope = Bmch.generate_training_set_condition(mch)
oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MS,max_initialisations,max_operations,bscope,DS)
os.system(oscmd)

sg = Bgenlib.BStateGraphForNN()
sg.ReadStateGraph(DS)
TL = sg.GetTransList()
VList = sg.GetVbleList()
DS = TL


if W == "NEW":
    print "====== W <-- training a new semantics model ======"

    sdir = resdir + "/SemMdlDir/"
    SemLearnLib.TrainingSemanticsModel([DS,VList],conffile,sdir,learn_tails = True)
    cmd = "mv %s/semantics.mdl %s/semantics.mdl"%(sdir,resdir)
    os.system(cmd)
    s = resdir + "/semantics.mdl"
    W = s
else:
    s = resdir + "/semantics.mdl"
    cmd = "cp %s %s"%(W,s)
    os.system(cmd)
    W = s

filehandler = open(W,'r')
W = pickle.load(filehandler)




FL = []
for x in TL:
    if not(x[1] in FL):
        FL.append(x[1])

RAAll = []
RCAll = []

for i in xrange(len(LGP)):
    GP = LGP[i]
    GN = LGN[i]

    print "====== GS <-- solving the goal predicate GP using a constraint solver ======"

    mcdir = resdir + "/GSS/"
    GS = SemLearnLib.GOALStateSampling(MS,GP,W,conffile,mcdir)
    for x in GS: print x

    wdir = resdir + "/GeneInsSyn/"

    print "====== SINS1 <-- normal insertion synthesis to GS ======"
    SINS1 = Bmch.InsertionSynthesisForGoalPredicates(TL,FL,GS)

    if Bmch.read_config(conffile,"use_genetic_insertions","bool") == True:
        print "====== SINS2 <-- genetic insertion synthesis to GS ======"
        SINS2 = Bmch.GeneticInsertionSynthesisForGoalPredicates(TL,GP,GS,FL,W,MS,conffile,wdir)
    else:
        SINS2 = []

    print "====== SINS <-- SINS1 + SINS2 ======"

    SINS = Bmch.list_union(SINS1,SINS2)

    print "====== RA <-- rank and select the first GN insertions that are in SINS and not in TL. ======"
    prev_trans = ["GOAL","GOAL","GOAL"]
    for x in SINS: print x
    SINST = []
    for x in SINS:
        SINST.append(x[4])
    SINST = Bmch.list_difference(SINST,TL)
    RAP = Bmch.RankInsertions(prev_trans,SINST,W,conffile,wdir)
    RAP = RAP[0:min(GN,len(RAP))]
    RA = []
    for x in RAP:
        RA.append(x[4])

    RAAll = RAAll + RA

    for x in RA: print x


NUM = len(RAAll)
NRR = -1
NUMSEL = -1
if LMT != None:
    print "====== SACC = Accuracy(LRT,LMT) ======"
    SACC = SemLearnLib.GOALRepairAccuracyS2(RAAll,LMT)
    print "Accuracy:",SACC
    NRR = len(LMT)
    NUMSEL = len(RAAll)

apply_reachability_repair_merging = Bmch.read_config(conffile,"apply_reachability_repair_merging","bool")

if apply_reachability_repair_merging == True:
    print "====== RC <-- simplify RAAll ======"
    PExp = RAAll
    NExp = Bmch.list_union(TL,RAAll)

    f = open("pos.exp","w")
    f.write(str(PExp))
    f.close()
    f = open("neg.exp","w")
    f.write(str(NExp))
    f.close()
       
  

    RAS = RepSimpLib.AtomicReachabilityRepair(RAAll,VList)
    for X in RAS: print X
    raw_input("hhhhhh")

    wdir = resdir + "/repair_simplification/"
    RCS = RepSimpLib.CFGReachabilityRepairSimplification(PExp,NExp,VList,conffile,wdir)


else:
    RCS = "None"
    RAS = RepSimpLib.AtomicReachabilityRepair(RAAll,VList)
    NUMSimp = -1

# mchA - repaired machine with atomic repairs
mchA = mch + []
for X in RAS:
    op = X[0]
    rep = X[1]
    mchA = Bmch.apply_S_change(mchA,op,rep)

if RCS != None:
    RS = RCS
else:
    RS = RAS

NUMSubs = 0
NUMCond = 0
FOP = []
for X in RS:
    op = X[0]
    rep = X[1]
    mch = Bmch.apply_S_change(mch,op,rep)
    for Y in rep:
        NUMCond = NUMCond + len(Y[0].split(" or "))
    NUMSubs = NUMSubs + len(rep)
    if not(op in FOP):
        FOP.append(op)
for X in mch: print X

NUMFOP = len(FOP)

MR = resdir + "/result.mch"
f = open(MR,"w")
for X in mch:
    f.write(X)
    f.write("\n")
f.close()


MRA = resdir + "/resultA.mch"
f = open(MRA,"w")
for X in mchA:
    f.write(X)
    f.write("\n")
f.close()
wdir = resdir + "/comparison/"
cmd = "python src/python/state_graph_comparison.py %s %s %s"%(MRA,MR,wdir)
os.system(cmd)

end_time = time.time()
elapsed_time = end_time - start_time

print "Number of required repairs is %d."%NRR
print "Number of suggested repairs is %d."%NUM
print "Number of selected repairs is %d."%NUMSEL
print "Number of changed operations is %d."%NUMFOP
print "Number of simplified preconditions is %d."%NUMCond
print "Number of simplified substitutions is %d."%NUMSubs
print "Accuracy is %f."%SACC
print "Time of repair is %f."%elapsed_time

M = sys.argv[1]
MdlType = W.MdlType

fn = resdir + "/summary"
logf = open(fn,"w")
logf.write("Machine: %s\n"%(M))
logf.write("Type of Semantics Model: %s\n"%(MdlType))
logf.write("Number of Required Repairs: %d\n"%(NRR))
logf.write("Number of Suggested Repairs: %d\n"%(NUM))
logf.write("Number of Selected Repairs: %d\n"%(NUMSEL))
logf.write("Number of Changed Operations: %d\n"%(NUMFOP))
logf.write("Number of Simplified Preconditions: %d\n"%(NUMCond))
logf.write("Number of Simplified Substitutions: %d\n"%(NUMSubs))
logf.write("Reachability Repair Accuracy: %f\n"%(SACC))
logf.write("Elapsed Time (s): %f\n"%(elapsed_time))
logf.close()

