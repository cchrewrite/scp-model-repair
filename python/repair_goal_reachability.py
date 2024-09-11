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

if G != "NONE":
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

if G != "NONE":
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


for x in RAAll: print x

NUM = len(RAAll)

RS = RepSimpLib.AtomicReachabilityRepair(RAAll,VList)

for x in RS: print x

for X in RS:
    ope = X[0]
    rep = X[1]
    mch = Bmch.apply_insertions(mch,ope,rep)

for X in mch: print X


MR = resdir + "/MR.mch"
f = open(MR,"w")
for X in mch:
    f.write(X)
    f.write("\n")
f.close()


fn = resdir + "/MR_pp.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,MR)
os.system(oscmd)
MR = fn

fn = resdir + "/result.mch"
cmd = "cp %s %s"%(MR,fn)
os.system(cmd)
MR = fn

M = sys.argv[1]
MdlType = W.MdlType


end_time = time.time()
elapsed_time = end_time - start_time



print "Source Machine: %s\n"%(M)
print "Type of Semantics Model: %s\n"%(MdlType)
print "Number of repairs: %d."%NUM
#print "Number of epochs: %d."%epoch
print "Elapsed Time (s): %f."%elapsed_time
print "Repaired Machine: %s\n"%(MR)

fn = resdir + "/summary"
logf = open(fn,"w")
logf.write("Source Machine: %s\n"%(M))
logf.write("Type of Semantics Model: %s\n"%(MdlType))
logf.write("Number of Repairs: %d\n"%(NUM))
#logf.write("Number of Epochs: %d\n"%(epoch))
logf.write("Elapsed Time (s): %f\n"%(elapsed_time))
logf.write("Repaired Machine: %s\n"%(MR))
logf.close()




