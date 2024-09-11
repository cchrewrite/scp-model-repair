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

# The Fast B-repair Model Repair Approach
# Usage: python [this script] [source abstract machine] [semantics model] [oracle] [configuration file] [result folder]

# =================================================

start_time = time.time()

if len(sys.argv) != 6:
    print "Error: The number of input parameters should be 5."
    print "Usage: python [this script] [source abstract machine] [semantics model] [oracle file] [configuration file] [result folder]"
    exit(1)



M = sys.argv[1]
W = sys.argv[2]
G = sys.argv[3]
conffile = sys.argv[4]
resdir = sys.argv[5]

print "Source Abstract Machine:", M
print "Semantics Model:", W
print "Oracle File:", G
print "Configuration File:", conffile
print "Result Folder:", resdir

print "Data Preparation..."

cmd = "mkdir %s"%resdir
os.system(cmd)

s = resdir + "/source.mch"
cmd = "cp %s %s"%(M,s)
os.system(cmd)
M = s


if W == "NEW":
    print "====== Training a New Semantics Model ======"
    sdir = resdir + "/SemMdlDir/"
    SemLearnLib.TrainingSemanticsModel(M,conffile,sdir)
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

print "====== MCSS <-- MonteCarloStateSampling(M) ======"

mcdir = resdir + "/MCSS/"
MCSS = SemLearnLib.MonteCarloStateSampling(M,W,conffile,mcdir)

print "====== MR <-- M ======"

sdir = resdir + "/0/"
cmd = "mkdir %s"%sdir
os.system(cmd)
MR = sdir + "/MR.mch"
cmd = "cp %s %s"%(M,MR)
os.system(cmd)

print "====== CMP <-- {} ======"
CMP = []

print "====== CMQ <-- {} ======"
CMQ = []

print "====== RS <-- {} ======"
RS = []

print "====== while True: ======"

epoch = 0
max_num_repair_epochs = Bmch.read_config(conffile,"max_num_repair_epochs","int")

INUM = 0
while True:

    print "====== D <-- StateDiagram(MR) ======"

    fn = sdir + "/MR_pp.mch"
    oscmd = "./../ProB/probcli -pp %s %s"%(fn,MR)
    os.system(oscmd)
    MR = fn

    with open(MR) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]

    D = sdir + "/D.txt"
    max_initialisations = Bmch.read_config(conffile,"max_initialisations","int")
    max_operations = Bmch.read_config(conffile,"max_operations","int")
    bscope = Bmch.generate_training_set_condition(mch)
    oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MR,max_initialisations,max_operations,bscope,D)
    os.system(oscmd)

    sg = Bgenlib.BStateGraphForNN()
    sg.ReadStateGraph(D)
    TL = sg.GetTransList()
    VList = sg.GetVbleList()

    print "====== SF <-- FaultyStates(D) + (ProhibitedStates(G) * AllStates(D)) ======"
    FSD = sg.GetStatesWithoutOutgoingTransitions(TL)
    PSG = [] # ProhibitedStates(D,G)
    ASD = RepSimpLib.extract_all_states(TL)
    SF = FSD + Bmch.list_intersect(PSG,ASD)
    SF = Bmch.remove_duplicate_elements(SF)

    print "====== TF <-- FaultyTransitions(D,SF) + (ProhibitedTransitions(D) * D) ======"
    FTDSF = sg.GetTransitionsWithPostStates(TL,SF)
    PTD = [] # ProhibitedTransitions(D)
    TF = FTDSF + Bmch.list_intersect(PTD,D)
    TF = Bmch.remove_duplicate_elements(TF)
    for x in TF: print x
     
    if TF == []:
        print "====== if TF is {} then break endif ======"
        print "No faulty transition found. The model has been repaired."
        break


    print "====== SREV <-- AllStates(D) + EnabledStates(G) + MonteCarloStateSampling(M) - ProhibitedStates(G) - SF ======"
    
    ESG = [] # EnabledStates(G)

    SREV = ASD + ESG + MCSS
    SREV = Bmch.list_difference(SREV,PSG)
    SREV = Bmch.list_difference(SREV,SF)
    SREV = Bmch.remove_duplicate_elements(SREV)

    print "====== RREV <-- Revision(TF,SREV) ======"
    RREV = Bmch.RevisionSynthesis(TF,SREV)
    #for x in RREV: print x

    print "====== RISO <-- Isolation(TF) ======"
    RISO = Bmch.IsolationSynthesis(TF)

    print "====== RSEL <-- RepairSelection(W, RISO + RREV) ======"
   
    revdir = sdir + "/revision/"
    cmd = "mkdir %s"%revdir
    os.system(cmd)
 
    REVP = SemLearnLib.ScoreRevisionsUsingSemanticsModel(W,RREV,revdir)
    for i in xrange(len(RREV)):
        RREV[i].append(REVP[i])
    #for x in RREV: print x
 
    RSEL = SemLearnLib.ClassifyAndSortRepairs(RREV + RISO)

    if "REVISION" in G[0] and "ISOLATION" in G[0]:

        print "====== CMP <-- CMP + CompareWithIRAnswer(RSEL,G) ======"
        X = SemLearnLib.CompareWithIRAnswer(RSEL,G,W.Threshold)
        RVA = X[0]
        IVA = X[1]
        CMP = CMP + RVA
        CMQ.append(IVA)

        print "====== IR <--- GetIRAnswer(RSEL,G) ======"
        IR = SemLearnLib.GetIRAnswer(RSEL,G)
        R = []
        for X in IR:
            if X[3] != "isolation":
                R.append(X)
        S = []
        for X in IR:
            if X[3] == "isolation":
                S.append(X)
        """
        print "====== RB <--- GetBestRepair(RSEL) ======"
        RB = SemLearnLib.GetBestRepair(RSEL)
        print "====== RS <--- RS + RB ======"
        RS = RS + RB
        """
        
    else:
        num_displayed_repairs = Bmch.read_config(conffile,"num_displayed_repairs","int")
        pppp

    R = Bmch.convert_changes_to_substitutions(R,VList)

    print "====== MR <-- Update(MR,R) ======"
    #subs_list = Bmch.convert_changes_to_substitutions(R,VList)
    for x in R:
        op = x[0]
        cond = x[1]
        subs = x[2]
        mch = Bmch.apply_A_change(mch,op,cond,subs)

    print "====== MR <-- Update(MR,S) ======"
    fn1 = resdir + "/" + str(epoch) + "/" + "/MR1.mch"
    fn2 = resdir + "/" + str(epoch) + "/" + "/MR1_pp.mch"
    Bmch.print_mch_to_file(mch,fn1)
    oscmd = "./../ProB/probcli -pp %s %s"%(fn2,fn1)
    os.system(oscmd)
    oscmd = "mv %s %s"%(fn2,fn1)
    os.system(oscmd)
    with open(fn1) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]

    INUM = INUM + len(S)
    FL = []
    for x in S:
        if not(x[1] in FL):
            FL.append(x[1])
    for op in FL:
        PS = []
        for x in S:
            if x[1] == op:
                PS.append(x[0])
        PS = Bmch.list_union(PS,[])

        cond = Bmch.convert_states_to_conditions(PS,VList)
        cond = "not(" + cond + ")"
        mch = Bmch.add_precond_to_mch(mch,op,cond)


    epoch = epoch + 1
    pdir = sdir + ""
    sdir = resdir + "/" + str(epoch) + "/"
    cmd = "mkdir %s"%sdir
    os.system(cmd)

    fn = sdir + "/MR.mch"
    Bmch.print_mch_to_file(mch,fn)
    MR = fn

    print "Epoch", epoch, "done. Resulting machine is", fn

    if epoch > max_num_repair_epochs:
        epoch = -1
        break

    #x = raw_input("press any symbol + Enter to continue.")



NumMRep = -1


fn = resdir + "/result.mch"
cmd = "cp %s %s"%(MR,fn)
os.system(cmd)
MR = fn
print "Model Repair Done. The repaired machine is %s."%fn

print "====== RACC <-- ComputeAccuracy(CMP)  ======"

RNUM = len(CMP)
RACC = 0.0
for x in CMP:
    RACC = RACC + SemLearnLib.RevisionVariableAccuracy(x[0],x[1])
RACC = RACC / RNUM

print "====== IACC <-- ComputeAccuracy(CMQ) ======"
NX = 0
IACC = 0.0

for x in CMQ:
    print x
   
    NX = NX + len(x[0]) + len(x[1])
    IACC = IACC + len(x[0])
IACC = IACC / NX

end_time = time.time()
elapsed_time = end_time - start_time

print "Number of isolation repairs is %d."%INUM
print "Number of revision repairs is %d."%RNUM
print "Number of merged repairs is %d."%NumMRep
print "Isolation Accuracy is %f."%IACC
print "Revision Accuracy is %f."%RACC
print "Number of epochs is %d."%epoch
print "Time of repair is %f."%elapsed_time

M = sys.argv[1]
MdlType = W.MdlType

fn = resdir + "/summary"
logf = open(fn,"w")
logf.write("Machine: %s\n"%(M))
logf.write("Type of Semantics Model: %s\n"%(MdlType))
logf.write("Number of Isolation Repairs: %d\n"%(INUM))
logf.write("Number of Revision Repairs: %d\n"%(RNUM))
logf.write("Number of Merged Repairs: %d\n"%(NumMRep))
logf.write("Isolation Accuracy: %f\n"%(IACC))
logf.write("Revision Accuracy: %f\n"%(RACC))
logf.write("Number of Epochs: %d\n"%(epoch))
logf.write("Elapsed Time (s): %f\n"%(elapsed_time))
logf.close()

