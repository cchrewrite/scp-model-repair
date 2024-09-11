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

fn = resdir + "/source_pp.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,M)
os.system(oscmd)

with open(fn) as mchf:
    mch_source = mchf.readlines()
mch_source = [x.strip() for x in mch_source]

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
RAll = []
SAll = []
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
    if epoch == 0:
        TL_source = TL + []
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

    """
    print "SAFSADFSA"
    for x in R: print x
    raw_input("sadfsafsa")
    for x in S: print x
    raw_input("sadfsafsa")
    """

    """

    f = open("TL.tmp","w")
    f.write(str(TL))
    f.close()
    f = open("R.tmp","w")
    f.write(str(R))
    f.close()
    f = open("VList.tmp","w")
    f.write(str(VList))
    f.close()

    exit()
 
    f = open("TL.tmp","r")
    TL = eval(f.readlines()[0])
    f.close()
    f = open("R.tmp","r")
    R = eval(f.readlines()[0])
    f.close()
    f = open("VList.tmp","r")
    VList = eval(f.readlines()[0])
    f.close()

    """



    print "====== MR <-- Update(MR,RMD) ======"
    RAll = RAll + R
    SAll = SAll + S
    # RMD --- list of atomic modifications and deletions
    RMD = R + S
    RMDS = []
    epid = "ep%d"%epoch
    for X in RMD:
        Y = Bmch.atomic_modification_or_deletion_to_conditional_substitution(X,VList,epid)
        RMDS.append(Y)

    FL = []
    for x in RMD:
        if not(x[1] in FL):
            FL.append(x[1])
        
    for ope in FL:
        RL = []
        for x in RMDS:
            if x[0] != ope:
                continue
            RL.append([x[1],x[2]])
        if RL == []: 
            continue
        mch = Bmch.apply_modifications_and_deletions(mch,ope,RL,VList,epid)

    INUM = INUM + len(S)

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




f = open("TL.tmp","w")
f.write(str(TL_source))
f.close()
f = open("R.tmp","w")
f.write(str(RAll))
f.close()
f = open("VList.tmp","w")
f.write(str(VList))
f.close()


# CFG simplification.
if Bmch.read_config(conffile,"apply_CFG_simplification","bool") == True:
    VPF = ["_pre_mod_simp",""]
    simpdir = resdir + "/repair_simplification/"
    RSimp = RepSimpLib.CFGModificationSimplification(RAll,TL_source,VList,VPF,conffile,simpdir)
    for x in RSimp:
        print x
        print "Here, I start working on the ICSE paper for repair simplification. After completing the paper, I will come back."
        raw_input("adf")
        

    mch_CFG = mch_source
    for X in RSimp:
        ope = X[0]
        RL = X[1]
        mch_CFG = Bmch.apply_modifications_and_deletions(mch_CFG,ope,RL,VList,"mod_simp")
    
    fn = resdir + "/result_CFG.mch"
    Bmch.print_mch_to_file(mch_CFG,fn)

pppp

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

