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

print "**************** Repairing deadlock states using modifications / deletions / insertions *****************"

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

print "====== RS <-- {} ======"
RS = []
RMAll = []
RDAll = []
RIAll = []

print "====== while True: ======"

epoch = 0
max_num_repair_epochs = Bmch.read_config(conffile,"max_num_repair_epochs","int")

while True:

    break
    print "====== D <-- StateDiagram(MR) ======"

    fn = sdir + "/MR_pp.mch"
    oscmd = "./../ProB/probcli -pp %s %s"%(fn,MR)
    os.system(oscmd)
    MR = fn

    with open(MR) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]

    ex_ope = [";","Good_States = "] + Bmch.generate_good_state_detection_operation(mch)
    for x in ex_ope: print x
    
    i = len(mch)-1
    while not("END" in mch[i]):
        i = i - 1
    mch_ex = mch[0:i] + ex_ope + mch[i:len(mch)]
    for x in mch_ex: print x
   
    fn = sdir + "/MR_ex.mch"
    MR_ex = open(fn,"w")
    for x in mch_ex:
        MR_ex.write(x + "\n")
    MR_ex.close()
    MR_ex = fn

    bscope = ""
    for i in xrange(3,len(ex_ope)-3):
        bscope = bscope + ex_ope[i] + " "

    D = sdir + "/D.txt"
    max_initialisations = Bmch.read_config(conffile,"max_initialisations","int")
    max_operations = Bmch.read_config(conffile,"max_operations","int")
    #bscope = Bmch.generate_training_set_condition(mch)
    oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MR_ex,max_initialisations,max_operations,bscope,D)
    os.system(oscmd)

    sg = Bgenlib.BStateGraphForNN()
    sg.ReadStateGraph(D)
    TL = sg.GetTransList()
    VList = sg.GetVbleList()

    # GS --- good states
    ASD = RepSimpLib.extract_all_states(TL)
    GS = []
    TLP = []
    for x in TL:
        if x[1] == "Good_States":
            GS.append(x[0])
        else:
            TLP.append(x)
    GS = Bmch.list_union(GS,[])

    # FS --- states violating invariant & assertions. These states will be repaired using modifications and deletions.
    FS = Bmch.list_difference(ASD,GS)

    # DDLS --- deadlock states that do not violate invariant & assertions. These states will be repaired using modifications, deletions and insertions.
    FSP = sg.GetStatesWithoutOutgoingTransitions(TLP)
    DDLS = Bmch.list_difference(FSP,FS)

    NFS = len(FS)   
    NDDLS = len(DDLS)
    print "Number of states violating invariant & assertions is :%d.\n"%(NFS)
    print "Number of deadlock states that do not violate invariant & assertions is :%d.\n"%(NDDLS)
 
    TL = TLP 
    FL = []
    for x in TL:
        if not(x[1] in FL):
            FL.append(x[1])
    if epoch == 0:
        TL_source = TL + []

    print "====== SF <-- DeadlockStates(D)  ======"

    #FSD = sg.GetStatesWithoutOutgoingTransitions(TL)
    FSD = DDLS

    if FSD == []:
        break

    PSG = []
    ASD = RepSimpLib.extract_all_states(TL)    
    SF = FSD
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

    SINS = SREV

    print "====== RREV <-- Revision(TF,SREV) ======"
    RREV1 = Bmch.RevisionSynthesis(TF,SREV)
    RINS1 = Bmch.InsertionSynthesisForDeadlocks(TF,FL,SINS)

    if Bmch.read_config(conffile,"use_genetic_modifications","bool") == True:
        genedir = sdir + "/genetic/"
        cmd = "mkdir %s"%genedir
        os.system(cmd)
        RREV2 = Bmch.GeneticRevisionSynthesis(TF,W,M,conffile,genedir)
        RINS2 = Bmch.GeneticInsertionSynthesisForDeadlocks(TF,FL,W,M,conffile,genedir)
    else:
        RREV2 = []
        RINS2 = []


    RREV = Bmch.list_union(RREV1,RREV2)
    RINS = Bmch.list_union(RINS1,RINS2)
    #for x in RREV: print x

    print "====== RISO <-- Isolation(TF) ======"
    RISO = Bmch.IsolationSynthesis(TF)

    print "====== RSEL <-- RepairSelection(W, RISO + RREV + RINS) ======"
   
    revdir = sdir + "/revision/"
    cmd = "mkdir %s"%revdir
    os.system(cmd)
 
    REVP = SemLearnLib.ScoreRevisionsUsingSemanticsModel(W,RREV,revdir)
    INSP = SemLearnLib.ScoreInsertionsUsingSemanticsModel(W,RINS,revdir)
    
    for i in xrange(len(RREV)):
        RREV[i].append(REVP[i])
    for i in xrange(len(RINS)):
        RINS[i].append(INSP[i])
    for i in xrange(len(RISO)):
        RISO[i].append("isolation")
        RISO[i].append(0.5)
 
    RSEL = SemLearnLib.ClassifyAndSortIMDRepairs(RREV + RISO + RINS)

   
    print "====== R <--- GetBestRepair(RSEL,TL) ======"
    R = SemLearnLib.GetBestIMDRepair(RSEL,TL)
    print "====== RS <--- RS + RB ======"
    RS = RS + R

    # RM --- modification repairs.
    # RD --- deletion repairs.
    # RI --- insertion repairs.

    RM = []
    RD = []
    RI = []
    for X in R:
        if X[3] == "isolation":
            Y = [X[0],X[1],X[2],X[4]]
            RD.append(Y)
        elif X[3] == "revision":
            Y = [X[0],X[1],X[2],X[4]]
            RM.append(Y)
        elif X[3] == "insertion":
            RI.append(X[4])
        else:
             pppp

    RMAll = RMAll + RM
    RDAll = RDAll + RD
    RIAll = RIAll + RI

    epid = "v%d"%(int(time.time()) % 10000)
    RMD = RM + RD
    RMDS = []
    for X in RMD:
        Y = Bmch.atomic_modification_or_deletion_to_conditional_substitution(X,VList,epid)
        RMDS.append(Y)

    RIS = RepSimpLib.AtomicReachabilityRepair(RI,VList)

    print "====== MR <-- Update(MR,R) ======"


    # apply insertions
    for X in RIS:
        op = X[0]
        rep = X[1]
        mch = Bmch.apply_insertions(mch,op,rep)

    """
    # rewrite the machine to pretty-printed format
    fn = sdir + "/MR.mch"
    Bmch.print_mch_to_file(mch,fn)
    MR = fn
    fn = sdir + "/MR_pp.mch"
    oscmd = "./../ProB/probcli -pp %s %s"%(fn,MR)
    os.system(oscmd)
    MR = fn
    with open(MR) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]

    raw_input("check here")
    """

    # apply modifications and deletions
    for ope in FL:
        RL = []
        for x in RMDS:
            if x[0] != ope:
                continue
            RL.append([x[1],x[2]])
        if RL == []:
            continue
        mch = Bmch.apply_modifications_and_deletions(mch,ope,RL,VList,epid)

    epoch = epoch + 1
    pdir = sdir + ""
    sdir = resdir + "/" + str(epoch) + "/"
    cmd = "mkdir %s"%sdir
    os.system(cmd)

    fn = sdir + "/MR.mch"
    Bmch.print_mch_to_file(mch,fn)
    MR = fn

    raw_input("check here")

    print "Epoch", epoch, "done. Resulting machine is", fn

    if epoch > max_num_repair_epochs:
        epoch = -1
        break

    #x = raw_input("press any symbol + Enter to continue.")

"""

f = open("TL_source.tmp","w")
f.write(str(TL_source))
f.close()
f = open("RMAll.tmp","w")
f.write(str(RMAll))
f.close()
f = open("RDAll.tmp","w")
f.write(str(RDAll))
f.close()
f = open("RIAll.tmp","w")
f.write(str(RIAll))
f.close()
f = open("VList.tmp","w")
f.write(str(VList))
f.close()
"""


f = open("TL_source.tmp","r")
TL_source = eval(f.readlines()[0])
f.close()
f = open("RMAll.tmp","r")
RMAll = eval(f.readlines()[0])
f.close()
f = open("RDAll.tmp","r")
RDAll = eval(f.readlines()[0])
f.close()
f = open("RIAll.tmp","r")
RIAll = eval(f.readlines()[0])
f.close()
f = open("VList.tmp","r")
VList = eval(f.readlines()[0])
f.close()



simp_flag = Bmch.read_config(conffile,"apply_repair_simplification","bool")

# The following two statements should be removed.
simp_flag = True
TL = TL_source + RIAll

if simp_flag == True:

    raw_input("start_simp")

    #VPF = ["_pre_mod_simp",""]
    VPF = ["_v%d"%(int(time.time()) % 10000),""]
    simpdir = resdir + "/repair_simplification/"
    RISimp = None
    RMSimp = None
    RDSimp = None

    if RIAll != []:
        
        RISimp = RepSimpLib.CFGInsertionSimplification(RIAll,TL,VList,conffile,simpdir)

    for x in RISimp:
        print x
    raw_input("simp")

    if RMAll != []:
        RMSimp = RepSimpLib.CFGModificationSimplification(RMAll,TL_source,VList,VPF,conffile,simpdir)

    
    
    for x in RMSimp:
        print x

    


    raw_input("end_simp")



    pppp
    print "====== RCFG <--- CFGRepairSimplification(RS,D) ======"
    cgfdir = sdir + "/CFGGeneration/"
    RCFG = RepSimpLib.CFGRepairSimplification(RS,TL,VList,conffile,cgfdir)
    NumMRep = len(RCFG)
    #RCFG = Bmch.convert_changes_to_substitutions(RCFG,VList)

    print "====== MCFG <--- Update(M,RCFG) ======"

    M = resdir + "/0/MR_pp.mch"
    with open(M) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]
    
    for x in RCFG:
        op = x[0]
        cond = x[1]
        subs = x[2]
        mch = Bmch.apply_A_change(mch,op,cond,subs)

    fn = resdir + "/result_CFG.mch"
    Bmch.print_mch_to_file(mch,fn)

else:
    NumMRep = -1


fn = resdir + "/result.mch"
cmd = "cp %s %s"%(MR,fn)
os.system(cmd)
MR = fn
print "Model Repair Done. The repaired machine is %s."%fn

print "====== ACC <-- ComputeAccuracy(CMP)  ======"

NUM = len(CMP)
ACC = 0.0
for x in CMP:
    ACC = ACC + SemLearnLib.RevisionVariableAccuracy(x[0],x[1])
ACC = ACC / NUM

end_time = time.time()
elapsed_time = end_time - start_time

print "Number of repairs is %d."%NUM
print "Number of merged repairs is %d."%NumMRep
print "Accuracy is %f."%ACC
print "Number of epochs is %d."%epoch
print "Time of repair is %f."%elapsed_time

M = sys.argv[1]
MdlType = W.MdlType

fn = resdir + "/summary"
logf = open(fn,"w")
logf.write("Machine: %s\n"%(M))
logf.write("Type of Semantics Model: %s\n"%(MdlType))
logf.write("Number of Repairs: %d\n"%(NUM))
logf.write("Number of Merged Repairs: %d\n"%(NumMRep))
logf.write("Modification Accuracy: %f\n"%(ACC))
logf.write("Number of Epochs: %d\n"%(epoch))
logf.write("Elapsed Time (s): %f\n"%(elapsed_time))
logf.close()

