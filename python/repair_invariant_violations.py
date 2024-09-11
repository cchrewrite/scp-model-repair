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

print "**************** Repairing invariant violations using modifications / deletions *******************"

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

    learn_tails = Bmch.read_config(conffile,"learn_tails","bool")
    
    if type(learn_tails) == type("string_type"):
        learn_tails = False
       
    #print learn_tails, type(learn_tails)
    #raw_input("asdf")
    SemLearnLib.TrainingSemanticsModel(M, conffile, sdir, learn_tails = learn_tails)
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

print "====== RS <-- {} ======"
RS = []

RMAll = []
RDAll = []

print "====== while True: ======"

epoch = 0
max_num_repair_epochs = Bmch.read_config(conffile,"max_num_repair_epochs","int")

while True:

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

    # In configuration file, we can decide whether to repair deadlock states using modifications and deletions.
    repair_deadlocks = Bmch.read_config(conffile,"repair_deadlocks","bool")
    if type(repair_deadlocks) == type("string_type"):
        repair_deadlocks = False
    if repair_deadlocks == True:
        FS = FS + DDLS
        
    TL = TLP

    FL = []
    for x in TL:
        if not(x[1] in FL):
            FL.append(x[1])
    if epoch == 0:
        TL_source = TL + []

    print "====== SF <-- FaultyStates(D) + (ProhibitedStates(G) * AllStates(D)) ======"

    #FSD = sg.GetStatesWithoutOutgoingTransitions(TL)
    FSD = FS
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
    RREV1 = Bmch.RevisionSynthesis(TF,SREV)
    if Bmch.read_config(conffile,"use_genetic_modifications","bool") == True:
        genedir = sdir + "/genetic/"
        cmd = "mkdir %s"%genedir
        os.system(cmd)
        RREV2 = Bmch.GeneticRevisionSynthesis(TF,W,M,conffile,genedir)
    else:
        RREV2 = []

    for x in RREV1: print x
    #raw_input("sdfas")
    for x in RREV2: print x
    #raw_input("sadfsa")

    RREV = Bmch.list_union(RREV1,RREV2)
    #for x in RREV: print x

    print "====== RISO <-- Isolation(TF) ======"
    RISO = Bmch.IsolationSynthesis(TF)
    #for x in RISO: print x

    print "====== RSEL <-- RepairSelection(W, RISO + RREV) ======"
   
    revdir = sdir + "/revision/"
    cmd = "mkdir %s"%revdir
    os.system(cmd)
 
    REVP = SemLearnLib.ScoreRevisionsUsingSemanticsModel(W,RREV,revdir)
    for i in xrange(len(RREV)):
        RREV[i].append(REVP[i])


    #for x in RREV: print x

    DelSP = Bmch.read_config(conffile,"deletion_boundary","float")
    if type(DelSP) == type("string_type"):
        DelSP = 0.5

    for i in xrange(len(RISO)):
        RISO[i].append("isolation")
        RISO[i].append(DelSP)
   
    MaxDist = Bmch.read_config(conffile,"max_absolute_distance","int")
    if type(MaxDist) == type("string_type"):
        MaxDist = None

    RSEL = SemLearnLib.ClassifyAndSortIMDRepairs(RREV + RISO, MaxDist = MaxDist)


    print "====== R <--- GetBestRepair(RSEL,TL) ======"
    R = SemLearnLib.GetBestIMDRepair(RSEL,TL)
    
    print "====== RS <--- RS + RB ======"
    RS = RS + R



    
    # RM --- modification repairs.
    # RD --- deletion repairs.

    RM = []
    RD = []
    for X in R:
        if X[3] == "isolation":
            Y = [X[0],X[1],X[2],X[4]]
            RD.append(Y)
        elif X[3] == "revision":
            Y = [X[0],X[1],X[2],X[4]]
            RM.append(Y)
        else:
             pppp

    RMAll = RMAll + RM
    RDAll = RDAll + RD


    epid = "v%d"%(int(time.time()) % 10000)
    RMD = RM + RD
    RMDS = []
    for X in RMD:
        Y = Bmch.atomic_modification_or_deletion_to_conditional_substitution(X,VList,epid)
        RMDS.append(Y)

    print "====== MR <-- Update(MR,R) ======"

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

    
    print "Epoch", epoch, "done. Resulting machine is", fn

    if epoch > max_num_repair_epochs:
        epoch = -1
        break

NUM = len(RMAll) + len(RDAll)
M = sys.argv[1]
MdlType = W.MdlType

fn = resdir + "/result.mch"
cmd = "cp %s %s"%(MR,fn)
os.system(cmd)
MR = fn

end_time = time.time()
elapsed_time = end_time - start_time


print "Source Machine: %s\n"%(M)
print "Type of Semantics Model: %s\n"%(MdlType)
print "Number of repairs: %d."%NUM
print "Number of epochs: %d."%epoch
print "Elapsed Time (s): %f."%elapsed_time
print "Repaired Machine: %s\n"%(MR)


fn = resdir + "/summary"
logf = open(fn,"w")
logf.write("Source Machine: %s\n"%(M))
logf.write("Type of Semantics Model: %s\n"%(MdlType))
logf.write("Number of Repairs: %d\n"%(NUM))
logf.write("Number of Epochs: %d\n"%(epoch))
logf.write("Elapsed Time (s): %f\n"%(elapsed_time))
logf.write("Repaired Machine: %s\n"%(MR))
logf.close()

