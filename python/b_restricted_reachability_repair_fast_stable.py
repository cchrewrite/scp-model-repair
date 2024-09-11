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
# Usage: python [this script] [source abstract machine] [goal file] [semantics model] [oracle (answer) file] [configuration file] [result folder]

# =================================================

start_time = time.time()

if len(sys.argv) != 7:
    print "Error: The number of input parameters should be 6."
    print "Usage: python [this script] [source abstract machine] [goal file] [semantics model] [oracle (answer) file] [configuration file] [result folder]"
    exit(1)



MS = sys.argv[1]
LGS = sys.argv[2]
W = sys.argv[3]
G = sys.argv[4]
conffile = sys.argv[5]
resdir = sys.argv[6]

print "Source Abstract Machine:", MS
print "Goal File:", LGS
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
f = open(LGS,"r")
for x in f.readlines():
    L.append(eval(x))

LGS = []
LGN = []
for X in L:
    LGS.append(X[0])
    LGN.append(X[1])
print LGS
print LGN

if True == True:

    print "====== DS = StateDiagram(MS) ======"

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

    print "====== DSG = DS + State-Diagram(Initialisation(MS,LGS)) ======"

    init1u = RepSimpLib.initialise_vble_by_examples(VList,LGS)
    with open(MS) as mchf:
        mch1 = mchf.readlines()
    mch1 = [x.strip() for x in mch1]
    mch1u = Bmch.replace_initialisation(mch1,init1u)
    fn = "%s/MG.mch"%resdir
    f = open(fn,"w")
    for x in mch1u:
        f.write(x)
        f.write("\n")
    f.close()
    MG = fn
    
    fn = resdir + "/MG_pp.mch"
    oscmd = "./../ProB/probcli -pp %s %s"%(fn,MG)
    os.system(oscmd)
    MG = fn

    with open(MG) as mchf:
        mch1 = mchf.readlines()
    mch1 = [x.strip() for x in mch1]


    DG = resdir + "/DG.txt"

    max_initialisations = Bmch.read_config(conffile,"max_initialisations","int")
    max_operations = Bmch.read_config(conffile,"max_operations","int")
    bscope1 = Bmch.generate_training_set_condition(mch1)
    oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MG,max_initialisations,max_operations,bscope1,DG)
    logtxt = os.popen(oscmd).read()
    print logtxt

    
    # "BinomialCoefficientConcurrent.mch" has CLPFD error. If CLPFD error occurs, use this.
    if "state_error" in logtxt:
        print "A CLPFD error occured. Disable CLPFD and re-make the state graph..."
        oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p CLPFD FALSE -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MG,max_initialisations,max_operations,bscope1,DG)
        logtxt = os.popen(oscmd).read()
        print logtxt
    

    sg1 = Bgenlib.BStateGraphForNN()
    sg1.ReadStateGraph(DG)
    DG = sg1.GetTransList()

    DSG = Bmch.list_union(DS,DG)



    if W == "NEW":
        print "====== Training a New Semantics Model ======"

        # Excluding negative training data that post states are goal states
        ED = []
        for X in LGS:
            ED.append(["ExcludedNegPostStates",X])

        sdir = resdir + "/SemMdlDir/"
        SemLearnLib.TrainingSemanticsModel([DSG,VList],conffile,sdir,learn_tails = True, excluded_data = ED)
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



    print "====== SX = SG + Initial-States(MS) + All-States(DSG)"
    X1 = sg.GetInitList()
    X2 = RepSimpLib.extract_all_states(DSG)
    SX = LGS + X1 + X2
    SX = Bmch.list_union(SX,[])
 
    print "====== TC = Candidate-Transitions(SX,Operations(DSG),SG) ======"
    print "====== R = Selected-Transitions(TC,W,NG,DSG) ======"

    LP = sg.GetAllOpeNames(DSG)
    wdir = resdir + "/SemProb/"
    R = []
    for i in xrange(len(LGS)):
        Q = LGS[i]
        N = LGN[i]
        TC = SemLearnLib.ComputeAllSemanticProbabilities(W,SX,LP,[Q],wdir)
        TC.sort(key = lambda x:x[1],reverse=True)
        # Select transitions.
        TCT = []
        NT = 0
        for X in DSG:
            if X[2] == Q:
                NT = NT + 1
        for X in TC:
            if NT >= N: break
            if not(X[0] in DSG):
                TCT.append(X)
                NT = NT + 1
        R.extend(TCT)

    RT = []
    for X in R:
        RT.append(X[0])

    R = Bmch.list_difference(RT,DSG)
    for X in R: print X

    print "====== DR = DSG + R ======"
    DR = Bmch.list_union(DSG,R)

    LRT = R
    DSU = DR
   
    if Bmch.check_set_order_in_transitions(DSU) == False:
        ppppppp 

    NUM = len(LRT)


    print "====== if LMT != None: ======"
    NRR = -1
    NUMSEL = -1
    if LMT != None:
        print "====== SACC = Accuracy(LRT,LMT) ======"
        """
        P = SemLearnLib.PredictUsingSemanticsModel(W,LRT,wdir)
        X = []
        for i in xrange(len(LRT)):
            X.append([LRT[i],P[i]])
        X.sort(key=lambda y:y[1], reverse=True)
        LRT = []
        for i in xrange(min(len(LMT),len(X))):
            LRT.append(X[i][0])
        """
        SACC = SemLearnLib.ReachabilityRepairAccuracyS2(LRT,LMT)
        print "Accuracy:",SACC
        NRR = len(LMT)
        NUMSEL = len(LRT)

    apply_reachability_repair_merging = Bmch.read_config(conffile,"apply_reachability_repair_merging","bool")

    if apply_reachability_repair_merging == True:
        PExp = LRT
        NExp = DSU


        f = open("pos.exp","w")
        f.write(str(PExp))
        f.close()
        f = open("neg.exp","w")
        f.write(str(NExp))
        f.close()
        
        """
        print "Positive Examples:" 
        print PExp
        print "Negative Examples:"
        print NExp 
        """

        SRLA = RepSimpLib.AtomicReachabilityRepair(LRT,VList)
        for X in SRLA: print X

        wdir = resdir + "/repair_simplification/"
        SRL = RepSimpLib.CFGReachabilityRepairSimplification(PExp,NExp,VList,conffile,wdir)
        #SRL = SRLA

    else:
        SRLA = "None"
        SRL = RepSimpLib.AtomicReachabilityRepair(LRT,VList)
        NUMSimp = -1

    if SRLA != "None":
        mchA = mch + []
        for X in SRLA:
            op = X[0]
            rep = X[1]
            mchA = Bmch.apply_S_change(mchA,op,rep)
    else:
        mchA = "None"


    NUMSubs = 0
    NUMCond = 0
    FOP = []
    for X in SRL:
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

    if mchA != "None":
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

