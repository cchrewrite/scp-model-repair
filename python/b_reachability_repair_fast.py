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


if W == "NEW":
    print "====== Training a New Semantics Model ======"
    sdir = resdir + "/SemMdlDir/"
    SemLearnLib.TrainingSemanticsModel(MS,conffile,sdir,learn_tails = True)
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
LMT = None
if "MISSING_TRANSITIONS" in G[0]:
    i = G[0].index("MISSING_TRANSITIONS")
    LMT = G[1][i]

print "====== XS = RandomSamplingWithInvariants(MS,NX) ======"

mcdir = resdir + "/MCSS/"
XS = SemLearnLib.MonteCarloStateSampling(MS,W,conffile,mcdir)

#epoch = 0
#max_num_repair_epochs = Bmch.read_config(conffile,"max_num_repair_epochs","int")

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

    print "====== SD = States(DS) ======"
    SD = RepSimpLib.extract_all_states(DS)

    print "====== SI = InitialStates(MS) ======"
    SI = sg.GetInitList()

    L = []
    f = open(LGS,"r")
    for x in f.readlines():
        L.append(eval(x))
    LGS = L + []
    
    print "====== XS = XS + SD + SI + LGS ======"
    #XS = XS[0:1000]
    XS = SI + LGS + SD + XS
    XS = Bmch.list_union(XS,[])

    flag = True
    for x in LGS:
        if not(x in SI) and not(x in SD):
            flag = False
            break
    if flag == True:
        print "All goal states are reachable!"
        #exit()

    print "====== LP = Operations(DS) ======"
    LP = sg.GetAllOpeNames(DS)

    for x in LP: print x

    wdir = resdir + "/SemProb/"
    print "====== BC = ClassificationBoundary(W,DS) ======"
    X = SemLearnLib.EstimateClassificationBoundary(W,DS,wdir)
    theta = Bmch.read_config(conffile,"boundary_theta","float")
    BC = X[0] - X[1] * theta
    BC = 0.85
    print BC
    VTU = []
    print "====== ST = LGS ======"
    ST = LGS + []
    print "====== STU = LGS ======"
    STU = LGS + []
    print "====== VTU = [] ======"
    VTU = []
    print "====== TSPA = [] ======"
    TSPA = []

    VTS = []

    
    print "====== for j = 0 to SSD do: ======"
    SSD = Bmch.read_config(conffile,"semantic_search_depth","int")
    SSD = 1
    for j in xrange(SSD):
        if ST == []: break
        print "====== TSP = SemanticProbability(W,XS,LP,ST) ======"
        #XS = XS[0:1000]
        sss1 = time.time()
        TSP = SemLearnLib.ComputeAllSemanticProbabilities(W,XS,LP,ST,wdir)
        sss2 = time.time()
        TSP.sort(key = lambda x:x[1],reverse=True)
        print "====== TSPA = TSPA + TSP ======"
        TSPA = Bmch.list_union(TSPA,TSP)

        ADC = Bmch.read_config(conffile,"apply_deduction_clustering","bool")
        if ADC == True:
            print "safdsafa"
            i = 0
            while i < len(TSP) and TSP[i][1] > BC - 0.2:
                i = i + 1
            TSPC = TSP[0:i]
            DC = RepSimpLib.DeductiveClustering(TSPC,VList,conffile,wdir)
            for X in DC: print X

            raw_input("DCDC")
            print "\n\n\n\n"

            AS = RepSimpLib.ComputeAlphaScore(DC)
            for X in AS: print X
            print len(AS)
            print AS[0]
            raw_input("CACA")
            
            for X in AS:
                for i in xrange(len(TSP)):
                    if TSP[i][0] == X[0]:
                        TSP[i][1] = TSP[i][1] + X[1]
                        break

            TSP.sort(key = lambda x:x[1],reverse=True)

        print "====== ValidTransitions(TSP,BC) ======"
        MaxNT = 100
        VT = SemLearnLib.ValidReachabilityTransitions(TSP,BC,MaxNT)
        """
        i = 0
        while i < len(TSP) and TSP[i][1] > BC:
            i = i + 1
        VT = TSP[0:i]
        """


        VTS = Bmch.list_union(VTS,VT)

        
        """
        print "use answer"
        VT = []
        for X in LMT:
            if X[2] in ST:
                VT.append([X,1.0])
        print "end"
        """

        print "====== VTU = VTU + VT ======"
        
        nn = 0
        for x in VT:
            print x
        print len(TSP)
        print sss2 - sss1

        VTU = Bmch.list_union(VTU,VT)

        print "====== ST = States(VT) - STU ======"
        VT0 = Bmch.get_list_column(VT,0)
        print "VT len ",len(VT)

        X = RepSimpLib.extract_all_states(VT0)
        ST = Bmch.list_difference(X,STU)
        print len(ST)

        print "====== STU = STU + States(VT) ======"
        STU = Bmch.list_union(STU,X)
        print "STU len ",len(STU)

  
    print "====== endfor ======"

    """
    print "use super answer"
    VTU = []
    for X in LMT:
        VTU.append([X,1.0])

    print "end"
    """

    print len(VTU)
    #raw_input("VTU")

    if Bmch.check_set_order_in_transitions(DS) == False:
        ppppppp

    print "====== DSU = UpdateReachability(SI,DS,VTU) ======"
    DSU = Bmch.update_reachability(SI,DS,VTU)
    if Bmch.check_set_order_in_transitions(DSU) == False:
        ppppppp

    print "====== LGSU = LGS - States(DSU) ======"
    X = RepSimpLib.extract_all_states(DSU)
    LGSU = Bmch.list_difference(LGS,X)
    print len(LGS),len(LGSU)

    #raw_input("LGSLGSU")

    print "====== if LGSU != []: ======"
    if LGSU != []:

        print "====== ETU = ShortestExtraPath(SI,DSU,TSPA,LGSU) ======"
        ETU = SemLearnLib.ShortestExtraPaths(SI,DSU,TSPA,LGSU)

        print "====== VTU = VTU + ETU ======"
        print len(VTU)
        VTU = Bmch.list_union(VTU,ETU)
        print len(VTU)
        #raw_input("pp")

        print "====== DSU = UpdateReachability(SI,DSU,VTU) ======"
        DSU = Bmch.update_reachability(SI,DSU,VTU)
    print "====== endif ======"
    
    print "====== SPT = PreStates(DSU) ======"
    SPT = RepSimpLib.extract_all_pre_states(DSU)

    print "====== MT = Initialisation(MS,SPT) ======"
    init1u = RepSimpLib.initialise_vble_by_examples(VList,SPT)
    with open(MS) as mchf:
        mch1 = mchf.readlines()
    mch1 = [x.strip() for x in mch1]
    mch1u = Bmch.replace_initialisation(mch1,init1u)
    fn = "%s/MT.mch"%resdir
    f = open(fn,"w")
    for x in mch1u:
        f.write(x)
        f.write("\n")
    f.close()
    MT = fn
    
    print "====== DT = StateDiagram(MT) ======"
    
    fn = resdir + "/MT_pp.mch"
    oscmd = "./../ProB/probcli -pp %s %s"%(fn,MT)
    os.system(oscmd)
    MT = fn

    with open(MT) as mchf:
        mch1 = mchf.readlines()
    mch1 = [x.strip() for x in mch1]

    DT = resdir + "/DT.txt"
    max_initialisations1 = 10000 * Bmch.read_config(conffile,"max_initialisations","int")
    max_operations1 = Bmch.read_config(conffile,"max_operations","int")
    bscope1 = Bmch.generate_training_set_condition(mch1)
    oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MT,max_initialisations1,max_operations1,bscope1,DT)
    os.system(oscmd)

    sg1 = Bgenlib.BStateGraphForNN()
    sg1.ReadStateGraph(DT)
    DT = sg1.GetTransList()

    print "====== LRT = DSU - DT ======"
    LRT = Bmch.list_difference(DSU,DT)
   
    if Bmch.check_set_order_in_transitions(DSU) == False:
        ppppppp 


    print len(DSU),len(DT),len(LRT)

    NUM = len(LRT)

    """
    LRT = []
    for X in VTS: LRT.append(X[0])
    NUM = len(LRT)
    """

    print "====== if LMT != None: ======"
    NRR = -1
    NUMSEL = -1
    if LMT != None:
        print "====== SACC = Accuracy(LRT,LMT) ======"
        P = SemLearnLib.PredictUsingSemanticsModel(W,LRT,wdir)
        X = []
        for i in xrange(len(LRT)):
            X.append([LRT[i],P[i]])
        X.sort(key=lambda y:y[1], reverse=True)
        for Y in X[0:20]: print Y
        LRT = []
        for i in xrange(min(len(LMT),len(X))):
            LRT.append(X[i][0])
        SACC = SemLearnLib.ReachabilityRepairAccuracy(LRT,LMT)
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

print "Number of required repair is %d."%NRR
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

