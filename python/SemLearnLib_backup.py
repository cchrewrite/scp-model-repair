import sys
import Bmch
import os
import time
import Bgenlib
import random
import gc
from nnet.nnetlib import *
from Cartlib import *
from NBayes import *
from SKCART import *
from SKClassifier import *
import numpy
import logging
import pickle
import json
from sklearn.metrics import roc_auc_score
import numpy
import networkx as nx

# ==================================================

# This is a library for semantic learning

# =================================================


def ValidReachabilityTransitions(TSP,BC,MaxNT): 
    i = 0
    LG = []
    while i < len(TSP) and TSP[i][1] > BC:
        i = i + 1
        if i > 0 and TSP[i][1] > TSP[i-1][1]:
            print "Error: TSP is not sorted."
            ppp
        if not(TSP[i][0][2]) in LG:
            LG.append(TSP[i][0][2])
    VT = TSP[0:i]    

    NTL = []
    for i in xrange(len(LG)):
        NTL.append(0)
    
    res = []
    for X in VT:
        idx = LG.index(X[0][2])
        if NTL[idx] > MaxNT:
            continue
        NTL[idx] = NTL[idx] + 1
        res.append(X)

    return res

def ShortestExtraPaths(SI,DS,TSP,GS):

    S = ["source"]
    for X in SI:
        if not(X in S):
            S.append(X)
    for X in DS:
        if not(X[0] in S):
            S.append(X[0])
        if not(X[2] in S):
            S.append(X[2])
    for Y in TSP:
        X = Y[0]
        if not(X[0] in S):
            S.append(X[0])
        if not(X[2] in S):
            S.append(X[2])
    G = nx.DiGraph()
    for X in SI:
        P = S.index("source")
        Q = S.index(X)
        W = 0.0
        G.add_edge(P,Q,weight=W)
    for X in DS:
        P = S.index(X[0])
        Q = S.index(X[2])
        W = 0.0
        G.add_edge(P,Q,weight=W)
    for Y in TSP:
        X = Y[0]
        P = S.index(X[0])
        Q = S.index(X[2])
        W = 1.0 - Y[1]
        G.add_edge(P,Q,weight=W)

    RS = []
    for T in GS:
        P = S.index("source")
        Q = S.index(T)
        R = nx.dijkstra_path(G, source=P, target=Q)
        print R
        print nx.dijkstra_path_length(G, source=P, target=Q)
        
        for i in xrange(len(R)-1):
            P = S[R[i]]
            if P == "source": continue
            Q = S[R[i+1]]
            W = 10000
            V = None
            print P,Q
            for X in DS:
                WX = 0.0
                if WX < W and X[0] == P and X[2] == Q:
                    W = WX
                    V = X
                    break
            if V != None:
                continue
            for Y in TSP:
                X = Y[0]
                WX = 1.0 - Y[1]
                if WX < W and X[0] == P and X[2] == Q:
                    W = WX
                    V = X
            if V == None:
                print "Error!"
                ppppp
            if not(V in RS):
                RS.append([V,1.0-W])
        print R

    for x in RS: print x

    return RS



# accuracy = number of aligned values / number of required values
# S --- answers to be evaluated
# T --- standard answers
def ReachabilityRepairAccuracy(S,T):
    P = S
    Q = T

    # number of values = number of variables * 2 + 1 (1 is the name of the operation)
    NumV = len(Q[0][0]) * 2 + 1

    # total number of values
    TNumV = len(Q) * NumV

    P = P + []
    Q = Q + []
    S = 0.0
    for X in P:
        Z = ""
        DXZ = -1
        for Y in Q:
            dt = 0.0
            XT = X[0] + [X[1]] + X[2]
            YT = Y[0] + [Y[1]] + Y[2]
            if len(XT) != NumV or len(YT) != NumV:
                print "Error!!!"
                ppp
            for i in xrange(NumV):
                if XT[i] == YT[i]:
                    dt = dt + 1.0
            if dt > DXZ:
                DXZ = dt
                Z = Y
        if DXZ == -1:
            continue
        S = S + DXZ
        Q.remove(Z)
    Acc = S / TNumV
    
    return Acc




# accuracy = number of aligned values / number of suggested values
# S --- answers to be evaluated (i.e. suggested values)
# T --- standard answers
def ReachabilityRepairAccuracyS(S,T):
    P = T
    Q = S

    print Q
    

    # number of values = number of variables * 2 + 1 (1 is the name of the operation)
    NumV = len(Q[0][0]) * 2 + 1

    # total number of values
    TNumV = len(Q) * NumV

    P = P + []
    Q = Q + []
    S = 0.0
    for X in P:
        Z = ""
        DXZ = -1
        for Y in Q:
            dt = 0.0
            XT = X[0] + [X[1]] + X[2]
            YT = Y[0] + [Y[1]] + Y[2]
            if len(XT) != NumV or len(YT) != NumV:
                print "Error!!!"
                ppp
            for i in xrange(NumV):
                if XT[i] == YT[i]:
                    dt = dt + 1.0
            if dt > DXZ:
                DXZ = dt
                Z = Y
        if DXZ == -1:
            continue
        S = S + DXZ
        Q.remove(Z)
    Acc = S / TNumV
    
    return Acc





# accuracy = number of aligned values / total number of values
# total number of values = max(total number of values in S, total number of values in T)
# S --- answers to be evaluated
# T --- standard answers
def ReachabilityRepairAccuracyST(S,T):
    if len(S) < len(T):
        P = S
        Q = T
    else:
        P = T
        Q = S

    # number of values = number of variables * 2 + 1 (1 is the name of the operation)
    NumV = len(Q[0][0]) * 2 + 1

    # total number of values
    TNumV = len(Q) * NumV

    P = P + []
    Q = Q + []
    S = 0.0
    for X in P:
        Z = ""
        DXZ = -1
        for Y in Q:
            dt = 0.0
            XT = X[0] + [X[1]] + X[2]
            YT = Y[0] + [Y[1]] + Y[2]
            if len(XT) != NumV or len(YT) != NumV:
                print "Error!!!"
                ppp
            for i in xrange(NumV):
                if XT[i] == YT[i]:
                    dt = dt + 1.0
            if dt > DXZ:
                DXZ = dt
                Z = Y
        if DXZ == -1:
            continue
        S = S + DXZ
        Q.remove(Z)
    Acc = S / TNumV
    
    return Acc


def EstimateClassificationBoundary(W,TL,wdir):
    SP = PredictUsingSemanticsModel(W,TL,wdir)
    U = numpy.mean(SP)
    A = numpy.std(SP)
    #XL = numpy.min(SP)
    #XR = numpy.max(SP)

    #print U,A,XL,XR
    return [U,A]
    

def GetFreeMemoryGB():
    x = os.popen("vmstat -s").read()
    x = x.split("\n")
    for y in x:
        z = 0
        if "K free memory" in y:
            z = y.replace("K free memory","")
            z = int(z)
            z = z * 1.0 / 1024 / 1024
            break
    return z



# W --- Semantics model
# SP --- Set of pre-states
# SF --- Set of operations
# SQ --- Set of post-states
# wdir --- working directory
import itertools
def ComputeAllSemanticProbabilities(W,SP,SF,SQ,wdir):
    cmd = "mkdir %s"%wdir
    os.system(cmd)
    TL = []
    i = 0
    res = []

    """
    PFQ = itertools.product([SP,SF,SQ])
    print len(PFQ)
    raw_input("xxx")
    i = 0
    N = 1000000
    while i < len(PFQ):
        TL = itertools.islice(PFQ,i,min(i+N,len(PFQ)))
        res.extend(CASP_Subs(W,TL,wdir))
        del TL
        gc.collect()
        i = i + N
        raw_input("yyyy")
    return res
    """

    """
    raw_input("xxxx")
    for P in SP:
        for Q in SQ:
            for F in SF:
                TL.append([P,F,Q])
    raw_input("yyyyy")
    del TL
    gc.collect()
    TL = []
    """

    """
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    mem_gib = mem_bytes/(1024.**3)
    """
   

    stt = time.time() 
    for P in SP:
        for F in SF:
            for Q in SQ:
                TL.append([P,F,Q])
                i = i + 1
                if i % 10000 == 0:
                    if time.time() - stt > 300:
                        print "Timeout!"
                        return res
                    
                if i >= 500000:
                    res.extend(CASP_Subs(W,TL,wdir))
                    del TL
                    gc.collect()
                    TL = []
                    i = 0
    if TL != []:
        res.extend(CASP_Subs(W,TL,wdir))
    return res

"""
import itertools
for element in itertools.product(*somelists):
    print(element)
"""

def CASP_Subs(W,TL,wdir):
    S = PredictUsingSemanticsModel(W,TL,wdir)
    res = []
    for i in xrange(len(TL)):
        res.append([TL[i],S[i]])
    return res

"""
def ComputeAllSemanticProbabilities_stable(W,SP,SF,SQ,wdir):
    cmd = "mkdir %s"%wdir
    os.system(cmd)
    TL = []
    for P in SP:
        for Q in SQ:
            for F in SF:
                TL.append([P,F,Q])
    #for x in TL: print x
    S = PredictUsingSemanticsModel(W,TL,wdir)
    res = []
    for i in xrange(len(TL)):
        res.append([TL[i],S[i]])
    return res
"""

# RL --- classified and sorted list of repairs
def GetBestRepair(RL):
    print "Note: Ignore isolation repairs and only get best revision repairs."
    res = []
    for X in RL:
        T = X[0]
        for P in X:
            if P[0] == "revision":
                T.append(P[1])
                break
        res.append(T)
    return res



# RL --- classified and sorted list of repairs
# G --- Oracle
def GetIRAnswer(RL,G):
    i = G[0].index("REVISION")
    ANS1 = G[1][i]

    i = G[0].index("ISOLATION")
    ANS2 = G[1][i]

    ANS = ANS1 + []
    for X in ANS2:
        ANS.append(X + ["isolation"])
    ANS.sort()

    res = []
    for X in RL:
        T = X[0]
        S = ""
        for Y in ANS:
            P = [Y[0],Y[1],Y[2]]
            if P == T:
                S = Y[3]
                break
        if S == "":
            print "WARNING: Cannot find an answer for",T,"."
            res.append(T + ["NotFound"])
            continue
        res.append(T + [S])
    return res




# RL --- classified and sorted list of repairs
# G --- Oracle
def GetRepairAnswer(RL,G):
    i = G[0].index("ANSWER")
    ANS = G[1][i]

    """
    for x in RL:
        if x[0][0] != [['PID1'], ['PID3'], ['PID2']]: continue
        for y in x:
            
            print y
    """

    res = []
    for X in RL:
        T = X[0]
        S = ""
        for Y in ANS:
            P = [Y[0],Y[1],Y[2]]
            if P == T:
                S = Y[3]
                break
        if S == "":
            print "WARNING: Cannot find an answer for",T,"."
            res.append(T + ["NotFound"])
            continue
        res.append(T + [S])
    return res


# RL --- classified and sorted list of repairs
# G --- Oracle
def CompareRevisionWithAnswer(RL,G):
    i = G[0].index("ANSWER")
    ANS = G[1][i]

    res = []
    for X in RL:
        print "\n\n"
        T = X[0]
        S = ""
        for Y in ANS:
            P = [Y[0],Y[1],Y[2]]
            if P == T:
                S = Y[3]
                break
        if S == "":
            print "WARNING: Cannot find an answer for",T,"."
            res.append(["","NotFound"])
            continue

        # find the answer's rank and probability.
        ans_rank = 0
        ans_prob = -1
        for i in xrange(1,len(X)):
            V = X[i]
            if V[0] != "revision":
                continue
            ans_rank = ans_rank + 1
            if S == V[1]:
                ans_prob = V[2]
                break
        if ans_prob == -1:
            print "WARNING: Cannot find probability of the answer",S,"."
            ans_rank = -1

        # compare the best revision with the answer and compute accuracy.
        for i in xrange(1,len(X)):
            V = X[i]
            if V[0] != "revision":
                continue
            # The first revision is the best one.
            best_acc = RevisionVariableAccuracy(V[1],S)
            print "Faulty Transitions is:",T
            print "Best Revision is:",V[1],", probability is",V[2],", accuracy is",best_acc
            print "Answer is:",S,", probability is",ans_prob,", rank is",ans_rank
            res.append([V[1],S])
            break
    return res




# RL --- classified and sorted list of repairs
# G --- Oracle
def CompareWithIRAnswer(RL,G):
    i = G[0].index("REVISION")
    ANS1 = G[1][i]

    i = G[0].index("ISOLATION")
    ANS2 = G[1][i]

    ANS = ANS1 + []
    for X in ANS2:
        ANS.append(X + ["isolation"])
    ANS.sort()

    res = []
    ISO_T = []
    ISO_F = []
    for X in RL:
        print "\n\n"
        T = X[0]
        S = ""
        for Y in ANS:
            P = [Y[0],Y[1],Y[2]]
            if P == T:
                S = Y[3]
                break
        if S == "":
            print "WARNING: Cannot find an answer for",T,"."
            res.append(["","NotFound"])
            continue

        # find the answer's rank and probability.

        if S == "isolation":
            ans_rank = 1
            ans_prob = 0.5 # Isolation Boundary
            best_rev_prob = -1
            for i in xrange(1,len(X)):
                V = X[i]
                if V[0] != "revision":
                    continue
                if V[2] > best_rev_prob:
                    best_rev_prob = V[2]
                if V[2] <= 0.5: # Isolation Boundary
                    break
                ans_rank = ans_rank + 1

        else:
            ans_rank = 0
            ans_prob = -1
            for i in xrange(1,len(X)):
                V = X[i]
                if V[0] != "revision":
                    continue
                ans_rank = ans_rank + 1
                if S == V[1]:
                    ans_prob = V[2]
                    break


        if ans_prob == -1:
            print "WARNING: Cannot find probability of the answer",S,"."
            ans_rank = -1


        if S == "isolation":
            if ans_rank == 1:
                ISO_T.append(T)
                print "Faulty Transitions is:",T
                print "Best repair is isolation."
            else:
                ISO_F.append(T)
                print "Faulty Transitions is:",T
                print "Best revision is:",S,", probability is",best_rev_prob,"."
                print "Answer is isolation, and rank is",ans_rank,"."
 
            continue

        # compare the best repair with the answer and compute accuracy.
        for i in xrange(1,len(X)):
            V = X[i]
            if V[0] != "revision":
                continue
            # The first revision is the best one.
            best_acc = RevisionVariableAccuracy(V[1],S)
            print "Faulty Transitions is:",T
            print "Best Revision is:",V[1],", probability is",V[2],", accuracy is",best_acc
            print "Answer is:",S,", probability is",ans_prob,", rank is",ans_rank
            res.append([V[1],S])
            break

    # ISO_T --- list of true isolation answers.
    # ISO_F --- list of false isolation answers.
    return [res,[ISO_T,ISO_F]]





# RL --- classified and sorted list of repairs
# G --- Oracle
def CompareRevisionWithRevisionAnswer(RL,G):
    i = G[0].index("REVISION")
    ANS = G[1][i]

    res = []
    for X in RL:
        print "\n\n"
        T = X[0]
        S = ""
        for Y in ANS:
            P = [Y[0],Y[1],Y[2]]
            if P == T:
                S = Y[3]
                break
        if S == "":
            print "WARNING: Cannot find an answer for",T,"."
            res.append(["","NotFound"])
            continue

        # find the answer's rank and probability.
        ans_rank = 0
        ans_prob = -1
        for i in xrange(1,len(X)):
            V = X[i]
            if V[0] != "revision":
                continue
            ans_rank = ans_rank + 1
            if S == V[1]:
                ans_prob = V[2]
                break
        if ans_prob == -1:
            print "WARNING: Cannot find probability of the answer",S,"."
            ans_rank = -1

        # compare the best revision with the answer and compute accuracy.
        for i in xrange(1,len(X)):
            V = X[i]
            if V[0] != "revision":
                continue
            # The first revision is the best one.
            best_acc = RevisionVariableAccuracy(V[1],S)
            print "Faulty Transitions is:",T
            print "Best Revision is:",V[1],", probability is",V[2],", accuracy is",best_acc
            print "Answer is:",S,", probability is",ans_prob,", rank is",ans_rank
            res.append([V[1],S])
            break
    return res




def RevisionVariableAccuracy(S,T):
    if T == "NotFound":
        return 0.0
    acc = 0.0
    for i in xrange(len(S)):
       if S[i] == T[i]:
           acc = acc + 1
    acc = acc / len(S) 
    return acc

# M --- file name of an abstract machine
# W --- trained semantics model
# conddile --- configuration file
# resdir --- working directory
def MonteCarloStateSampling(M,W,conffile,sdir):

    cmd = "mkdir %s"%sdir
    os.system(cmd)
    s = sdir + "/M.mch"
    cmd = "cp %s %s"%(M,s)
    os.system(cmd)
    M = s
    
    fn = sdir + "/M_pp.mch"
    oscmd = "./../ProB/probcli -pp %s %s"%(fn,M)
    os.system(oscmd)
    M = fn

    with open(M) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]

    rev_cond = Bmch.generate_revision_condition(mch, [], "")

    # If integers exist in the model, then we limit the search space of integers.
    int_flag = False
    for x in rev_cond:
        for y in [": INTEGER",": NATURAL",": NATURAL1",": INT",": NAT",": NAT1"]:
            if x[len(x)-len(y):len(x)] == y:
                int_flag = True
                break

    Int_CompX = 1
    if int_flag == True:
        SType = W.SType
        VList = W.VList
        for j in xrange(len(SType)):
            if SType[j][0] != "Int":
                continue
            V = VList[j]
            T = SType[j][1:len(SType[j])]
            Int_CompX = Int_CompX * len(T)

            for i in xrange(len(rev_cond)):
                x = rev_cond[i]
                for P in [": INTEGER",": NATURAL",": NATURAL1",": INT",": NAT",": NAT1"]:
                    y = V + "_init " + P
                    if x[len(x)-len(y):len(x)] == y:
                        Q = ""
                        for u in T:
                            Q = Q + str(u) + ","
                        Q = Q[0:len(Q)-1]
                        Q = V + "_init : {" + Q + "}"
                        z = x.replace(y,Q)
                        rev_cond[i] = z
                        break

    all_opes = Bmch.get_all_opes(mch)

    # MS --- machine for sampling
    MS = []
    i = 0
    mchlen = len(mch)
    while i < mchlen:
        tt = Bmch.get_first_token(mch[i])
        # Based on the syntax of <The B-book>, p.273.
        if tt == "INITIALIZATION": break
        if tt == "INITIALISATION": break
        MS.append(mch[i])
        i = i + 1
    MS.append("INITIALISATION")
    MS = MS + rev_cond
    #res.append("OPERATIONS")
    #res = res + all_opes
    MS.append("END")
   
    fn = sdir + "/sampling.mch"
    Bmch.print_mch_to_file(MS,fn)
    MS = fn

    mcss_max_num_samples = Bmch.read_config(conffile,"mcss_max_num_samples","int")
    
    D = sdir + "D.txt"
    #genmode = "-mc %d -mc_mode random -p MAX_INITIALISATIONS %d -p RANDOMISE_ENUMERATION_ORDER TRUE -p MAX_DISPLAY_SET -1"%(mcss_max_num_samples * 100, mcss_max_num_samples * 100)

    genmode = "-mc %d -mc_mode random -p MAX_INITIALISATIONS %d -p RANDOMISE_ENUMERATION_ORDER TRUE -p MAX_DISPLAY_SET -1"%(mcss_max_num_samples, mcss_max_num_samples)


    mkgraph = "./../ProB/probcli %s %s -nodead -spdot %s -c"%(MS,genmode,D)
    os.system(mkgraph)

    sg = Bgenlib.BStateGraphForNN()
    sg.ReadStateGraph(D)
    SI = sg.GetInitList()

    """
    random.shuffle(SI)

    if len(SI) > mcss_max_num_samples:
        SI = SI[0:mcss_max_num_samples]
    """
    print "Sample %d times. Get %d samples that satisfies requirements."%(mcss_max_num_samples,len(SI))

    return SI

# RL --- list of repairs
def ClassifyAndSortRepairs(RL):
    TL = []
    for X in RL:
        P = [X[1],X[2],X[3]]
        if not(P in TL):
            TL.append(P)
    TL.sort()
    L = len(TL)
    res = []
    for i in xrange(L):
        res.append([TL[i]])

    # Add isolation repairs.
    for X in RL:
        if X[0] != "isolation":
            continue
        P = [X[1],X[2],X[3]]
        idx = TL.index(P)
        Q = [X[0]]
        res[idx].append(Q)

    # Sort revision repairs.
    RevS = []
    for i in xrange(L):
        RevS.append([])
    for X in RL:
        if X[0] != "revision":
            continue
        P = [X[1],X[2],X[3]]
        idx = TL.index(P)
        Q = [X[0],X[4],X[5]]
        RevS[idx].append(Q)
    for i in xrange(L):
        RevS[i].sort(key = lambda x: x[2], reverse = True)
        res[i] = res[i] + RevS[i]
    
    for i in xrange(L):
        print "\n\n",TL[i]
        for x in res[i]: print x
   
    return res


def GeneratingTrainingData(M,conf,resdir,learn_tails = False):

    mchfile = M
    conffile = conf
    resfolder = resdir

    print "Generating Training Data for Semantics Learning..."
    print "Source File:", mchfile
    print "Configuration File:", conffile
    print "Working Folder:", resfolder

    cmd = "mkdir %s"%resfolder
    os.system(cmd)

    ff = resfolder + "/source.mch"
    cmd = "./../ProB/probcli -pp %s %s"%(ff,mchfile)
    os.system(cmd)
    mchfile = ff

    ff = resfolder + "/config"
    cmd = "cp %s %s"%(conffile,ff)
    os.system(cmd)
    conffile = ff

    outfile = resfolder + "/trset.mch"
    sgfile = resfolder + "/trset.statespace.dot"
    dsfile = resfolder + "/data.txt"
    
    with open(mchfile) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]


    additional_sampling = Bmch.read_config(conffile,"additional_sampling","bool")
    if additional_sampling == True:
        print "\nUse additional sampling.\n"
        trsetmch = Bmch.generate_training_set_machine(mch,"")
    else:
        print "\nNot use additional sampling.\n"
        trsetmch = mch

    bscope = Bmch.generate_training_set_condition(mch)

    Bmch.print_mch_to_file(trsetmch,outfile)


    max_num_sampling_states = Bmch.read_config(conffile,"max_num_sampling_states","int")
    max_operations = Bmch.read_config(conffile,"max_operations","int")

    print "\nMaximum number of samples is", max_num_sampling_states, ".\n"

    # "-mc 100 and -p MAX_INITIALISATIONS 100" works well. But now I am trying more initialisations. 
    genmode = "-mc %d -mc_mode random -p MAX_INITIALISATIONS %d -p RANDOMISE_ENUMERATION_ORDER TRUE -p MAX_OPERATIONS %d -p MAX_DISPLAY_SET -1"%(max_num_sampling_states,max_num_sampling_states,max_operations)

    # We still need to carefully examine the performance of ProB-SMT and KODKOD.
    # When search space is small, NO-SMT, ProB-SMT and KODKOD have similar speed.
    #smtmode = "-p KODKOD TRUE -p SMT TRUE -p CLPFD TRUE"
    smtmode = ""

    mkgraph = "./../ProB/probcli %s %s -nodead -scope \"%s\" -spdot %s %s -c"%(outfile,genmode,bscope,sgfile,smtmode)

    os.system(mkgraph)

    sg = Bgenlib.BStateGraphForNN()
    sg.ReadStateGraph(sgfile)

    TL = sg.GetTransList()

    TL = sg.SortSetsInTransList(TL)

    # Remove faulty transitions.
    # FS --- Faulty States.
    # FT --- Faulty Transitions.
    if learn_tails != True:
        FS = sg.GetStatesWithoutOutgoingTransitions(TL)
        FT = sg.GetTransitionsWithPostStates(TL,FS)
        TL = Bmch.list_difference(TL,FT)

    SType = sg.GetSetTypeFromTransList(TL)
    VList = sg.GetVbleList()

    rd_seed = Bmch.read_config(conffile,"rd_seed","int")
    neg_prop = Bmch.read_config(conffile,"neg_prop","float")
    cv_prop = Bmch.read_config(conffile,"cv_prop","float")

    SilasData = sg.SilasTransListToData(TL,SType,VList,neg_prop,rd_seed)

    VData = SilasData[0]
    FData = SilasData[1:len(SilasData)]
    print len(FData)

    random.seed(rd_seed)
    random.shuffle(FData)

    num_tr = int(len(FData) * (1-cv_prop))


    TrData = [VData] + FData[0:num_tr]
    CvData = [VData] + FData[num_tr:len(FData)]

    fname = resfolder + "/train.csv"
    Bgenlib.write_list_to_csv(TrData,fname)
    fname = resfolder + "/valid.csv"
    Bgenlib.write_list_to_csv(CvData,fname)

    fname = resfolder + "/datatypes.txt"
    DataTypes = [VList] + SType
    f = open(fname,"w")
    for x in DataTypes:
        f.write(str(x) + "\n")
    f.close()

    Num_Tr = len(TrData) - 1
    Num_Cv = len(CvData) - 1

    return [Num_Tr,Num_Cv]


# SD[0] is the state diagram
# SD[1] is the variable list
def GeneratingTrainingDataByStateDiagram(SD,conf,resdir,learn_tails = False):

    conffile = conf
    resfolder = resdir

    sg = Bgenlib.BStateGraphForNN()

    TL = sg.SortSetsInTransList(SD[0])
    VList = SD[1]

    # Remove faulty transitions.
    # FS --- Faulty States.
    # FT --- Faulty Transitions.
    if learn_tails != True:
        FS = sg.GetStatesWithoutOutgoingTransitions(TL)
        FT = sg.GetTransitionsWithPostStates(TL,FS)
        TL = Bmch.list_difference(TL,FT)

    SType = sg.GetSetTypeFromTransList(TL)

    rd_seed = Bmch.read_config(conffile,"rd_seed","int")
    neg_prop = Bmch.read_config(conffile,"neg_prop","float")
    cv_prop = Bmch.read_config(conffile,"cv_prop","float")

    SilasData = sg.SilasTransListToData(TL,SType,VList,neg_prop,rd_seed)

    VData = SilasData[0]
    FData = SilasData[1:len(SilasData)]
    print len(FData)

    random.seed(rd_seed)
    random.shuffle(FData)

    num_tr = int(len(FData) * (1-cv_prop))


    TrData = [VData] + FData[0:num_tr]
    CvData = [VData] + FData[num_tr:len(FData)]

    fname = resfolder + "/train.csv"
    Bgenlib.write_list_to_csv(TrData,fname)
    fname = resfolder + "/valid.csv"
    Bgenlib.write_list_to_csv(CvData,fname)

    fname = resfolder + "/datatypes.txt"
    DataTypes = [VList] + SType
    f = open(fname,"w")
    for x in DataTypes:
        f.write(str(x) + "\n")
    f.close()

    Num_Tr = len(TrData) - 1
    Num_Cv = len(CvData) - 1

    return [Num_Tr,Num_Cv]




# Note: tails are transitions that do not have outgoing transitions.
# In reachability checking, tails are correct transitions.
# In invariant checking, tails are faulty transitions or deadlocks.
# In deadlock checking, tails are deadlocks.
# We can either learn or do not learn tails.
def TrainingSemanticsModel(M,conf,resdir,learn_tails = False):

    cmd = "mkdir %s"%resdir
    os.system(cmd)

    conffile = conf
    s = resdir + "/config"
    cmd = "cp %s %s"%(conffile,s)
    os.system(cmd)
    conffile = s

    start_time = time.time()

    if type(M) == type([]):
        N = GeneratingTrainingDataByStateDiagram(M,conffile,resdir, learn_tails = learn_tails)
    else:
        N = GeneratingTrainingData(M,conffile,resdir, learn_tails = learn_tails)
    Num_Tr = N[0]
    Num_Cv = N[1]

    training_data = resdir + "/train.csv"
    valid_data = resdir + "/valid.csv"
    datatypes_file = resdir + "/datatypes.txt"
    conffile = conf

    f = open(datatypes_file,"r")
    T = f.readlines()
    DType = []
    for x in T:
        DType.append(eval(x))
    VList = DType[0]
    SType = DType[1:len(DType)]

    print "Training Data:", training_data
    print "Cross Validation Data", valid_data

    tmtype = Bmch.read_config(conffile,"tendency_model","str")
    sg = Bgenlib.BStateGraphForNN()
    SD = sg.ReadCSVSemanticsDataAndComputeTypes([training_data,valid_data])
    SData = SD[0]
    SemTypes = SD[1]

    train_txt = resdir + "/train.txt"
    valid_txt = resdir + "/valid.txt"


    sg.WriteSemanticDataToTxt(SData[0],train_txt)
    sg.WriteSemanticDataToTxt(SData[1],valid_txt)

    #tmtype = "BNBayes"

 
    if tmtype == "Random":

        # ============== Random Section ==============

        RD = BernoulliNB(alpha=1.0, binarize=0.5, class_prior=None, fit_prior=True)

        Acc = -1.0
        AUC = -1.0

        RD.MdlType = "Random" 
        RD.VList = VList
        RD.SType = SType
        RD.SemTypes = SemTypes
 
        tmfile = resdir + "/semantics.mdl" 
        print "Writing Random tendency model to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(RD, filehandler)
        print "Tendency model has been written to the file."

   

    elif tmtype == "Logistic":

        # ============== Logistic Model Section ==============

        nnet_idim = len(SData[0][0][0])
        nnet_odim = 2

        logging.basicConfig()
        tr_log = logging.getLogger("mlp.optimisers")
        tr_log.setLevel(logging.DEBUG)

        rng = numpy.random.RandomState([2018,03,31])
        rng_state = rng.get_state()

        lrate = Bmch.read_config(conffile,"logistic_lrate","float")
        max_epochs = Bmch.read_config(conffile,"logistic_max_epochs","int")
        batch_size = Bmch.read_config(conffile,"logistic_minibatch_size","int")

        #max_epochs = 1000
        #lrate = lrate * 2

        BNNet = BLogistic_Init([nnet_idim, nnet_odim], rng)
        lr_scheduler = LearningRateFixed(learning_rate=lrate, max_epochs=max_epochs)
        #lr_scheduler = LearningRateNewBob(start_rate = lrate, scale_by = 0.5, min_derror_ramp_start = -0.1, min_derror_stop = 0, patience = 100, max_epochs = max_epochs)
        dp_scheduler = None #DropoutFixed(p_inp_keep=1.0, p_hid_keep=0.9)
        BNNet, Tr_Stat, Cv_Stat, Ev_Stat = BNNet_Semantic_Learning(BNNet, lr_scheduler, [train_txt,valid_txt,test_txt], dp_scheduler, batch_size = batch_size)

        tmfile = resdir + "/logistic.mdl"
        print "Writing logistic tendency model to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(BNNet, filehandler)
        print "Tendency model has been written to the file."

    elif tmtype == "ResNet":

        # ============== ResNet Net Section ==============

        nnet_idim = len(SData[0][0][0])
        nnet_odim = 2

        logging.basicConfig()
        tr_log = logging.getLogger("mlp.optimisers")
        tr_log.setLevel(logging.DEBUG)

        rng = numpy.random.RandomState([2018,03,31])
        rng_state = rng.get_state()


        lrate = Bmch.read_config(conffile,"resnet_lrate","float")
        max_epochs = Bmch.read_config(conffile,"resnet_max_epochs","int")
        batch_size = Bmch.read_config(conffile,"resnet_minibatch_size","int")
        num_hid = Bmch.read_config(conffile,"resnet_num_hid","int")
        num_layers = Bmch.read_config(conffile,"resnet_num_layers","int")

        #lrate = lrate * 2
        #max_epochs = 200

        BNNet = BResNet_Init([nnet_idim, num_hid, num_layers, nnet_odim], rng, 'Softmax')
        lr_scheduler = LearningRateFixed(learning_rate=lrate, max_epochs=max_epochs)
        #lr_scheduler = LearningRateNewBob(start_rate = lrate, scale_by = 0.5, min_derror_ramp_start = -0.1, min_derror_stop = 0, patience = 100, max_epochs = max_epochs)
        dp_scheduler = None #DropoutFixed(p_inp_keep=1.0, p_hid_keep=0.9)
        test_txt = None
        BNNet, Tr_Stat, Cv_Stat, Ev_Stat = BNNet_Semantic_Learning(BNNet, lr_scheduler, [train_txt,valid_txt,test_txt], dp_scheduler, batch_size = batch_size)

        Acc = -1
        AUC = -1

        BNNet.MdlType = "ResNet"
        BNNet.VList = VList
        BNNet.SType = SType
        BNNet.SemTypes = SemTypes


        tmfile = resdir + "/semantics.mdl"
        print "Writing ResNet tendency model to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(BNNet, filehandler)
        print "Tendency model has been written to the file."

    elif tmtype == "CART":

        # ============== Classification and Regression Tree Section ==============


        print "Not Implemented Error!"
        Not_Implemented_Error

        tr_data = dt[0]+dt[1]+dt[2]

        num_tree = Bmch.read_config(conffile,"cart_num_tree","int")
        min_var_exp = Bmch.read_config(conffile,"cart_min_var_exp","int")
        max_var_exp = Bmch.read_config(conffile,"cart_max_var_exp","int")
        data_prop = Bmch.read_config(conffile,"cart_data_prop","float")
        use_mp = Bmch.read_config(conffile,"cart_use_mp","bool")

        CARTree = RandMultiRegTree(data=tr_data, num_tree=num_tree, min_var_exp_scale=[min_var_exp,max_var_exp], data_prop=data_prop, use_mp=use_mp)

        CARTree.MType = "CART"    
        CARTree.SType = SType
        CARTree.OpeList = OpeList

        print "Writing CART tendency model to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(CARTree, filehandler)
        print "Tendency model has been written to the file."

    elif tmtype == "BNBayes":

        # ============== Bernoulli Naive Bayes Section ==============

        rng = numpy.random.RandomState([2018,03,31])
        rng_state = rng.get_state()

        tr_feat = []
        tr_tgt = []
        for x in SData[0]:
            tr_feat.append(x[0])
            tr_tgt.append(x[1])
        tr_data = [tr_feat,tr_tgt]

        cv_feat = []
        cv_tgt = []
        for x in SData[1]:
            cv_feat.append(x[0])
            cv_tgt.append(x[1])
        cv_data = [cv_feat,cv_tgt]

        #num_tree = 256
        
        #st_time = time.time()
        # Training

        #RF = RandomForestClassifier(n_estimators = num_tree, min_impurity_decrease = 0.0)
        #RF.fit(tr_feat, tr_tgt)


        BNB = BernoulliNB(alpha=1.0, binarize=0.5, class_prior=None, fit_prior=True)
        BNB.fit(tr_feat, tr_tgt)


        # Testing.
        #Acc = RF.score(cv_feat,cv_tgt)
        Acc = BNB.score(cv_feat,cv_tgt)
        print "Accuracy on Cross Validation Set is:", Acc * 100, "%."

        cv_proba = BNB.predict_proba(cv_feat)[:,1]
        AUC = roc_auc_score(cv_tgt, cv_proba)
        print "ROC-AUC is:", AUC, "."


        #ed_time = time.time()
        #print ed_time - st_time 

        BNB.MdlType = "BNBayes" 
        BNB.VList = VList
        BNB.SType = SType
        BNB.SemTypes = SemTypes
 
        tmfile = resdir + "/semantics.mdl" 
        print "Writing BNBayes tendency model to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(BNB, filehandler)
        print "Tendency model has been written to the file."

    elif tmtype == "MLP":

        # ============== MLP Section ==============

        rng = numpy.random.RandomState([2018,03,31])
        rng_state = rng.get_state()

        tr_feat = []
        tr_tgt = []
        for x in SData[0]:
            tr_feat.append(x[0])
            tr_tgt.append(x[1])
        tr_data = [tr_feat,tr_tgt]

        cv_feat = []
        cv_tgt = []
        for x in SData[1]:
            cv_feat.append(x[0])
            cv_tgt.append(x[1])
        cv_data = [cv_feat,cv_tgt]

        #num_tree = 256
        
        #st_time = time.time()
        # Training

        # MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(256,5), random_state=1) # This setting is significantly better than the default.
        MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
       
        MLP.fit(tr_feat, tr_tgt)


        # Testing.
        #Acc = RF.score(cv_feat,cv_tgt)
        Acc = MLP.score(cv_feat,cv_tgt)
        print "Accuracy on Cross Validation Set is:", Acc * 100, "%."

        cv_proba = MLP.predict_proba(cv_feat)[:,1]
        AUC = roc_auc_score(cv_tgt, cv_proba)
        print "ROC-AUC is:", AUC, "."
 
        #ed_time = time.time()
        #print ed_time - st_time 

        MLP.MdlType = "MLP" 
        MLP.VList = VList
        MLP.SType = SType
        MLP.SemTypes = SemTypes
 
        tmfile = resdir + "/semantics.mdl" 
        print "Writing BNBayes tendency model to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(MLP, filehandler)
        print "Tendency model has been written to the file."


    elif tmtype == "LR":

        # ============== Logistic Regression Section ==============

        rng = numpy.random.RandomState([2018,03,31])
        rng_state = rng.get_state()

        tr_feat = []
        tr_tgt = []
        for x in SData[0]:
            tr_feat.append(x[0])
            tr_tgt.append(x[1])
        tr_data = [tr_feat,tr_tgt]

        cv_feat = []
        cv_tgt = []
        for x in SData[1]:
            cv_feat.append(x[0])
            cv_tgt.append(x[1])
        cv_data = [cv_feat,cv_tgt]

        #num_tree = 256
        
        #st_time = time.time()
        # Training

        LR = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr')
        LR.fit(tr_feat, tr_tgt)


        # Testing.
        #Acc = RF.score(cv_feat,cv_tgt)
        Acc = LR.score(cv_feat,cv_tgt)
        print "Accuracy on Cross Validation Set is:", Acc * 100, "%."

        cv_proba = LR.predict_proba(cv_feat)[:,1]
        AUC = roc_auc_score(cv_tgt, cv_proba)
        print "ROC-AUC is:", AUC, "."
 
        #ed_time = time.time()
        #print ed_time - st_time 

        LR.MdlType = "LR" 
        LR.VList = VList
        LR.SType = SType
        LR.SemTypes = SemTypes
 
        tmfile = resdir + "/semantics.mdl" 
        print "Writing BNBayes tendency model to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(LR, filehandler)
        print "Tendency model has been written to the file."

    elif tmtype == "SVM":

        # ============== SVM Section ==============

        rng = numpy.random.RandomState([2018,03,31])
        rng_state = rng.get_state()

        tr_feat = []
        tr_tgt = []
        for x in SData[0]:
            tr_feat.append(x[0])
            tr_tgt.append(x[1])
        tr_data = [tr_feat,tr_tgt]

        cv_feat = []
        cv_tgt = []
        for x in SData[1]:
            cv_feat.append(x[0])
            cv_tgt.append(x[1])
        cv_data = [cv_feat,cv_tgt]

        #num_tree = 256
        
        #st_time = time.time()
        # Training

        #SVM = svm.SVC(kernel='linear')
        SVM = svm.SVC(kernel='rbf',probability=True)
        SVM.fit(tr_feat, tr_tgt)


        # Testing.
        #Acc = RF.score(cv_feat,cv_tgt)
        Acc = SVM.score(cv_feat,cv_tgt)
        print "Accuracy on Cross Validation Set is:", Acc * 100, "%."

        cv_proba = SVM.predict_proba(cv_feat)[:,1]
        AUC = roc_auc_score(cv_tgt, cv_proba)
        print "ROC-AUC is:", AUC, "."
 
        #ed_time = time.time()
        #print ed_time - st_time 

        SVM.MdlType = "SVM" 
        SVM.VList = VList
        SVM.SType = SType
        SVM.SemTypes = SemTypes
 
        tmfile = resdir + "/semantics.mdl" 
        print "Writing BNBayes tendency model to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(SVM, filehandler)
        print "Tendency model has been written to the file."




    elif tmtype == "SKCART":

        # ============== Scikit-learn CARTs Section ==============


        rng = numpy.random.RandomState([2018,03,31])
        rng_state = rng.get_state()

        num_tree = Bmch.read_config(conffile,"skcart_num_tree","int")

        tr_feat = []
        tr_tgt = []
        for x in SData[0]:
            tr_feat.append(x[0])
            tr_tgt.append(x[1])
        tr_data = [tr_feat,tr_tgt]

        cv_feat = []
        cv_tgt = []
        for x in SData[1]:
            cv_feat.append(x[0])
            cv_tgt.append(x[1])
        cv_data = [cv_feat,cv_tgt]

        #num_tree = 256
        
        #st_time = time.time()
        # Training

        #RF = RandomForestRegressor(n_estimators = num_tree, min_impurity_decrease = 0.0)

        #RF = RandomForestClassifier(n_estimators = num_tree, min_impurity_decrease = 0.0)

        if num_tree <= 0:
            # By default, the number of tree is 10 before scikit-learn version 0.20 and 100 after version 0.22. Here we use 100.
            num_tree = 100

        RF = RandomForestClassifier(n_estimators = num_tree)

        #RF = RandomForestClassifier(min_impurity_decrease = 0.0)
       
        RF.fit(tr_feat, tr_tgt)

        # Testing.
        Acc = RF.score(cv_feat,cv_tgt)
        print "Accuracy on Cross Validation Set is:", Acc * 100, "%."

        cv_proba = RF.predict_proba(cv_feat)[:,1]
        AUC = roc_auc_score(cv_tgt, cv_proba)
        print "ROC-AUC is:", AUC, "."
 
        #ed_time = time.time()
        #print ed_time - st_time 

        RF.MdlType = "SKCART" 
        RF.VList = VList
        RF.SType = SType
        RF.SemTypes = SemTypes
 
        tmfile = resdir + "/semantics.mdl" 
        print "Writing SKCART tendency model (single) to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(RF, filehandler)
        print "Tendency model has been written to the file."


    elif tmtype == "Silas":
        
        silas_dir = resdir + "/silas/"
        cmd = "rm -r %s"%silas_dir
        os.system(cmd)
        cmd = "mkdir %s"%silas_dir
        os.system(cmd)
        cmd = "cp -r src/silas-json-schemata/ json-schemata"
        os.system(cmd)

        cmd = "silas gen-all -o %s %s/train.csv %s/valid.csv"%(silas_dir,resdir,resdir)
        os.system(cmd)

        silas_num_tree = Bmch.read_config(conffile,"silas_num_tree","int")
        silas_feature_proportion = "1.0"

        #silas_num_tree = 3000

        sf = silas_dir + "/settings.json"
        ChangeSilasSetting(sf,"feature_proportion",0.25,"float")
        ChangeSilasSetting(sf,"max_depth",32,"int")
        ChangeSilasSetting(sf,"desired_leaf_size",32,"int")
        #ChangeSilasSetting(sf,"sampling_method","uniform","str")
        # if silas_num_tree < 0, then use default settings.
        if silas_num_tree > 0:
            # ssf --- Silas setting files
            ChangeSilasSetting(sf,"number_of_trees",silas_num_tree,"int")
            """
            ssf = open(f,"r")
            ss = ssf.readlines()
            ssf.close()
            for i in xrange(len(ss)):
                x = ss[i]
                if "number_of_trees" in x:
                    y = "    \"number_of_trees\": %d,\n"%silas_num_tree
                    ss[i] = y
                    ssf = open(f,"w")
                    for p in ss:
                        ssf.write(p)
                    ssf.close()
                    break
            """
        cmd = "silas learn -o %s/model/ %s/settings.json"%(silas_dir,silas_dir)
        #os.system(cmd)
        P = os.popen(cmd)
        P = P.read()
        print P
        

        # Get Accuracy.
        i = 0
        x = "Accuracy:"
        while P[i:i+len(x)] != x:
            i = i + 1
        i = i + len(x)
        j = i + 1
        while P[j] != "\n":
            j = j + 1
        Acc = P[i:j]
        Acc = float(Acc)

        # Get ROC-AUC
        i = 0
        x = "ROC-AUC:"
        while P[i:i+len(x)] != x:
            i = i + 1
        i = i + len(x)
        j = i + 1
        while P[j] != "\n":
            j = j + 1
        AUC = P[i:j]
        AUC = float(AUC)

        #cmd = "silas predict -o %s/predictions.csv %s/model %s/valid.csv"%(silas_dir,silas_dir,resdir)
        #os.system(cmd)

        SM = SilasModel()
        SM.MdlType = "Silas"
        SM.SilasNumTrees = silas_num_tree
        SM.SilasDir = silas_dir
        SM.Data = []
        SM.Data.append("%s/train.csv"%resdir)
        SM.Data.append("%s/valid.csv"%resdir)
        SM.VList = VList
        SM.SType = SType
        SM.SemTypes = SemTypes

        # Get output labels.
        # smd --- Silas metadata
        f = silas_dir + "/model/metadata.json"
        ssf = open(f,"r")
        ss = ssf.readlines()
        ssf.close()
        for i in xrange(len(ss)-1):
            x1 = ss[i]
            x2 = ss[i+1]
            if "Available-Transition" in x1 and "collection_definition" in x2:
                x3 = ss[i+2]
                if "N" in x3:
                    label_N = 0
                    label_Y = 1
                elif "Y" in x3:
                    label_Y = 0
                    label_N = 1
                break
        SM.label_Y = label_Y
        SM.label_N = label_N
        
        tmfile = resdir + "/semantics.mdl"
        print "Writing silas model to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(SM, filehandler)
        print "Tendency model has been written to the file."

    else:
        print "Not Implemented Error!"
        Not_Implemented_Error

    end_time = time.time()
    elapsed_time = end_time - start_time

    print "Training Finished."
    print "Number of Training Examples:",Num_Tr
    print "Number of Validation Examples:",Num_Cv
    print "Type of Semantics Model:",tmtype
    print "Elapsed Time (s):",elapsed_time
    print "Classification Accuracy:",Acc
    print "ROC-AUC:",AUC

    return [Num_Tr,Num_Cv,tmtype,elapsed_time,Acc,AUC] 

# f --- setting file name.
# t --- setting identifier
# v --- value
# tp --- type of value
def ChangeSilasSetting(f,t,v,tp):
    ssf = open(f,"r")
    ss = ssf.readlines()
    ssf.close()
    for i in xrange(len(ss)):
        x = ss[i]
        if t in x:
            if tp == "int":
                y = "\"%s\": %d,\n"%(t,v)
            elif tp == "float":
                y = "\"%s\": %.2f,\n"%(t,v)
            else:
                y = "\"%s\": \"%s\",\n"%(t,v)
            ss[i] = y
    ssf = open(f,"w")
    for p in ss:
        ssf.write(p)
    ssf.close()

    return


# W --- semantics model
# TL --- list of revisions
# wdir --- working directory
def ScoreRevisionsUsingSemanticsModel(W,RL,wdir):
    TL = []
    for x in RL:
        s = [x[1],x[2],x[4]]
        TL.append(s)
    res = PredictUsingSemanticsModel(W,TL,wdir)
    return res

# Predict probablities of transitions using trained semantics model.
# W --- semantics model
# TL --- list of transitions
# wdir --- working directory
def PredictUsingSemanticsModel(W,TL,wdir):
    MdlType = W.MdlType
    VList = W.VList
    SType = W.SType
    SemTypes = W.SemTypes

    #DType = W.DType
    P = []
    sg = Bgenlib.BStateGraphForNN()
    PPD = sg.TransListToPrePostData(TL,SType,VList)


    if MdlType == "Random":
        Q = []
        for i in xrange(len(TL)):
            Q.append(random.random())
        return Q

    elif MdlType == "Silas":
        silas_dir = W.SilasDir
        cmd = "mkdir %s"%wdir
        os.system(cmd)
        fname = wdir + "/transitions.csv"
        #SType = sg.GetSetTypeFromTransList(TL)
        #VList = sg.GetVbleList()
        #SilasData = sg.SilasTransListToData(TL,SType,VList,neg_prop,rd_seed)
        Bgenlib.write_list_to_csv(PPD,fname)
        pname = wdir + "/predictions.csv"

        # Re-train Silas:
        silas_dir = W.SilasDir
        tr_data = W.Data[0]
        tname = wdir + "/train.csv"
        cmd = "cp %s %s"%(tr_data,tname)
        os.system(cmd)
        tr_data = tname
        fr_data = fname
        silas_num_tree = W.SilasNumTrees
        cmd = "rm -r %s"%silas_dir
        os.system(cmd)
        cmd = "mkdir %s"%silas_dir
        os.system(cmd)
        cmd = "cp -r src/silas-json-schemata/ json-schemata"
        os.system(cmd)

        cmd = "silas gen-all -o %s %s %s"%(silas_dir,tr_data,fr_data)
        os.system(cmd)
        sf = silas_dir + "/settings.json"
        ChangeSilasSetting(sf,"feature_proportion",0.25,"float")
        ChangeSilasSetting(sf,"max_depth",32,"int")
        ChangeSilasSetting(sf,"desired_leaf_size",32,"int")
        #ChangeSilasSetting(sf,"sampling_method","uniform","str")
        # if silas_num_tree < 0, then use default settings.
        if silas_num_tree > 0:
            # ssf --- Silas setting files
            ChangeSilasSetting(sf,"number_of_trees",silas_num_tree,"int")
        
        """
        # if silas_num_tree < 0, then use default settings.
        if silas_num_tree > 0:
            # ssf --- Silas setting files
            f = silas_dir + "/settings.json"
            ssf = open(f,"r")
            ss = ssf.readlines()
            ssf.close()
            for i in xrange(len(ss)):
                x = ss[i]
                if "number_of_trees" in x:
                    y = "    \"number_of_trees\": %d,\n"%silas_num_tree
                    ss[i] = y
                    ssf = open(f,"w")
                    for p in ss:
                        ssf.write(p)
                    ssf.close()
                    break
        """
        cmd = "silas learn -o %s/model/ %s/settings.json"%(silas_dir,silas_dir)
        os.system(cmd)
        # Get output labels.
        # smd --- Silas metadata
        f = silas_dir + "/model/metadata.json"
        ssf = open(f,"r")
        ss = ssf.readlines()
        ssf.close()
        for i in xrange(len(ss)-1):
            x1 = ss[i]
            x2 = ss[i+1]
            if "Available-Transition" in x1 and "collection_definition" in x2:
                x3 = ss[i+2]
                if "N" in x3:
                    label_N = 0
                    label_Y = 1
                elif "Y" in x3:
                    label_Y = 0
                    label_N = 1
                break
        W.label_Y = label_Y
        W.label_N = label_N
 
        # Re-training finished.

        cmd = "silas predict -o %s %s/model %s"%(pname,silas_dir,fname)
        os.system(cmd)
        label_Y = W.label_Y
        label_N = W.label_N
        P = []
        pf = open(pname,"r")
        SP = pf.readlines()
        for i in xrange(1,len(SP)):
            x = SP[i].split(",")
            t = int(x[0])
            x = x[1]
            x = x.replace("\n","")
            x = float(x)
            if t == label_N:
                x = 1.0 - x
            P.append(x)
        pf.close()
    else:
        SData = sg.VectorisePrePostData(PPD,SemTypes)
        SData = SData[0]
        feat = SData[0]
        tgt = SData[1]
        TgtIdx = SemTypes[-1].index("Y")
        if MdlType in ["SKCART","BNBayes","LR","SVM","MLP"]:
        #if MdlType == "SKCART":
            Q = W.predict_proba(feat)
            P = []
            for x in Q:
                P.append(x[TgtIdx])
        elif MdlType == "ResNet":
            ND = 4096
            NS = len(feat) / ND

            Q = []
            for i in xrange(NS):
                X = feat[i*ND:(i+1)*ND]
                Y = W.fprop(numpy.asarray(X))
                Q.extend(Y)

            if ND*NS < len(feat):
                X = feat[ND*NS:len(feat)]
                Y = W.fprop(numpy.asarray(X))
                Q.extend(Y)

            P = []
            for x in Q:
                P.append(x[TgtIdx])
        else:
            pppp
    
    return P


