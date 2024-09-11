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
from collections import Counter
from PyTorchNnet import *

# ==================================================

# This is a library for semantic learning

# =================================================

# Data auditing algorithm for semantic learning. 
# T --- training set
# TgtIdx --- The index of labels for positive training data.
def DataAuditing(T,TgtIdx):


    rng = numpy.random.RandomState([2018,03,31])
    rng_state = rng.get_state()

    SData = T + []
    random.shuffle(SData)

    l = len(SData)
    SData = [SData[0:l/2],SData[l/2:l]]

    num_tree = 10
    decision_threshold = 0.5
    res = []

    tr1_feat = []
    tr1_tgt = []
    for x in SData[0]:
        tr1_feat.append(x[0])
        tr1_tgt.append(x[1])
    tr1_data = [tr1_feat,tr1_tgt]

    tr2_feat = []
    tr2_tgt = []
    for x in SData[1]:
        tr2_feat.append(x[0])
        tr2_tgt.append(x[1])
    tr2_data = [tr2_feat,tr2_tgt]

    # Use tr1 to audit tr2
    RF1 = RandomForestClassifier(n_estimators = num_tree)
    RF1.fit(tr1_feat, tr1_tgt)

    Acc1 = RF1.score(tr2_feat,tr2_tgt)
    print "Accuracy on TR2 Set is:", Acc1 * 100, "%."

    if RF1.classes_[TgtIdx] != 0:
        raw_input("Index error in classifier.")
        pppp

    tr2_proba = RF1.predict_proba(tr2_feat)
    for i in xrange(len(tr2_tgt)):
        good_instance = True
        if abs(tr2_proba[i][TgtIdx] - decision_threshold) <= 0.1:
            good_instance = False
            continue
        if tr2_proba[i][TgtIdx] > decision_threshold:
            pred = 0
        else:
            pred = 1
        if pred != tr2_tgt[i]:
            good_instance = False
        if good_instance == True:
            res.append([tr2_feat[i],tr2_tgt[i]])

    
    # Use tr2 to audit tr1
    RF2 = RandomForestClassifier(n_estimators = num_tree)
    RF2.fit(tr2_feat, tr2_tgt)

    Acc2 = RF2.score(tr1_feat,tr1_tgt)
    print "Accuracy on TR1 Set is:", Acc2 * 100, "%."

    if RF2.classes_[TgtIdx] != 0:
        raw_input("Index error in classifier.")
        pppp

    tr1_proba = RF2.predict_proba(tr1_feat)
    for i in xrange(len(tr1_tgt)):
        good_instance = True
        if abs(tr1_proba[i][TgtIdx] - decision_threshold) <= 0.1:
            good_instance = False
            continue
        if tr1_proba[i][TgtIdx] > decision_threshold:
            pred = 0
        else:
            pred = 1
        if pred != tr1_tgt[i]:
            good_instance = False
        if good_instance == True:
            res.append([tr1_feat[i],tr1_tgt[i]])

    print "Data Auditing: get %s instances from %s instances."%(len(res),len(T))

    return res
  



# Instance weighting algorithm for semantic learning. The weight of a state is equal to the number of occurences of this state in state transitions. The weight of a state transition = the weight of its pre-state + the weight of its post-state.
# T --- training data for semantic models.
def InstanceWeighting(T):
    # SL - list of state features
    SL = []
    for p in T:
        x = p[0]
        l = len(x)
        sf1 = str(x[0:l/2])
        sf2 = str(x[l/2+1:l])
        SL.append(sf1)
        SL.append(sf2)

    # SC - counter of state features 
    SC = Counter(SL)
    minN = 1000000000
    for sf,N in SC.items():
        if N < minN:
            minN = N
    if minN <= 0:
        minN = 1

    res = []
    for p in T:
        x = p[0]
        l = len(x)
        sf1 = str(x[0:l/2])
        sf2 = str(x[l/2+1:l])
        w = (SC[sf1] + SC[sf2]) / (minN * 2)
        for i in xrange(w):
            res.append(p)

    return res



# 0 --- False ; 1 --- True
def SpecEqSens(clf,X,y):
    if list(clf.classes_) != [0,1]:
        print "Error: class labels of the classifier is not [0 1], but it is %s."%str(clf.classes_)
        pppp
    XT = []
    XF = []
    for i in xrange(len(X)):
        # 0 --- False ; 1 --- True
        if y[i] == 1:
            XT.append(X[i])
        elif y[i] == 0:
            XF.append(X[i])
        else:
            print "Error: Classes should be an integer 0 or 1, but it is %s."%y[i]
            pppp

    ST = clf.predict_proba(XT)
    ST = list(ST)
    for i in xrange(len(ST)):
        ST[i] = ST[i][0]
    print ST

    SF = clf.predict_proba(XF)
    SF = list(SF)
    for i in xrange(len(SF)):
        SF[i] = SF[i][0]
    print SF

    ST.sort()
    SF.sort(reverse=True)


    L = 0.0
    R = 1.0
    while (R - L > 0.00001):
        a = (L + R) / 2

        PT = ST[int((len(ST)-1) * a)]
        PF = SF[int((len(SF)-1) * a)]

        if PT > PF:
            R = a
        else:
            L = a
    print L,R
    a = L
    PT = ST[int((len(ST)-1) * a)]
    PF = SF[int((len(SF)-1) * a)]
    P = PF #(PT + PF) / 2
    print PT,PF,P

    return P


# 0 --- False ; 1 --- True
def AveThre(clf,X,y):
    if list(clf.classes_) != [0,1]:
        print "Error: class labels of the classifier is not [0 1], but it is %s."%str(clf.classes_)
        pppp
    XT = []
    XF = []
    for i in xrange(len(X)):
        # 0 --- False ; 1 --- True
        if y[i] == 1:
            XT.append(X[i])
        elif y[i] == 0:
            XF.append(X[i])
        else:
            print "Error: Classes should be an integer 0 or 1, but it is %s."%y[i]
            pppp

    ST = clf.predict_proba(XT)
    ST = list(ST)
    for i in xrange(len(ST)):
        ST[i] = ST[i][0]
    print ST

    SF = clf.predict_proba(XF)
    SF = list(SF)
    for i in xrange(len(SF)):
        SF[i] = SF[i][0]
    
    S = 0.0
    for x in ST + SF:
        S = S + x
    S = S / len(ST + SF)

    return S



# 0 --- False ; 1 --- True
def ReqSens(clf,X,y,a):
    if list(clf.classes_) != [0,1]:
        print "Error: class labels of the classifier is not [0 1], but it is %s."%str(clf.classes_)
        pppp
    XT = []
    XF = []
    for i in xrange(len(X)):
        # 0 --- False ; 1 --- True
        if y[i] == 1:
            XT.append(X[i])
        elif y[i] == 0:
            XF.append(X[i])
        else:
            print "Error: Classes should be an integer 0 or 1, but it is %s."%y[i]
            pppp

    ST = clf.predict_proba(XT)
    ST = list(ST)
    for i in xrange(len(ST)):
        ST[i] = ST[i][0]
    print ST

    SF = clf.predict_proba(XF)
    SF = list(SF)
    for i in xrange(len(SF)):
        SF[i] = SF[i][0]
    print SF

    L = max(SF)
    print L

    ST.sort(reverse=True)
    print ST
    R = ST[int((len(ST)-1) * a)]
    return R
    #L = max(SF)
    print R
    

    T = max([L,R])
    print T
    return T




# 0 --- False ; 1 --- True
def ReqSpec(clf,X,y,a):
    if list(clf.classes_) != [0,1]:
        print "Error: class labels of the classifier is not [0 1], but it is %s."%str(clf.classes_)
        pppp
    XT = []
    XF = []
    for i in xrange(len(X)):
        # 0 --- False ; 1 --- True
        if y[i] == 1:
            XT.append(X[i])
        elif y[i] == 0:
            XF.append(X[i])
        else:
            print "Error: Classes should be an integer 0 or 1, but it is %s."%y[i]
            pppp

    ST = clf.predict_proba(XT)
    ST = list(ST)
    for i in xrange(len(ST)):
        ST[i] = ST[i][0]
    print ST

    SF = clf.predict_proba(XF)
    SF = list(SF)
    for i in xrange(len(SF)):
        SF[i] = SF[i][0]
    print SF

    R = min(ST)
    print R

    SF.sort()
    print SF
    L = SF[int((len(SF)-1) * a)]
    #L = max(SF)
    print L-0.01
    return L
    

    T = max([R,L])
    print T
    return T


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



# accuracy = number of aligned values / number of suggested values (pre-states + operation names)
# S --- answers to be evaluated (i.e. suggested values)
# T --- standard answers
def ReachabilityRepairAccuracyS2(S,T):
    P = S
    Q = T

    # number of values = number of variables + 1 (1 is the name of the operation)
    NumV = len(Q[0][0]) + 1

    # total number of values
    TNumV = len(P) * NumV

    P = P + []
    Q = Q + []
    S = 0.0
    for X in P:
        Z = ""
        DXZ = -1
        for Y in Q:
            if X[2] != Y[2]:
                continue
            dt = 0.0
            XT = X[0] + [X[1]]
            YT = Y[0] + [Y[1]]
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
            print "Error: cannot find answers for",X
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


# accuracy = number of aligned values / number of suggested values (pre-states + operation names)
# S --- answers to be evaluated (i.e. suggested values)
# T --- standard answers
# output: accuracy
def GOALRepairAccuracyS2(S,T):
    P = S
    Q = T

    # number of values = number of variables + 1 (1 is the name of the operation)
    NumV = len(Q[0][0]) * 2 + 1

    # total number of values
    TNumV = len(P) * NumV

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
            print "Error: cannot find answers for",X
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
    print "Note: Ignore isolation repairs and only get best revision / insertion repairs."
    res = []
    for X in RL:
        T = X[0]
        for P in X:
            if P[0] in ["revision","insertion"]:
                T.append(P[1])
                break
        res.append(T)
    return res



# RL --- classified and sorted list of repairs
# TL --- list of existing state transitions. If a repair suggests a state transition in TL, this repair will not be selected, unless there is no other choice.
def GetBestIMDRepair(RL,TL):

    res = []
    for X in RL:
        T = X[0]
        # U is used to store the repair with the highest score, regardless of whether or not U suggests a state transition in TL. U is used when no other choice can be found.
        U = "None"
        for P in X:
            if U == "None" and P[0] in ["isolation","revision","insertion"]:
                U = P

            if P[0] == "isolation":
                T = T + P
                break

            elif P[0] == "revision":
                z = [T[0],T[1],P[1]]
                if z in TL:
                    continue
                T = T + P
                break

            elif P[0] == "insertion":
                z = P[1]
                if z in TL:
                    continue
                T = T + P
                break            

        if T == X[0]: # U is selected when no other choice can be found.
            T = T + U
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
def CompareWithIRAnswer(RL,G,Th = None):
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
        if Th == None:
            iso_prob = 0.5
        else:
            iso_prob = Th

        if S == "isolation":
            ans_rank = 1
            ans_prob = iso_prob # Isolation Boundary
            best_rev_prob = -1
            for i in xrange(1,len(X)):
                V = X[i]
                if V[0] != "revision":
                    continue
                if V[2] > best_rev_prob:
                    best_rev_prob = V[2]
                if V[2] <= iso_prob: # Isolation Boundary
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

        # Repair Selection
        XL = X[1:len(X)]
        for i in xrange(len(XL)):
            if XL[i][0] == "isolation":
                XL[i] = ["isolation","isolation",iso_prob]
        XL.sort(key=lambda x:x[2],reverse=True)
        XRep = XL[0]

        """
        for V in XL:
            print V
            raw_input("adsfsad")
        """

        if S != "isolation" and XRep[0] == "isolation":
            ISO_F.append(T)


        if S == "isolation":
            if XRep[0] == "isolation":
                ISO_T.append(T)
                print "Faulty Transitions is:",T
                print "Best repair is isolation."
            else:
                ISO_F.append(T)
                print "Faulty Transitions is:",T
                print "Best revision is:",XRep[0],", probability is",best_rev_prob,"."
                print "Answer is isolation, and rank is",ans_rank,"."
 
            continue

        # compare the best repair with the answer and compute accuracy.
        
        V = XRep
        if V[0] == "revision":
            # The first revision is the best one.
            best_acc = RevisionVariableAccuracy(V[1],S)
            print "Faulty Transitions is:",T
            print "Best Revision is:",V[1],", probability is",V[2],", accuracy is",best_acc
            print "Answer is:",S,", probability is",ans_prob,", rank is",ans_rank
            res.append([V[1],S])
        else:
            print "Faulty Transitions is:",T
            print "Best repair is isolation."
            res.append([V[1],S])
 
    # ISO_T --- list of true isolation answers.
    # ISO_F --- list of false isolation answers.
    ISO_T = Bmch.list_union(ISO_T,[])
    ISO_F = Bmch.list_union(ISO_F,[])

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
    if S == "isolation":
        return 0.0
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
# conffile --- configuration file
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



# M --- file name of an abstract machine
# GP --- a goal predicate to be solved
# W --- trained semantics model
# conffile --- configuration file
# resdir --- working directory
def GOALStateSampling(M,GP,W,conffile,sdir):

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

    rev_cond = Bmch.generate_revision_condition(mch, [" ( " + GP + " ) "], "")

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

    # Sort revision and insertion repairs.
    RS = []
    for i in xrange(L):
        RS.append([])
    for X in RL:
        if X[0] in ["revision","insertion"]:
            P = [X[1],X[2],X[3]]
            idx = TL.index(P)
            Q = [X[0],X[4],X[5]]
            RS[idx].append(Q)
    for i in xrange(L):
        RS[i].sort(key = lambda x: x[2], reverse = True)
        res[i] = res[i] + RS[i]
  
    """  
    for i in xrange(L):
        print "\n\n",TL[i]
        for x in res[i]:
            print x
            raw_input("next?")
    """

    return res

# Compute the absolute distance between states S and T.
def AbsoluteDistance(S,T):
    dist = 0
    for i in xrange(len(S)):
        X = S[i]
        Y = T[i]

        # check whether it is an integer
        is_int = False
        if type(X) != type([]):
            is_int = True
            try:
                int(X)
            except ValueError:
                is_int = False

        if is_int == True:
            dist = dist + abs(int(X) - int(Y))
        elif type(X) == type([]):
            S1 = Bmch.list_union(X,Y)
            S2 = Bmch.list_intersect(X,Y)
            S3 = Bmch.list_difference(S1,S2)
            dist = dist + len(S3)
        else:
            if X != Y: 
                dist = dist + 1
    return dist



# RL --- list of repairs
# MaxDist --- Max Distance for Distance Constraints.
def ClassifyAndSortIMDRepairs(RL, MaxDist = None):
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

    # Sort insertions / modifications / deletions.
    RS = []
    for i in xrange(L):
        RS.append([])
    for X in RL:
        if X[0] in ["revision","insertion","isolation"]:

            # Ignore revision repairs that do not satisfy distance constraints.
            if MaxDist != None and X[0] == "revision":
                if AbsoluteDistance(X[3],X[4]) > MaxDist:
                    continue

            P = [X[1],X[2],X[3]]
            idx = TL.index(P)
            Q = [X[0],X[4],X[5]]
            RS[idx].append(Q)
    for i in xrange(L):
        RS[i].sort(key = lambda x: x[2], reverse = True)
        res[i] = res[i] + RS[i]

    return res





def GeneratingTrainingData(M,conf,resdir,learn_tails = False,excluded_data = []):

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
    feat_type = Bmch.read_config(conffile,"feat_type","str")

    SilasData = sg.SilasTransListToData(TL,SType,VList,neg_prop,rd_seed,ExcludedData=excluded_data,feat_type=feat_type)

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
def GeneratingTrainingDataByStateDiagram(SD,conf,resdir,learn_tails = False,excluded_data = []):

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
    feat_type = Bmch.read_config(conffile,"feat_type","str")

    SilasData = sg.SilasTransListToData(TL,SType,VList,neg_prop,rd_seed,ExcludedData=excluded_data,feat_type=feat_type)

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
def TrainingSemanticsModel(M,conf,resdir,learn_tails = False,excluded_data = []):

    cmd = "mkdir %s"%resdir
    os.system(cmd)

    conffile = conf
    s = resdir + "/config"
    cmd = "cp %s %s"%(conffile,s)
    os.system(cmd)
    conffile = s

    start_time = time.time()

    if type(M) == type([]):
        N = GeneratingTrainingDataByStateDiagram(M,conffile,resdir, learn_tails = learn_tails,excluded_data = excluded_data)
    else:
        N = GeneratingTrainingData(M,conffile,resdir, learn_tails = learn_tails,excluded_data = excluded_data)
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

    #print "Training Data:", training_data
    #print "Cross Validation Data", valid_data

    tmtype = Bmch.read_config(conffile,"tendency_model","str")
    feat_type = Bmch.read_config(conffile,"feat_type","str")

    sg = Bgenlib.BStateGraphForNN()
    SD = sg.ReadCSVSemanticsDataAndComputeTypes([training_data,valid_data])
    SData = SD[0]
    SemTypes = SD[1]

    train_txt = resdir + "/train.txt"
    valid_txt = resdir + "/valid.txt"

    apply_data_auditing = True
    apply_instance_weighting = False

    if apply_data_auditing == True:
        SData[0] = DataAuditing(SData[0],SemTypes[-1].index("Y"))
    if apply_instance_weighting == True:
        SData[0] = InstanceWeighting(SData[0])

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
        RD.Threshold = 0.5
        RD.FeatType = feat_type


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
        BNNet.Threshold = 0.5
        BNNet.FeatType = feat_type


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

        cv_proba = BNB.predict_proba(cv_feat)[:,0]
        AUC = roc_auc_score(cv_tgt, cv_proba)
        print "ROC-AUC is:", AUC, "."


        #ed_time = time.time()
        #print ed_time - st_time 

        BNB.MdlType = "BNBayes" 
        BNB.VList = VList
        BNB.SType = SType
        BNB.SemTypes = SemTypes
        BNB.Threshold = 0.5
        BNB.FeatType = feat_type

 
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

        cv_proba = MLP.predict_proba(cv_feat)[:,0]
        AUC = roc_auc_score(cv_tgt, cv_proba)
        print "ROC-AUC is:", AUC, "."
 
        #ed_time = time.time()
        #print ed_time - st_time 

        MLP.MdlType = "MLP" 
        MLP.VList = VList
        MLP.SType = SType
        MLP.SemTypes = SemTypes
        MLP.Threshold = 0.5
        MLP.FeatType = feat_type
 
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

        cv_proba = LR.predict_proba(cv_feat)[:,0]
        AUC = roc_auc_score(cv_tgt, cv_proba)
        print "ROC-AUC is:", AUC, "."
 
        #ed_time = time.time()
        #print ed_time - st_time 

        LR.MdlType = "LR" 
        LR.VList = VList
        LR.SType = SType
        LR.SemTypes = SemTypes
        LR.Threshold = 0.5
        LR.FeatType = feat_type

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

        cv_proba = SVM.predict_proba(cv_feat)[:,0]
        AUC = roc_auc_score(cv_tgt, cv_proba)
        print "ROC-AUC is:", AUC, "."
 
        #ed_time = time.time()
        #print ed_time - st_time 

        SVM.MdlType = "SVM" 
        SVM.VList = VList
        SVM.SType = SType
        SVM.SemTypes = SemTypes
        SVM.Threshold = 0.5
        SVM.FeatType = feat_type

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

        # default
        RF = RandomForestClassifier(n_estimators = num_tree)

        #RF = RandomForestClassifier(n_estimators = num_tree, max_features = "sqrt", min_samples_leaf = 100)

        #RF = RandomForestClassifier(min_impurity_decrease = 0.0)
       
        RF.fit(tr_feat, tr_tgt)

        # Testing.
        Acc = RF.score(cv_feat,cv_tgt)
        print "Accuracy on Cross Validation Set is:", Acc * 100, "%."

        cv_proba = RF.predict_proba(cv_feat)[:,0]
        AUC = -1
        #AUC = roc_auc_score(cv_tgt, cv_proba)
        print "ROC-AUC is:", AUC, "."
 
        #ed_time = time.time()
        #print ed_time - st_time 

        Threshold = 0.5 #SpecEqSens(RF,tr_feat,tr_tgt) #AveThre(RF,cv_feat,cv_tgt) # ReqSpec(RF,cv_feat,cv_tgt,0.9) #SpecEqSens(RF,cv_feat,cv_tgt) #0.5 #ReqSpec(RF,cv_feat,cv_tgt,0.7)

      
        """
        
        P1 = SpecEqSens(RF,tr_feat,tr_tgt)
        P2 = AveThre(RF,tr_feat,tr_tgt)
        P3 = ReqSpec(RF,tr_feat,tr_tgt,0.8)
        P4 = ReqSens(RF,tr_feat,tr_tgt,1.0)
        print "SpecEqSens",P1
        print "AveThre",P2
        print "ReqSpec",P3
        print "ReqSens",P4
        #print Threshold
        raw_input("asdfsdasa")
        """
        

        RF.MdlType = "SKCART" 
        RF.VList = VList
        RF.SType = SType
        RF.SemTypes = SemTypes
        RF.Threshold = Threshold
        RF.FeatType = feat_type
 
        tmfile = resdir + "/semantics.mdl" 
        print "Writing SKCART tendency model (single) to %s."%tmfile
        filehandler = open(tmfile, 'w')
        pickle.dump(RF, filehandler)
        print "Tendency model has been written to the file."


    elif tmtype in ["MLP2L", "MLP4L", "FNNRBB16L", "FNNRBB64L", "MLPAE4L", "MLPGAN4L"]:

        # ============== PyTorch Section ==============


        rng = numpy.random.RandomState([2018,03,31])
        rng_state = rng.get_state()

        if torch.cuda.is_available():
            print "Use a GPU to train a semantic model..."
            use_gpu = True
        else:
            print "Use a CPU to train a semantic model..."
            use_gpu = False

        # When the number of training examples is insufficient, we use GAN to generate more training examples.
        use_GAN = False
        if len(SData[0]) / 2 < 2048:
            use_GAN = True

        batch_size = 256

        in_dim = len(SData[0][0][0])
        GAN_init_lrate = 0.01
        GAN_lrate_dec_prop = 0.99
        GAN_max_epochs = 1000
        GAN_rand_dim = 64
        num_org_data = len(SData[0]) / 2
        GAN_data_times = 100000 / num_org_data
        data_fake = []

        # Generating positive instances.
        if "GAN" in tmtype or use_GAN == True:

            tr_feat = []
            #tr_tgt = []
            tr_GAN_pos_data = []
            i = 0
            for x in SData[0]:
                if x[1] != 0:
                    continue
                i = i + 1
                tr_feat.append(torch.tensor(x[0]).float())
                #tr_tgt.append(x[1])
                # We ignore the last batch.
                if i == batch_size:
                    tr_feat = torch.stack(tr_feat)
                    #tr_tgt = torch.tensor(tr_tgt)
                    if use_gpu == True:
                        tr_feat = tr_feat.cuda()
                        #tr_tgt = tr_tgt.cuda()
                    tr_GAN_pos_data.append(tr_feat)
                    tr_feat = []
                    #tr_tgt = []
                    i = 0

            rand_dim = GAN_rand_dim
            model_G = GAN_G_ReLU(num_nodes = [rand_dim, 128, 256, 512, 1024, in_dim])
            model_D = MLP_ReLU(num_nodes = [in_dim, 256, 256, 256, 256, 2])
            if use_gpu == True:
                model_G = model_G.cuda()
                model_D = model_D.cuda()
            model_G, model_D = train_GAN_multi_epochs(model_G, model_D, tr_GAN_pos_data, max_epochs = GAN_max_epochs, init_lrate = GAN_init_lrate, lrate_dec_prop = GAN_lrate_dec_prop, use_gpu = use_gpu)
            # Generating data using GAN
            rand_in_tensor = torch.randn(num_org_data * GAN_data_times, rand_dim)
            if use_gpu == True:
                rand_in_tensor = rand_in_tensor.cuda()
            feat_fake = model_G(rand_in_tensor)
            feat_fake = feat_fake.cpu()
            
            for x in feat_fake:
                data_fake.append([x, 0])

        #raw_input("sadfas")

        # Generate negative instances.
        if "GAN" in tmtype or use_GAN == True:

            tr_feat = []
            #tr_tgt = []
            tr_GAN_neg_data = []
            i = 0
            for x in SData[0]:
                if x[1] != 1:
                    continue
                i = i + 1
                tr_feat.append(torch.tensor(x[0]).float())
                #tr_tgt.append(x[1])
                # We ignore the last batch.
                if i == batch_size:
                    tr_feat = torch.stack(tr_feat)
                    #tr_tgt = torch.tensor(tr_tgt)
                    if use_gpu == True:
                        tr_feat = tr_feat.cuda()
                        #tr_tgt = tr_tgt.cuda()
                    tr_GAN_neg_data.append(tr_feat)
                    tr_feat = []
                    #tr_tgt = []
                    i = 0

            rand_dim = GAN_rand_dim
            model_G = GAN_G_ReLU(num_nodes = [rand_dim, 128, 256, 512, 1024, in_dim])
            model_D = MLP_ReLU(num_nodes = [in_dim, 256, 256, 256, 256, 2])
            if use_gpu == True:
                model_G = model_G.cuda()
                model_D = model_D.cuda()
            model_G, model_D = train_GAN_multi_epochs(model_G, model_D, tr_GAN_neg_data, max_epochs = GAN_max_epochs, init_lrate = GAN_init_lrate, lrate_dec_prop = GAN_lrate_dec_prop, use_gpu = use_gpu)
            # Generating data using GAN
            rand_in_tensor = torch.randn(num_org_data * GAN_data_times, rand_dim)
            if use_gpu == True:
                rand_in_tensor = rand_in_tensor.cuda()
            feat_fake = model_G(rand_in_tensor)
            feat_fake = feat_fake.cpu()
            
            for x in feat_fake:
                data_fake.append([x, 1])

        if "GAN" in tmtype or use_GAN == True:
            data_all = SData[0] + data_fake
            random.shuffle(data_all)
            SData[0] = data_all



        i = 0
        tr_feat = []
        tr_tgt = []
        tr_data = []
        tr_ae_data = []
        for x in SData[0]:
            i = i + 1
            if type(x[0]) == type([]):
                tr_feat.append(torch.tensor(x[0]).float())
            else:
                tr_feat.append(x[0].clone().detach())
            tr_tgt.append(x[1])
            if i == batch_size or (len(tr_data) * batch_size + i == len(SData[0])):
                #print len(tr_feat)
                tr_feat = torch.stack(tr_feat)
                tr_tgt = torch.tensor(tr_tgt)
                #print tr_feat.size()
                #print  tr_tgt.size()

                if use_gpu == True:
                    tr_feat = tr_feat.cuda()
                    tr_tgt = tr_tgt.cuda()
                tr_data.append([tr_feat,tr_tgt])
                if "AE" in tmtype:
                    tr_ae_data.append(tr_feat)
                tr_feat = []
                tr_tgt = []
                i = 0
        #raw_input("sadfas")

        i = 0
        cv_feat = []
        cv_tgt = []
        cv_data = []
        for x in SData[1]:
            i = i + 1
            if type(x[0]) == type([]):
                cv_feat.append(torch.tensor(x[0]).float())
            else:
                cv_feat.append(x[0].clone().detach())
            cv_tgt.append(x[1])
            if i == batch_size or (len(cv_data) * batch_size + i == len(SData[1])):
                cv_feat = torch.stack(cv_feat)
                cv_tgt = torch.tensor(cv_tgt)
                 
                if use_gpu == True:
                    cv_feat = cv_feat.cuda()
                    cv_tgt = cv_tgt.cuda()
                cv_data.append([cv_feat,cv_tgt])
                cv_feat = []
                cv_tgt = []
                i = 0

        if tmtype == "MLP2L":
            model_nnet = MLP_ReLU(num_nodes = [in_dim, 256, 256, 2])
        elif tmtype == "MLP4L":
            model_nnet = MLP_ReLU(num_nodes = [in_dim, 256, 256, 256, 256, 2])
        elif tmtype == "MLPGAN4L":
            model_nnet = MLP_ReLU(num_nodes = [in_dim, 256, 256, 256, 256, 2])
        elif tmtype == "FNNRBB16L":
            model_nnet = FNN_RBB(num_nodes = [in_dim, [256, 16], 2])
        elif tmtype == "FNNRBB64L":
            model_nnet = FNN_RBB(num_nodes = [in_dim, [256, 64], 2])
        elif tmtype == "MLPAE4L":
            model_ae = AE_ReLU(num_nodes = [in_dim, 128, 64, 128, in_dim])
            if use_gpu == True:
                model_ae = model_ae.cuda()
            #model_ae = train_auto_encoder_newbob_tr(model_ae, tr_ae_data, max_epochs = 1000, init_lrate = 0.001, imp0 = 0.00001, imp1 = 0.000001)
            model_ae = train_auto_encoder_newbob_tr(model_ae, tr_ae_data, max_epochs = 10000, init_lrate = 0.001, imp0 = 0.0001, imp1 = 0.000001)
            model_nnet = MLP_AE_ReLU(model_ae, num_nodes = [in_dim, 256, 256, 256, 256, 2])

        if use_gpu == True:
            model_nnet = model_nnet.cuda()
        model_nnet = train_nnet_newbob_tr(model_nnet, tr_data, max_epochs = 1000, init_lrate = 0.01, imp0 = 0.0001, imp1 = 0.00001)
        test_nnet(model_nnet, tr_data)

        #raw_input("asdfas")

        if use_gpu == True:
            # After using GPU to perform the training task, we convert the trained model to CPU version.
            model_nnet = model_nnet.cpu()

        Acc = -1
        AUC = -1

        Threshold = 0.5 #SpecEqSens(RF,tr_feat,tr_tgt) #AveThre(RF,cv_feat,cv_tgt) # ReqSpec(RF,cv_feat,cv_tgt,0.9) #SpecEqSens(RF,cv_feat,cv_tgt) #0.5 #ReqSpec(RF,cv_feat,cv_tgt,0.7)


        model_nnet.MdlType = tmtype + "" 
        model_nnet.VList = VList
        model_nnet.SType = SType
        model_nnet.SemTypes = SemTypes
        model_nnet.Threshold = Threshold
        model_nnet.FeatType = feat_type
        model_nnet.classes_ = [0, 1]

        tmfile = resdir + "/semantics.mdl" 
        print "Writing %s tendency model (single) to %s."%(tmtype,tmfile)
        filehandler = open(tmfile, 'w')
        pickle.dump(model_nnet, filehandler)
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
        SM.Threshold = 0.5
        SM.FeatType = feat_type

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
# RL --- list of revisions
# wdir --- working directory
def ScoreRevisionsUsingSemanticsModel(W,RL,wdir):
    TL = []
    for x in RL:
        s = [x[1],x[2],x[4]]
        TL.append(s)
    res = PredictUsingSemanticsModel(W,TL,wdir)
    return res


# W --- semantics model
# RL --- list of insertions
# wdir --- working directory
def ScoreInsertionsUsingSemanticsModel(W,RL,wdir):
    TL = []
    for x in RL:
        s = x[4]
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
    FeatType = W.FeatType

    #DType = W.DType
    P = []
    sg = Bgenlib.BStateGraphForNN()
    PPD = sg.TransListToPrePostData(TL,SType,VList,feat_type=FeatType)


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

        if W.classes_[TgtIdx] != 0:
            print W.classes_, W.classes_[TgtIdx]
            raw_input("Index error in classifier.")
            pppp

        if MdlType in ["SKCART","BNBayes","LR","SVM","MLP"]:
        #if MdlType == "SKCART":
            Q = W.predict_proba(feat)
            P = []
            for x in Q:
                P.append(x[TgtIdx])
        
        elif MdlType in ["MLP2L", "MLP4L", "MLPGAN4L", "FNNRBB16L", "FNNRBB64L", "MLPAE4L"]:

            Q = []

            batch_size = 1024
            i = 0
            batch_feat = []
            num_feat = 0
            for x in feat:
                i = i + 1
                num_feat = num_feat + 1
                batch_feat.append(torch.tensor(x).float())
                if i == batch_size or num_feat == len(feat):
                    #print len(tr_feat)
                    batch_feat = torch.stack(batch_feat)
                    with torch.no_grad():
                        batch_output = W(batch_feat)
                        batch_output = numpy.exp(batch_output)
                        Q = Q + batch_output.tolist()

                    i = 0
                    batch_feat = []

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


