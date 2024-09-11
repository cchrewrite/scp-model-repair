import sys
import pydotplus
import pydot
import time
import os
import re
import subprocess
import random
import RepSimpLib
import SemLearnLib
import Bgenlib

from graphviz import Graph

# Count the number of goals satisfied.
# fname - model file
# G - list of goals. Each goal is of the form [g,n], where g is a goal predicate, and n is a number. It means that at least n states should satisfy p.
# conffile - configuration file
# wdir - working directory
# output - list of goals with numbers of states satisfying the goals. Each line in the list is of the form [g,n,s], where g and n have the same meanings as those in G, and s is the number of states satisfying g. The order of results are the same as the order in G.
def count_goals(fname,G,conffile,wdir):
    cmd = "mkdir %s"%wdir
    os.system(cmd)
    mchfile = wdir + "/original_machine.mch"
    cmd = "cp %s %s"%(fname,mchfile)
    os.system(cmd)
    M = wdir + "/original_machine.mch"
    cmd = "./../ProB/probcli -pp %s %s"%(M,mchfile)
    os.system(cmd)

    with open(M) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]

    FGL = []

    for i in xrange(len(G)):
        F_goal = "GOAL_%d_count = PRE %s THEN skip END"%(i,G[i][0])
        FGL.append(F_goal)

    i = len(mch) - 1
    while not("END" in mch[i]):
        i = i - 1
    mch[i] = ";"
    for j in xrange(len(FGL)):
        x = FGL[j] + ""
        if j < len(FGL) - 1:
            x = x + " ; "
        mch.append(x)
    mch.append("END")

    fn = wdir + "/count_goals.mch"
    f = open(fn,"w")
    for x in mch:
        f.write(x)
        f.write("\n")
    f.close()
    M = fn

    fn = wdir + "/count_goals_pp.mch"
    cmd = "./../ProB/probcli -pp %s %s"%(fn,M)
    os.system(cmd)
    M = fn

    with open(M) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]

    T = wdir + "/T.txt"
    max_initialisations = read_config(conffile,"max_initialisations","int")
    max_operations = read_config(conffile,"max_operations","int")
    bscope = generate_training_set_condition(mch)
    oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(M,max_initialisations,max_operations,bscope,T)
    os.system(oscmd)
    sg = Bgenlib.BStateGraphForNN()
    sg.ReadStateGraph(T)
    T = sg.GetTransList()

    SL = []
    for i in xrange(len(G)):
        ope = "GOAL_%d_count"%i
        s = 0
        for x in T:
            if x[1] == ope:
                q = x[0]
                for y in T:
                    if y[2] == q and not("GOAL_" in y[1] and "_count" in y[1]):
                        s = s + 1
        SL.append(G[i] + [s])
    return SL

# Count the number of tokens in a B-model:
# fname - model file
# wdir - working directory
# output - number of words
def count_words(fname,wdir):
    cmd = "mkdir %s"%wdir
    os.system(cmd)
    mchfile = wdir + "/count_tokens.mch"
    cmd = "cp %s %s"%(fname,mchfile)
    os.system(cmd)
    M = wdir + "/count_tokens_pp.mch"
    cmd = "./../ProB/probcli -pp %s %s"%(M,mchfile)
    os.system(cmd)
    f = open(M,"r")
    mch = ""
    for x in f.readlines():
        mch = mch + " " + x
    f.close()
    
    mch = mch.replace("\n","")
    mch = mch.split(" ")
    s = 0
    for x in mch:
        y = x.replace(" ","")
        if y != "":
            s = s + 1

    return s


# compute the size of a set of lists
# S --- a set of lists
# output --- its size
def size_of_a_set_of_lists(S):
    res = 0
    for x in S:
        res = res + len(x)
    return res

# align two sets of lists, and output their alignment degree
# S --- the first set
# T --- the second set
# output --- [W,WP,SD,TD,DEG], where W is the set of full alighments, WP is the set of partial alignments, SD is the set of remaining lists in S, TD is the set of remaining lists in T, and DEG is the degree of alignments.
def align_two_sets_of_lists(S,T):
    # W --- full alignment
    W = list_intersect(S,T)
    SD = list_difference(S,W)
    TD = list_difference(T,W)

    # WP --- partial alignment
    WP = []

    CS = []
    for i in xrange(len(SD)):
        x = SD[i]
        for j in xrange(len(TD)):
            y = TD[j]
            s = 0
            for k in xrange(min(len(x),len(y))):
                if x[k] == y[k]:
                    s = s + 1
            CS.append([i,j,s])

    CS.sort(key = lambda x:x[2], reverse=True)
    
    SDX = set([])
    TDX = set([])

    for p in CS:
        if p[0] in SDX:
            continue
        if p[1] in TDX:
            continue
        SDX.add(p[0])
        TDX.add(p[1])
        WP.append([SD[p[0]],TD[p[1]],p[2]])
        
    """
    while True:
        s = 0
        x = "None"
        y = "None"
        for xt in SD:
            for yt in TD:
                st = 0
                for i in xrange(min(len(xt),len(yt))):
                    if xt[i] == yt[i]:
                        st = st + 1
                if st > s:
                    s = st
                    x = xt
                    y = yt
        if s == 0: break
        WP.append([x,y,s])
        SD.remove(x)
        TD.remove(y)
    """
    DEG = 0
    for x in W:
        DEG = DEG + len(x)
    for x in WP:
        DEG = DEG + x[2]

    return [W,WP,SD,TD,DEG]
        


# In US, each case is of the form [cond,subs]
def generate_select_when_substitution(US):
    res = []
    for i in xrange(len(US)):
        X = US[i]
        if i == 0: 
            res.append("SELECT " + X[0] + " THEN")
        else:
            res.append("WHEN " + X[0] + " THEN")
        res.append(X[1])
        if i == len(US)-1:
            res.append("END")
    return res
           

# TL --- list of transitions
def check_set_order_in_transitions(TL):
    for X in TL:
        for Y in X[0] + X[2]:
            if type(Y) == type([]):
                if Y != sorted(Y):
                    print "Error: set are not ordered:"
                    print X
                    return False
    return True 


# Update state diagrams during reachability repair
# SI --- set of initial states
# DS --- original state diagram
# VT --- valid transitions
def update_reachability(SI,DS,VT):
    DT = DS + []
    for P in SI:
        for X in VT:
            V = X[0]
            if V[0] == P and not(V in DT):
                DT.append(V)
    print len(DS),len(DT)
    while True:
        DX = []
        for Y in DT:    
            for X in VT:
                V = X[0]
                if V[0] == Y[2] and not(V in DT) and not(V in DX):
                    DX.append(V)
        DT = DT + DX
        if DX == []:
            break
    print len(DS),len(DT)
    return DT



# Delete a number of transitions leading to a goal state.
# D --- state diagram.
# G --- goal state.
# RN --- the number of removed transitions. If RN == "ALL", then delete all transitions to the goal state.
def delete_a_number_of_transitions(D,G,RN):

    GT = []
    for X in D:
        if X[2] == G:
            GT.append(X)

    if RN == "ALL":
        LDT = GT
    else:
        random.shuffle(GT)
        LDT = GT[0:min(RN,len(GT))]

    LNT = list_difference(D,LDT)
    
    # LNT --- Normal Transitions
    # LDT --- Deleted Transitions

    return [LNT,LDT]






# Delete partial transitions leading to a goal state.
# D --- state diagram.
# G --- goal state.
# PD --- probability of deletion.
# RD --- depth of deletion
def deleted_partial_transitions(D,G,PD,RD):

    if RD <= 0:
        return [D,[]]

    # LNT --- Normal Transitions
    # WL --- Nodes that in-edges should be removed.
    # LDT --- Deleted Transitions
    LNT = []
    WL = []
    LDT = []
    flag = None
    for X in D:
        flag = "norm"
        if X[2] == G:
            flag = "kept"
            p = random.random()
            if p <= PD:
                flag = "del"

        
        if flag == "norm":
            LNT.append(X)
            continue
        elif flag == "kept":
            LNT.append(X)
            if not(X[0] in WL):
                WL.append(X[0])
            continue
        elif flag == "del":
            LDT.append(X)
            if not(X[0] in WL):
                WL.append(X[0])
            continue
        else:
            print "Error !!!"
            ppppp

    for GX in WL:
        S = deleted_partial_transitions(LNT,GX,PD,RD-1)
        LNT = S[0]
        LDT = list_union(LDT,S[1])

    return [LNT,LDT]




# Delete transitions leading to a goal state.
# D --- state diagram.
# G --- goal state.
# PD --- probability of deletion.
# RD --- depth of deletion
def deleted_transitions(D,G,PD,RD):

    if RD <= 0:
        return [D,[]]

    has_in_edge = []
    for X in D:
        if not(X[2] in has_in_edge):
            has_in_edge.append(X[2])

    # LNT --- Normal Transitions
    # WL --- Nodes that in-edges should be removed.
    # LDT --- Deleted Transitions
    LNT = []
    WL = []
    LDT = []
    for X in D:
        flag = "norm"
        if X[2] == G:
            flag = "kept"
            p = random.random()
            if p <= PD:
                flag = "del"
            elif not(X[0] in has_in_edge):
                flag = "del"
            elif RD == 1:
                flag = "del"

        if flag == "norm":
            LNT.append(X)
            continue
        elif flag == "kept":
            LNT.append(X)
            if not(X[0] in WL):
                WL.append(X[0])
            continue
        elif flag == "del":
            LDT.append(X)
            continue
        else:
            print "Error !!!"
            ppppp

    for GX in WL:
        S = deleted_transitions(LNT,GX,PD,RD-1)
        LNT = S[0]
        LDT = list_union(LDT,S[1])

    return [LNT,LDT]

# randomly select an item.
# S --- a list
# N --- number of selected item
def random_selection(S,N):
    X = S + []
    random.shuffle(X)
    res = X[0:N]
    return res

# fn --- oracle file name
def read_oracle_file(fn):
    f = open(fn,"r")
    G = f.readlines()
    GD = []
    for x in G:
        y = x.replace("\n","")
        if y.replace(" ","") != "":
            GD.append(y)

    # HD --- Head of Oracle
    # BD --- Body of Oracle
    HD = []
    BD = []

    # Find Answer
    flag = 0
    RD = []
    for X in GD:
        if X.replace(" ","") == "BEGIN_ANSWER":
            flag = 1
            continue
        if X.replace(" ","") == "END_ANSWER":
            flag = 2
            break
        if flag == 0:
            continue
        # The following cases satisfy flag == 1.
        if X.replace(" ","")[0:6] == "FORMAT":
            continue
        RD.append(eval(X))
    if flag == 2:
        print "ANSWER section obtained in oracle."        
        HD.append("ANSWER")
        BD.append(RD)
  
    # Find Revision Answer
    flag = 0
    RD = []
    for X in GD:
        if X.replace(" ","") == "BEGIN_REVISION":
            flag = 1
            continue
        if X.replace(" ","") == "END_REVISION":
            flag = 2
            break
        if flag == 0:
            continue
        # The following cases satisfy flag == 1.
        if X.replace(" ","")[0:6] == "FORMAT":
            continue
        RD.append(eval(X))
    if flag == 2:
        print "REVISION section obtained in oracle."        
        HD.append("REVISION")
        BD.append(RD)
   

 
    # Find Isolation Answer
    flag = 0
    RD = []
    for X in GD:
        if X.replace(" ","") == "BEGIN_ISOLATION":
            flag = 1
            continue
        if X.replace(" ","") == "END_ISOLATION":
            flag = 2
            break
        if flag == 0:
            continue
        # The following cases satisfy flag == 1.
        if X.replace(" ","")[0:6] == "FORMAT":
            continue
        RD.append(eval(X))
    if flag == 2:
        print "ISOLATION section obtained in oracle."        
        HD.append("ISOLATION")
        BD.append(RD)
   

    # Find Missing Transitions
    flag = 0
    RD = []
    for X in GD:
        if X.replace(" ","") == "BEGIN_MISSING_TRANSITIONS":
            flag = 1
            continue
        if X.replace(" ","") == "END_MISSING_TRANSITIONS":
            flag = 2
            break
        if flag == 0:
            continue
        # The following cases satisfy flag == 1.
        if X.replace(" ","")[0:6] == "FORMAT":
            continue
        RD.append(eval(X))
    if flag == 2:
        print "MISSING_TRANSITIONS section obtained in oracle."
        HD.append("MISSING_TRANSITIONS")
        BD.append(RD)
   


 
    for i in xrange(len(HD)):
        print HD[i]
        for j in xrange(len(BD[i])):
            print BD[i][j]
    
    return [HD,BD]

# LC --- list of CFG changes, each change is of the form [operation,condition,substitution].
def convert_CFG_changes_to_substitutions(LC):
    FL = []
    for x in LC:
        if not(x[0] in LC):
            FL.append(x[0])
    res = []
    for F in FL:
        cond = ""
        subs = ""
        for x in LC:
            if not(x[0] == F):
                continue
            if cond != "":
                cond = cond + " or " 
            cond = cond + x[1]
            if subs != "":
                subs = subs + " ELSIF "
            else:
                subs = "IF "
            subs = subs + x[1] + " THEN " + x[2]
        if subs != "":
            subs = subs + " END"
            res.append([F,cond,subs])
    return res


# LC --- list of transition changes
# VL --- list of variables
def convert_changes_to_substitutions(LC,VL):
    SL = []

    for x in LC:
        op = x[1]
        P = x[0]
        R = x[3]
        
        cond = ""
        for i in xrange(len(VL)):
            V = VL[i]
            X = P[i]
            if type(X) == type([]):
                X = convert_python_set_to_b_string(X)
            VeX = V + " = " + X
            if cond != "":
                cond = cond + " & "
            cond = cond + VeX

        subs = ""
        for i in xrange(len(VL)):
            V = VL[i]
            X = R[i]
            if type(X) == type([]):
                X = convert_python_set_to_b_string(X)
            VeX = V + " := " + X
            if subs != "":
                subs = subs + " ; "
            subs = subs + VeX

        s = [cond,subs] #"IF " + cond + " THEN " + subs + " END"
        SL.append([op] + s)
    SL.sort()
 
    res = []
    for i in xrange(len(SL)+1):

        if i == len(SL):
            # all items have been converted, break.
            if i > 0:
                res.append([op,cond,subs])
            break

        x = SL[i]
        if i == 0 or (i > 0 and i < len(SL) and SL[i][0] != SL[i-1][0]):
            if i > 0:
                res.append([op,cond,subs])
            op = x[0]
            cond = ""
            subs = ""
        
        if cond != "":
            cond = cond + " or "
        if subs != "":
            subs = subs + " ; "

        print x[1]
        cond = cond + "(" + x[1] + ")"
        s = "IF " + x[1] + " THEN " + x[2] + " END"
        subs = subs + s
           
    return res
    

# Add an invariant to a B-Machine.
# mch --- A pretty-printed B-Machine.
# inv --- An invariant
def add_invariant_to_machine(mch,inv):
    res = []
    i = 0
    while get_first_token(mch[i]) != "INVARIANT":
        res.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    if i < mchlen:
        res.append(mch[i])
    j = i + 1
    while j < mchlen:
        tt = get_first_token(mch[j])
        # Based on the syntax of <The B-book>, p.273.
        if tt == "ASSERTIONS": break
        if tt == "DEFINITIONS": break
        if tt == "INITIALIZATION": break
        if tt == "INITIALISATION": break
        if tt == "OPERATIONS": break
        if tt == "END": break
        res.append(mch[j])
        j = j + 1
    if res == []:
        print "Warning: No invariant found! Return [\'TRUE\']."
        return ['TRUE']
    res.append("& " + inv)
    while j < mchlen:
        res.append(mch[j])
        j = j + 1
    return res



# Z --- set of states
# VList --- list of variables
def convert_states_to_conditions(Z,VList):
    res = ""
    for P in Z:
        s = ""
        for i in xrange(len(VList)):
            V = VList[i]
            X = P[i]
            if type(X) == type([]):
                X = convert_python_set_to_b_string(X)
            #print V
            #print X
            VeX = V + " = " + X
            if s != "":
                s = s + " & "
            s = s + VeX
        s = "(" + s + ")"
        if res != "":
            res = res + " or "
        res = res + s
    return res


# Z --- set of prohibited states
# VList --- list of variables
def convert_prohibited_states_to_invariants(Z,VList):
    res = ""
    for P in Z:
        s = ""
        for i in xrange(len(VList)):
            V = VList[i]
            X = P[i]
            if type(X) == type([]):
                X = convert_python_set_to_b_string(X)
            #print V
            #print X
            VeX = V + " = " + X
            if s != "":
                s = s + " & "
            s = s + VeX
        s = "(" + s + ")"
        if res != "":
            res = res + " or "
        res = res + s
    res = "not(" + res + ")"
    return res

def CanRepresentInt(s):
    if not(type(s) in [type(10),type("aaa")]):
        return False
    try: 
        int(s)
        return True
    except ValueError:
        return False

# Randomly generate prohibited states. i.e. states not in S
# S --- set of available states
# N --- maximum number of prohibited states
def generate_prohibited_states(S,N):
    
    VS = []
    L = len(S[0])
    for x in xrange(L):
        VS.append([])
    for P in S:
        for j in xrange(L):
            if not(P[j] in VS[j]):
                VS[j].append(P[j])
    for i in xrange(len(VS)):
        if CanRepresentInt(VS[i][0]) == False:
            continue
        else:
            IntS = []
            for x in VS[i]:
                IntS.append(int(x))
            MaxInt = max(IntS)
            MinInt = min(IntS)
            VS[i] = [str(MinInt-2),str(MinInt-1)] + VS[i] + [str(MaxInt+1),str(MaxInt+2)]

    R = []
    for i in xrange(1000000):
        if i >= N and R != []:
            # try N times. if found results, then break. otherwise, continue to find a result until one result is found
            break
        X = []
        for j in xrange(L):
            u = int(random.random() * len(VS[j]))
            X.append(VS[j][u])
        if not(X in S) and not(X in R):
            R.append(X)
    if R == []:
        print "WARNING: Failed to make prohibited states."
    #print "xxx"
    #for x in R: print x
    
    return R
            
    

# get the pth column of a list
def get_list_column(L,p):
    res = []
    for X in L:
        res.append(X[p])
    return res

# Compute the union of two lists.
# i.e. P + Q
def list_union(P,Q):
    res = []
    R = P + Q
    R.sort()
    for i in xrange(len(R)-1):
        if R[i] != R[i+1]:
            res.append(R[i])
    if len(R) > 0:
        res.append(R[-1])
    return res
    

# Compute the intersect of two lists.
# i.e. P * Q
def list_intersect(P,Q):
    res = []
    for x in P:
        if x in Q and not(x in res):
            res.append(x)
    return res

# Compute the difference of two lists.
# i.e. P - Q
def list_difference(P,Q):
    res = []
    for x in P:
        if not(x in Q) and not(x in res):
            res.append(x)
    return res


# Remove duplicate elements in a list.
def remove_duplicate_elements(L):
    R = sorted(L)
    res = []
    for i in xrange(len(R)):
        if i == 0:
           res.append(R[0])
           continue
        if R[i] != R[i-1]:
            res.append(R[i])
    return res
    
# Convert a python set to a B-string.
def convert_python_set_to_b_string(x):
    res = ""

    xt = sorted(x)

    for p in xt:
        res = res + str(p) + " , "
    
    if res != "":
        res = res[0:len(res)-3]

    res = "{ " + res + " }"
    return res


def apply_deterministic_modification(mch,ope,cond,subs):
    return apply_A_change(mch,ope,cond,subs)

def apply_deterministic_insertion(mch,ope,cond,subs):
    mch1 = add_precond_to_mch(mch,ope,cond)
    mch2 = apply_A_change(mch1,ope,cond,subs)
    return mch2

def apply_deterministic_deletion(mch,ope,cond):
    neg_cond = "not(%s)"%cond
    return add_precond_to_mch(mch,ope,neg_cond)

# Apply a A-change to a machine.
# mch --- a pretty-printed machine
# ope --- operation name
# cond --- condition
# subs --- substitution
def apply_A_change(mch,ope,cond,subs):

    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        y.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    opeflag = 0
    while i < mchlen:
        if not(ope in mch[i]) or not(proc_opes_decl(mch[i])[0] == ope) or (ope in mch[i] and ":=" in mch[i]):
            y.append(mch[i])
            i = i + 1
            continue
        opeflag = opeflag + 1
        if True:
            if get_first_token(mch[i+1]) == "skip":
                y.append(mch[i])
                y.append("BEGIN")
                y.append("IF not( %s )"%cond)
                y.append("THEN")
                y.append("skip")
                y.append("ELSE")
                y.append(subs)
                y.append("END")
                if ";" in mch[i+1]:
                    y.append("END;")
                else:
                    y.append("END")
                i = i + 2
            elif get_first_token(mch[i+1]) == "PRE":
                y.append(mch[i])
                y.append(mch[i+1])
                i = i + 2
                while get_first_token(mch[i]) != "THEN":
                    y.append(mch[i])
                    i = i + 1
                y.append(mch[i])
                i = i + 1
                subs_org = []
                flag = 1
                j = i 
                
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                y.append("IF not( %s )"%cond)
                y.append("THEN")
                subs_org = mch[i:j-1]
                """
                print "FULL MCH"
                for x in y: print x
                print "SUBS: ",subs 
                print "PPPP"
                print subs_org
                print "QQQQ"
                print mch[j-2]
                print mch[j-1]
                print mch[j]
                #pppp
                """ 
                y = y + subs_org
                y.append("ELSE")
                y.append(subs)
                y.append("END")
                y.append(mch[j-1])
                #y.append(mch[j]) 
                
                i = j# + 1
                

                #y.append(cond + " &")
                #i = i + 2
            else:
                flag = 1
                j = i + 2
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                y.append(mch[i])

                #y.append("PRE")
                y.append("IF")
                y.append("not( %s )"%cond)
                y.append("THEN")
                k = i + 1
                while k < j - 1:
                    y.append(mch[k])
                    k = k + 1
                y.append("END")
                y.append("ELSE")
                y.append(subs)
                y.append(mch[j-1])

                """
                if y[-1] == "END;":
                    y[-1] = "END"
                    y.append("END;")
                else:
                    y.append("END")
                """

                i = j
    if opeflag != 1:
        print "Error: %d operations named %s found!"%(opeflag,ope)
        exit()
    return y



# Apply a A-change to a machine.
# mch --- a pretty-printed machine
# ope --- operation name
# cond --- condition
# subs --- substitution
def apply_A_change_v2(mch,ope,cond,subs):

    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        y.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    opeflag = 0
    while i < mchlen:
        if not(ope in mch[i]) or not(proc_opes_decl(mch[i])[0] == ope) or (ope in mch[i] and ":=" in mch[i]):
            y.append(mch[i])
            i = i + 1
            continue
        opeflag = opeflag + 1
        if True:
            if get_first_token(mch[i+1]) == "skip":
                y.append(mch[i])
                y.append(subs[0:len(subs)-4])
                if ";" in mch[i+1]:
                    y.append("END;")
                else:
                    y.append("END")
                i = i + 2
            elif get_first_token(mch[i+1]) == "PRE":
                y.append(mch[i])
                y.append(mch[i+1])
                i = i + 2
                while get_first_token(mch[i]) != "THEN":
                    y.append(mch[i])
                    i = i + 1
                y.append(mch[i])
                i = i + 1
                subs_org = []
                flag = 1
                j = i 
                
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                ss = subs[0:len(subs)-4] + " ELSE"
                y.append(ss)
                subs_org = mch[i:j-1]
                y = y + subs_org
                y.append("END")
                y.append(mch[j-1])
                #y.append(mch[j]) 
                
                i = j# + 1
                

                #y.append(cond + " &")
                #i = i + 2
            else:
                flag = 1
                j = i + 2
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                y.append(mch[i])

                #y.append("PRE")
                ss = subs[0:len(subs)-4] + " ELSE"
                y.append(ss)
                k = i + 1
                while k < j - 1:
                    y.append(mch[k])
                    k = k + 1
                y.append(mch[j-1])
                i = j
    if opeflag != 1:
        print "Error: %d operations named %s found!"%(opeflag,ope)
        exit()
    return y

def split_M_changes_by_operations(LMT):
    res = []
    OpeList = []
    for X in LMT:
        if not(X[1] in OpeList):
            OpeList.append(X[1])
    for ope in OpeList:
        S = []
        for X in LMT:
            if X[1] == ope:
                S.append(X)
        S.sort()
        res.append([ope,S])
    return res


# Apply a M-change (multiple non-deterministic isolation) to a machine.
# mch --- a pretty-printed machine
# ope --- operation name
# LMT --- list of missing transitions
# VList --- list of variables
def apply_M_change(mch,ope,LMT,VList):

    PreVList = []
    for x in VList:
        y = x + ""
        y = y.replace("(","_lp_")
        y = y.replace(")","_rp_")
        y = "pre_" + y
        PreVList.append(y)

    RCFlag = "r_cond_flag"
    VARList = ""
    for v in PreVList:
        VARList = VARList + v + ", "
    VARList = VARList + RCFlag

    print VARList

    CGL = []
    for X in LMT:
        P = X[0]
        Q = X[2]

        PreCond = ""
        for i in xrange(len(PreVList)):
            if type(P[i]) == type([]):
                Y = convert_python_set_to_b_string(P[i])
            else:
                Y = P[i]
            rs = "%s = %s & "%(PreVList[i],Y)
            PreCond = PreCond + rs
        PreCond = PreCond[0:len(PreCond)-3]

        PostCond = ""            
        for i in xrange(len(VList)):
            if type(Q[i]) == type([]):
                Y = convert_python_set_to_b_string(Q[i])
            else:
                Y = Q[i]
            rs = "%s = %s & "%(VList[i],Y)
            PostCond = PostCond + rs
        PostCond = PostCond[0:len(PostCond)-3]

        IsoCond = PreCond + " & " + PostCond
        CGL.append(IsoCond)

    for x in CGL: print x

    # PS --- pre-assignment strings
    PS = ""
    for i in xrange(len(VList)):
        x = "%s := %s ; "%(PreVList[i],VList[i])
        PS = PS + x
    PS = PS[0:len(PS)-3]
        
 
    # RS --- repair strings
    RS = []
    RS.append("IF")
    for X in CGL:
        cond = "(%s) or "%X
        RS.append(cond)
    cond = RS[len(RS)-1] + ""
    RS[len(RS)-1] = cond[0:len(cond)-4]
    RS.append("THEN")
    IsoSubs = "%s := 1 ; %s : (TRUE = FALSE)"%(RCFlag,RCFlag)
    RS.append(IsoSubs)
    RS.append("END")
    
        

    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        y.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    opeflag = 0
    while i < mchlen:
        if not(ope in mch[i]) or not(proc_opes_decl(mch[i])[0] == ope) or (ope in mch[i] and ":=" in mch[i]):
            y.append(mch[i])
            i = i + 1
            continue
        opeflag = opeflag + 1
        if True:
            if get_first_token(mch[i+1]) == "skip":
                y.append(mch[i])
                y.append("VAR %s IN"%VARList)
                y.append(PS + " ;")
                #y.append("skip;")
                y = y + RS
                if ";" in mch[i+1]:
                    y.append("END;")
                else:
                    y.append("END")
                i = i + 2
            elif get_first_token(mch[i+1]) == "PRE":
                y.append(mch[i])
                y.append(mch[i+1])
                i = i + 2
                while get_first_token(mch[i]) != "THEN":
                    y.append(mch[i])
                    i = i + 1
                y.append(mch[i])
                i = i + 1
                subs_org = []
                flag = 1
                j = i 
                
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                #y.append("IF not( %s )"%cond)
                #y.append("THEN")
                subs_org = mch[i:j]

                y.append("VAR %s IN"%VARList)
                y.append(PS + " ;")
                y = y + subs_org
                #y.append("END ;")
                if get_first_token(y[len(y)-1]) == "END":
                    y[len(y)-1] = "END;"
                else:
                    y[len(y)-1] = y[len(y)-1] + ";"
                y = y + RS

                if ";" in mch[j-1]:
                    y.append("END;")
                else:
                    y.append("END")

                i = j# + 1
                

                #y.append(cond + " &")
                #i = i + 2
            else:
                flag = 1
                j = i + 2
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                y.append(mch[i])

                subs_org = mch[i+1:j]
                y.append("VAR %s IN"%VARList)
                y.append(PS + " ;")
                y = y + subs_org
                #y.append("END ;")
                if get_first_token(y[len(y)-1]) == "END":
                    y[len(y)-1] = "END;"
                else:
                    y[len(y)-1] = y[len(y)-1] + ";"
                y = y + RS

                if ";" in mch[j-1]:
                    y.append("END;")
                else:
                    y.append("END")

                i = j
    if opeflag != 1:
        print "Error: %d operations named %s found!"%(opeflag,ope)
        exit()
    return y



# convert an atomic modification to a conditional substitution
# X --- an atomic modification of the form [P,F,Q,R], where P is a pre-state, F is an operation, Q is a post-state, and R is a modification state or a deletion signal.
# VList --- Variable list
# epid --- The ID of epoch
# The output is a list of string [F,S,T], where F is the operation, S is a condition, and T is a substitution.
def atomic_modification_or_deletion_to_conditional_substitution(X,VList,epid):

    PreVList = []
    for x in VList:
        y = x + ""
        y = y.replace("(","_lp_")
        y = y.replace(")","_rp_")
        y = "%s_pre_%s"%(y,epid)
        PreVList.append(y)

    P = X[0]
    PS = ""
    for i in xrange(len(PreVList)):
        if type(P[i]) == type([]):
            Y = convert_python_set_to_b_string(P[i])
        else:
            Y = P[i]
        rs = "%s = %s & "%(PreVList[i],Y)
        PS = PS + rs

    Q = X[2]
    QS = ""
    for i in xrange(len(VList)):
        if type(Q[i]) == type([]):
            Y = convert_python_set_to_b_string(Q[i])
        else:
            Y = Q[i]
        rs = "%s = %s & "%(VList[i],Y)
        QS = QS + rs

    Cond = PS + QS
    Cond = Cond[0:len(Cond)-3]
 
    R = X[3]
    if R != "isolation":
        RS = ""
        for i in xrange(len(VList)):
            if type(R[i]) == type([]):
                Y = convert_python_set_to_b_string(R[i])
            else:
                Y = R[i]
            rx = "%s := %s ; "%(VList[i],Y)
            RS = RS + rx
        Subs = RS[0:len(RS)-3]
    else:
        # Method 1: Using a falsity substitution.
        RCFlag = "del_flag_%s"%epid
        Subs = "%s : (TRUE = FALSE)"%(RCFlag)

        """
        # Method 2: Create a self transition.
        Subs = ""
        for i in xrange(len(VList)):
            Subs = Subs + "%s := %s ; "%(VList[i],PreVList[i])
        Subs = Subs[0:len(Subs)-3]
        """

    F = X[1]
    res = [F,Cond,Subs]

    return res



# Apply modifications and deletions to a machine.
# mch --- a pretty-printed machine
# ope --- operation name
# RL --- list of repairs, and each repair is of the form [P,Q], where P is a condition, and Q is a substitution.
# VList --- list of variables.
# epid --- The ID of epoch
def apply_modifications_and_deletions(mch,ope,RL,VList,epid):

    PreVList = []
    for x in VList:
        y = x + ""
        y = y.replace("(","_lp_")
        y = y.replace(")","_rp_")
        y = "%s_pre_%s"%(y,epid)
        PreVList.append(y)

    RCFlag = "del_flag_%s"%epid
    VARList = ""
    for v in PreVList:
        VARList = VARList + v + ", "
    VARList = VARList + RCFlag

    # PS --- pre-assignment strings
    PS = ""
    for i in xrange(len(VList)):
        x = "%s := %s ; "%(PreVList[i],VList[i])
        PS = PS + x
    PS = PS + "%s := 0"%RCFlag
        
 
    # RS --- repair strings
    RS = []
    for X in RL:
        RS.append("IF")
        RS.append(X[0])
        RS.append("THEN")
        RS.append(X[1])
        RS.append("END;")
    RS[len(RS)-1] = "END"


    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        y.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    opeflag = 0
    while i < mchlen:
        if not(ope in mch[i]) or not(proc_opes_decl(mch[i])[0] == ope) or (ope in mch[i] and ":=" in mch[i]):
            y.append(mch[i])
            i = i + 1
            continue
        opeflag = opeflag + 1
        if True:
            if get_first_token(mch[i+1]) == "skip":
                y.append(mch[i])
                y.append("VAR %s IN"%VARList)
                y.append(PS + " ;")
                #y.append("skip;")
                y = y + RS
                if ";" in mch[i+1]:
                    y.append("END;")
                else:
                    y.append("END")
                i = i + 2
            elif get_first_token(mch[i+1]) == "PRE":
                y.append(mch[i])
                y.append(mch[i+1])
                i = i + 2
                while get_first_token(mch[i]) != "THEN":
                    y.append(mch[i])
                    i = i + 1
                y.append(mch[i])
                i = i + 1
                subs_org = []
                flag = 1
                j = i 
                
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                #y.append("IF not( %s )"%cond)
                #y.append("THEN")
                subs_org = mch[i:j-1]

                y.append("VAR %s IN"%VARList)
                y.append(PS + " ;")
                y = y + subs_org
                #y.append("END ;")
                if get_first_token(y[len(y)-1]) == "END":
                    y[len(y)-1] = "END;"
                else:
                    y[len(y)-1] = y[len(y)-1] + ";"
                y = y + RS
                y.append("END")

                if ";" in mch[j-1]:
                    y.append("END;")
                else:
                    y.append("END")
 
                i = j# + 1
                

                #y.append(cond + " &")
                #i = i + 2
            else:
                flag = 1
                j = i + 2
                if get_first_token(mch[i+1]) not in ["BEGIN","PRE","IF","CHOICE","ANY","SELECT","VAR","LET"]: flag = 0 # C1
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
 
                y.append(mch[i])

                subs_org = mch[i+1:j] # C2
                y.append("VAR %s IN"%VARList)
                y.append(PS + " ;")
                y = y + subs_org
                #y.append("END ;") # C3
                if get_first_token(y[len(y)-1]) == "END":
                    y[len(y)-1] = "END;"
                else:
                    y[len(y)-1] = y[len(y)-1] + ";"
                    y[len(y)-1] = y[len(y)-1].replace(";;",";") # C4
                y = y + RS
                #y.append("END") # C5
                if ";" in mch[j-1]:
                    y.append("END;")
                else:
                    y.append("END")

                """
                if y[-1] == "END;":
                    y[-1] = "END"
                    y.append("END;")
                else:
                    y.append("END")
                """

                i = j

    if opeflag != 1:
        print "Error: %d operations named %s found!"%(opeflag,ope)
        exit()
    return y





# Apply a S-change to a machine.
# mch --- a pretty-printed machine
# ope --- operation name
# chng --- change
def apply_S_change(mch,ope,chng):

    ssubs = generate_select_when_substitution(chng)
    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        y.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    opeflag = 0
    while i < mchlen:
        if not(ope in mch[i]) or not(proc_opes_decl(mch[i])[0] == ope) or (ope in mch[i] and ":=" in mch[i]):
            y.append(mch[i])
            i = i + 1
            continue
        opeflag = opeflag + 1
        if True:
            if get_first_token(mch[i+1]) == "skip":
                y.append(mch[i])
                y = y + ssubs
                y[len(y)-1] = "WHEN TRUE : BOOL THEN"
                y.append("skip")
                if ";" in mch[i+1]:
                    y.append("END;")
                else:
                    y.append("END")
                i = i + 2
            elif get_first_token(mch[i+1]) == "PRE":
                y.append(mch[i])
                y.append(mch[i+1])
                y[len(y)-1] = y[len(y)-1].replace("PRE","SELECT")
                i = i + 2
                while get_first_token(mch[i]) != "THEN":
                    y.append(mch[i])
                    i = i + 1
                y.append(mch[i])
                i = i + 1
                subs_org = []
                flag = 1
                j = i 
                
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                subs_org = mch[i:j-1]               
                y = y + subs_org
                y.append(ssubs[0].replace("SELECT","WHEN"))
                y = y + ssubs[1:len(ssubs)]
                y[len(y)-1] = mch[j-1]
                i = j
            else:
                flag = 1
                j = i + 2
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                y.append(mch[i])
                i = i + 1
                #y.append("PRE")
                y.append("SELECT TRUE : BOOL THEN")
                subs_org = mch[i:j-1]
                y = y + subs_org
                y.append("END")
                y.append(ssubs[0].replace("SELECT","WHEN"))
                y = y + ssubs[1:len(ssubs)]
                y[len(y)-1] = mch[j-1]
                i = j
    if opeflag != 1:
        print "Error: %d operations named %s found!"%(opeflag,ope)
        exit()
    return y


# Apply insertions to a machine.
# mch --- a pretty-printed machine
# ope --- operation name
# chng --- change
def apply_insertions(mch,ope,chng):
    scond = "( " + chng[0][0] + " )"
    for i in xrange(1,len(chng)):
        scond = scond + " or ( " + chng[i][0] + " )"
    ssubs = generate_select_when_substitution(chng)
    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        y.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    opeflag = 0
    while i < mchlen:
        if not(ope in mch[i]) or not(proc_opes_decl(mch[i])[0] == ope) or (ope in mch[i] and ":=" in mch[i]):
            y.append(mch[i])
            i = i + 1
            continue
        opeflag = opeflag + 1
        if True:
            if get_first_token(mch[i+1]) == "skip":
                y.append(mch[i])
                y = y + ssubs
                y[len(y)-1] = "WHEN TRUE : BOOL THEN"
                y.append("skip")
                if ";" in mch[i+1]:
                    y.append("END;")
                else:
                    y.append("END")
                i = i + 2
            elif get_first_token(mch[i+1]) == "PRE":
                y.append(mch[i])
                y.append(mch[i+1])
                y[len(y)-1] = y[len(y)-1] + " ( "
                i = i + 2
                while get_first_token(mch[i]) != "THEN":
                    y.append(mch[i])
                    i = i + 1

                y[len(y)-1] = y[len(y)-1] + " ) or ( %s )"%scond
                y.append(mch[i])
                y.append("SELECT TRUE : BOOL THEN")
                i = i + 1
                subs_org = []
                flag = 1
                j = i 
                
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                subs_org = mch[i:j-1]               
                y = y + subs_org
                y.append(ssubs[0].replace("SELECT","WHEN"))
                y = y + ssubs[1:len(ssubs)]
                y.append("END")
                y[len(y)-1] = mch[j-1]
                i = j
            else:
                flag = 1
                j = i + 2
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1

                y.append(mch[i])
                i = i + 1
                #y.append("PRE")
                y.append("SELECT TRUE : BOOL THEN")
                subs_org = mch[i:j-1]
                y = y + subs_org
                y.append("END")
                y.append(ssubs[0].replace("SELECT","WHEN"))
                y = y + ssubs[1:len(ssubs)]
                y[len(y)-1] = mch[j-1]
                i = j
    if opeflag != 1:
        print "Error: %d operations named %s found!"%(opeflag,ope)
        exit()
    return y








# Apply a consonance repair to a machine.
# mch --- a pretty-printed machine
# ope --- operation name
# cond --- condition
# subs --- substitution
def update_consonance(mch,ope,cond,subs):

    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        y.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    opeflag = 0
    while i < mchlen:
        if not(ope in mch[i]) or not(proc_opes_decl(mch[i])[0] == ope):
            y.append(mch[i])
            i = i + 1
            continue
        opeflag = opeflag + 1
        if True:
            if get_first_token(mch[i+1]) == "skip":
                y.append(mch[i])
                y.append("BEGIN")
                y.append("IF not( %s )"%cond)
                y.append("THEN")
                y.append("skip")
                y.append("ELSE")
                y.append(subs)
                y.append("END")
                if ";" in mch[i+1]:
                    y.append("END;")
                else:
                    y.append("END")
                i = i + 2
            elif get_first_token(mch[i+1]) == "PRE":
                y.append(mch[i])
                y.append(mch[i+1])
                i = i + 2
                while get_first_token(mch[i]) != "THEN":
                    y.append(mch[i])
                    i = i + 1
                y.append(mch[i])
                i = i + 1
                subs_org = []
                flag = 1
                j = i# + 1 
                
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                y.append("IF not( %s )"%cond)
                y.append("THEN")
                subs_org = mch[i:j-1]
                y = y + subs_org
                y.append("ELSE")
                y = y + subs
                y.append("END")
                y.append(mch[j-1])
                #y.append(mch[j])
                
                i = j# + 1
                

                #y.append(cond + " &")
                #i = i + 2
            else:
                flag = 1
                j = i + 2
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                y.append(mch[i])

                #y.append("PRE")
                y.append("IF")
                y.append("not( %s )"%cond)
                y.append("THEN")
                k = i + 1
                while k < j - 1:
                    y.append(mch[k])
                    k = k + 1
                y.append("END")
                y.append("ELSE")
                y = y + subs
                y.append(mch[j-1])

                """
                if y[-1] == "END;":
                    y[-1] = "END"
                    y.append("END;")
                else:
                    y.append("END")
                """

                i = j
    if opeflag != 1:
        print "Error: %d operations named %s found!"%(opeflag,ope)
        exit()
    return y



# Apply an insertion repair to a machine.
# mch --- a pretty-printed machine
# ope --- operation name
# cond --- condition
# subs --- substitution
def update_insertion(mch,ope,cond,subs):

    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        y.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    opeflag = 0
    while i < mchlen:
        if not(ope in mch[i]) or not(proc_opes_decl(mch[i])[0] == ope):
            y.append(mch[i])
            i = i + 1
            continue
        opeflag = opeflag + 1
        if True:
            if get_first_token(mch[i+1]) == "skip":
                raw_input("NotImplemented. PressAnyKey")
            elif get_first_token(mch[i+1]) == "PRE":
                y.append(mch[i])
                p = mch[i+1] + ""
                p = p.replace("PRE","PRE (")
                y.append(p)
                i = i + 2
                while get_first_token(mch[i]) != "THEN":
                    y.append(mch[i])
                    i = i + 1
                p = " ) or ( %s )"%cond
                y.append(p)
                y.append(mch[i])
                i = i + 1
                subs_org = []
                flag = 1
                j = i# + 1 
                
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                y.append("IF not( %s )"%cond)
                y.append("THEN")
                subs_org = mch[i:j-1]
                y = y + subs_org
                y.append("ELSE")
                y = y + subs
                y.append("END")
                y.append(mch[j-1])
                #y.append(mch[j])
                
                
                i = j# + 1
                

                #y.append(cond + " &")
                #i = i + 2
            else:
                flag = 1
                j = i + 2
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                y.append(mch[i])

                #y.append("PRE")
                y.append("IF")
                y.append("not( %s )"%cond)
                y.append("THEN")
                k = i + 1
                while k < j - 1:
                    y.append(mch[k])
                    k = k + 1
                y.append("END")
                y.append("ELSE")
                y = y + subs
                y.append(mch[j-1])

                """
                if y[-1] == "END;":
                    y[-1] = "END"
                    y.append("END;")
                else:
                    y.append("END")
                """

                i = j
    if opeflag != 1:
        print "Error: %d operations named %s found!"%(opeflag,ope)
        exit()
    return y




    

# fname --- conf file name
# pname --- parameter name
# ptype --- parameter type
def read_config(fname, pname, ptype):
    f = open(fname,"r")
    res = []
    for x in f.readlines():
        if "#" in x: continue
        y = x.replace("\n","")
        y = y.split(" = ")
        if len(y) != 2: continue
        if y[0] == pname:
            if ptype == "bool":
                res = "none"
                if y[1].lower() == "true": res = True
                if y[1].lower() == "false": res = False
                break
            else:
                res = eval("%s(y[1])"%ptype)
            break
    f.close()
    if res == []:
        res = "Error: Cannot find parameter %s."%pname
    return res



# Count the number of states and transitions of a ppt mch.
def CountMchStateAndTrans(mchfile):
    with open(mchfile) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]
    bscope = generate_training_set_condition(mch)
    print bscope
    countx = "./../ProB/probcli %s -model_check -nodead -timeout 100000 -scope \"%s\" -c"%(mchfile,bscope)
    #os.system(countx)
    reslog = subprocess.check_output(countx, shell=True)
    print "LOG:",reslog
    if "Timeout" in reslog:
        return -1, -1, -1
    p = re.compile("States analysed: (.*)\n")
    st = p.search(reslog).group(1)
    st = int(st)
    p = re.compile("Transitions fired: (.*)\n")
    tr = p.search(reslog).group(1)
    tr = int(tr)

    # Count deadlocks
    p = re.compile("deadlocked:(.*)\n")
    p = p.search(reslog)
    if p == None:
        dl = 0
    else:
        p = p.group(1)
        cx = p.index(',')
        p = p[0:cx]
        dl = int(p)

    # Count invariant violations & assertion violations
    p = re.compile("ignored:(.*)\n")
    p = p.search(reslog)
    if p == None:
        iav = 0
    else:
        p = p.group(1)
        cx = p.index(',')
        p = p[0:cx]
        iav = int(p)

    fs = dl + iav

    print "Number of Correct States:", st
    print "Number of Fired Transitions:", tr
    #print "Number of Deadlocks:", dl
    #print "Number of Invariant & Assertion Violations:", iav
    print "Number of Faulty States:", fs
    return st,tr,fs

"""
# Collect results from log files.
def CollectLog(fdir, logfile):
    fnames = os.listdir(fdir)
    fnames.sort()
    nep = 0
    num_iso = 0
    num_rev = 0
    while True:
        if not("%d"%(nep+1) in fnames):
            break
        fp = fdir + "%d/"%nep + logfile
        f = open(fp,"r")
        x = f.readline()
        if "Use Isolation." in x:
            num_iso += 1
        if "Use Revision." in x:
            num_rev += 1
        f.close()
        nep = nep + 1

    print "NUM ISO: %d."%num_iso
    print "NUM REV: %d."%num_rev
    print "Totally %d epochs."%nep
    return num_iso, num_rev
"""

# Read a pretty-printed mch file.
def read_mch_file(mchfile):
    with open(mchfile) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]
    res = []
    for x in mch:
        if "/*" in x:
            i = 0
            while x[i:i+2] != "/*":
                i = i + 1
            j = i + 1
            while x[j:j+2] != "*/":
                j = j + 1
            y = x[0:i] + x[j+2:len(x)]
        else:
            y = x
        res.append(y)
    return res




# Gnenrate a IF-THEN substitution for state revision.
# fstate --- A faulty state.
# cstate --- A correct state.
def gen_if_then_subs(fstate,cstate):
    subs = gen_subs(fstate,cstate)
    res = "IF %s THEN %s END"%(fstate,subs)
    return res


# Generate substitution from State X to State Y.
# x --- State X
# y --- State Y
def gen_subs(x,y):
    p = x.split("&")
    q = y.split("&")
    res = ""
    for i in xrange(len(q)):
        if i >= len(p) or p[i].replace(" ","") != q[i].replace(" ",""):
            sub = q[i].replace("=",":=")
            res = res + sub + " ; "
    if res == "":
        print "WARNING: No substitution procuded. Return \"skip\"."
        return "skip"
    else:
        res = res[0:len(res)-3]
        return res

#Bmch.generate_revision_set(mch, fstate, max_cost, max_operations, max_num_rev, rev_opt, mchfile)


# Generate a revision set.
# mch --- A B-machine. fstate --- A faulty state.
# max_cost --- The max cost of revision. 
# max_operations --- The max number of enabling transitions each operation computed. 
# max_num_rev --- The max number of revisions.
# rev_opt --- Revision option.
# file_annotation --- The annotation of temp files.
def generate_revision_set(mch, fstate, max_cost, max_operations, max_num_rev, rev_opt, file_annotation):
    revsetfilename = file_annotation + ".revset"
    revsetfile = open(revsetfilename,"w")
    sm = generate_revision_set_machine(mch, fstate, max_cost, rev_opt)
    for item in sm:
        revsetfile.write("%s\n"%item)
    revsetfile.close()

    bth = max_num_rev #256#65536
    print "Info: The maximum search breadth (including duplicate search when free variables exist) of the SMT solver is %d."%bth
    #mkgraph = "./../ProB/probcli -model_check -nodead -noinv -noass -p MAX_INITIALISATIONS %d -mc_mode bf -spdot %s.statespace.dot %s"%(bth,revsetfilename,revsetfilename)

    # Stable mode:
    #genmode = "-model_check -nodead -noinv -noass -disable-timeout -p MAX_INITIALISATIONS %d -mc_mode bf"%bth

    # Random mode, enable time-out:
    genmode = "-mc %d -nodead -noinv -noass -mc_mode random -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -p RANDOMISE_ENUMERATION_ORDER TRUE -p MAX_DISPLAY_SET -1"%(bth,bth,max_operations)

    mkgraph = "./../ProB/probcli " + genmode + " -spdot %s.statespace.dot %s "%(revsetfilename,revsetfilename)

    logtxt = os.popen(mkgraph).read()
    print logtxt
    if "state_error" in logtxt:
        clpfd_error = True
    else:
        clpfd_error = False

    #os.system(mkgraph)

    # If clpfd_error occurs, then disable CLPFD, and make the graph again.
    if clpfd_error == True:
        print "A CLPFD error occured. Disable CLPFD and re-make the state graph..."
        mkgraph = "./../ProB/probcli " + genmode + " -spdot %s.statespace.dot -p CLPFD FALSE %s "%(revsetfilename,revsetfilename)
        #mkgraph = "./../ProB/probcli -model_check -nodead -noinv -noass -disable-timeout -p MAX_INITIALISATIONS %d -mc_mode bf -spdot %s.statespace.dot -p CLPFD FALSE %s"%(bth,revsetfilename,revsetfilename)
        os.system(mkgraph)

    revset,numrev = extract_state_revision_from_file("%s.statespace.dot"%revsetfilename, max_cost)
    return revset,numrev


# Get the grand type ( bool / number / set / symbol ) of a value.
def get_grand_type(v):
    x = v.replace(" ","")
    if x == "TRUE" or x == "FALSE": return "BOOL"
    try:
        t = int(x)
        return "NUMBER"
    except ValueError:
        t = -1
    if x[0] == "{": return "SET"
    return "SYMBOL"


# Generate state differences.
# fst --- A faulty state ; dt --- The type of differences.
# dt can be "absolute" or "euclidean"
def generate_state_difference(fst, dt):
    fs = sentence_to_label(fst)
    res = ""
    if dt == "absolute":
        for i in xrange(len(fs) / 2):
            u = i * 2
            vt = get_grand_type(fs[u+1])
            if vt == "BOOL" or vt == "SYMBOL":
                diff = "card ( { %s } - { %s } )"%(fs[u],fs[u+1])
            elif vt == "SET":
                diff = "( card ( %s - %s ) + card ( %s - %s ) )"%(fs[u],fs[u+1],fs[u+1],fs[u])
            else: # The case of "NUMBER":
                diff = "max ( { %s - %s , %s - %s } )"%(fs[u],fs[u+1],fs[u+1],fs[u])
            res = res + diff + " + "
        res = res[0:len(res) - 3]
        return res
    else: 
        print "Error: The type of difference is invalid!"
        return None

# Generate restrictions of state differences.
# fst --- A faulty state ; dt --- The type of differences; md --- The max cost of diffrences.
# dt can be "absolute" or "euclidean"
def gen_sdr(fst, dt, md):
    sd = generate_state_difference(fst, dt)
    res = ""
    for i in xrange(md):
        y = sd + " = %d"%(i+1)
        res = res + y + " or "
    res = " ( " + res[0:len(res)-4] + " ) "
    return res

# Generate a list of operations for state selection.
# fst --- A faulty state ; dt --- The type of differences; md --- The max cost of diffrences.
# dt can be "absolute" or "euclidean"
def gen_ope_ss(fst, dt, md):
    res = []
    res.append("OPERATIONS")
    diff = generate_state_difference(fst, dt)
    for cost in xrange(md + 1):
        x = []
        x.append("cost_is_%d = "%cost)
        x.append("PRE")
        x.append("%s = %d"%(diff,cost))
        x.append("THEN")
        x.append("skip")
        if cost == md:
            x.append("END")
        else:
            x.append("END;")
        res = res + x
    return res

# Get a label and remove all line breaks:
def get_label_pretty(x):
    res = x.get_label()
    res = res.replace("\n","")
    res = res.replace("--\\>","-->")
    res = res.replace("\{","{")
    res = res.replace("\}","}")
    return res

# Print a B-machine to a file.
def print_mch_to_file(mch,filename):
    fp = open(filename,"w")
    for item in mch:
        fp.write("%s\n"%item)
    fp.close()
    return 0


# Generate a time annotation.
def time_annotation():
    x = time.time()
    x = int(x * 1000)
    x = "_%s"%str(x)
    return x

# Add a new operation to a B-machine:
# mch --- A B-machine. ope --- An operation.
def add_new_operation(mch,ope):
    mchlen = len(mch)
    i = 0
    while i < mchlen and get_first_token(mch[i]) != "OPERATIONS":
        i = i + 1
    res = mch[0:i+1] + ope + mch[i+1:mchlen]

    return res

# Create a new operation.
# precond -- A pre-condition. opename --- An operation name. revstate --- A state revision.
def create_new_operation(precond,opename,revstate):
    res = []
    res.append("%s = "%opename)
    res.append("PRE")
    res.append("%s"%precond)
    res.append("THEN")
    subst = state_to_substitution(revstate)
    res.append(subst)
    res.append("END;")
    return res

# Convert a state to substitution.
def state_to_substitution(st):
    res = st.replace("=",":=")
    res = res.replace("&",";")
    return res

# Extract revisions of a state.
# fname --- The filename of a state graph in the Dot format.
# mcost --- The max cost.
def extract_state_revision_from_file(fname,mcost):
    print "Extracting revisions of states from %s."%fname
    pp = pydotplus.graphviz.graph_from_dot_file(fname) 
    nlist = pp.get_node_list()
    elist = pp.get_edge_list()
    res = []
    lres = 0
    for cost in xrange(mcost + 1):
        edge_name = "\"cost_is_%d\""%cost
        res_sub = []
        for i in xrange(len(elist)):
            #if elist[i].get_label() == "\"INITIALISATION\"" or elist[i].get_label() == "\"INITIALIZATION\"":
            if elist[i].get_label() == edge_name:
                uname = elist[i].get_destination()
                for j in xrange(len(nlist)):
                    if nlist[j].get_name() == uname:
                        slabel = nlist[j].get_label()
                        break
                y = proc_state_label(slabel)
                rstate = label_to_sentence(y)
                res_sub.append(rstate)
        res.append(res_sub)
        lres = lres + len(res_sub)
        print "Cost = %d: %d state revisions are found."%(cost,len(res_sub))
    #print "Totally %d state revisions are found."%lres
    return res, lres

# A simple function for computing the difference between two states.
# st1 and st2 are two states.
def state_diff_simple(st1,st2):
    x = sentence_to_label(st1)
    y = sentence_to_label(st2)
    res = 0.0
    for i in xrange(len(x)):
        if x[i].replace(' ','') != y[i].replace(' ',''):
            res = res + 1.0
    return res

# Get variable names of a pretty-printed B-Machine.
def get_var_names(mch):
    mchlen = len(mch)
    i = 0
    while i < mchlen and not("VARIABLES" in mch[i]):
        i = i + 1
    if i == mchlen:
        print "Warning: No VARIABLES or ABSTRACT_VARIABLES token found in this B-machine."
        return []
    j = i + 1
    while j < mchlen and not("INVARIANT" in mch[j]) and not("PROPERTIES" in mch[j]):
        j = j + 1
    if j == mchlen:
        print "Warning: No INVARIANT token found in this B-machine."
        return []

    res = []
    k = i + 1
    while k < j:
        res.append(get_first_token(mch[k]))
        k = k + 1

    return res



def omitted_symbol_list():
    res = [
        ["+->","omittedpartialfunction"],
        ["-->","omittedtotalfunction"],
        ["+->>","omittedpartialsurjection"],
        ["-->>","omittedtotalsurjection"],
        [">+>","omittedpartialinjection"],
        [">->","omittedtotalinjection"],
        [">+>>","omittedpartialbijection"],
        [">->>","omittedtotalbijection"],
        ["..","omitteddomaintmpanno"],
        ["<->","omittedrelation"],
        ["|->","omittedmaplet"],
        ["|>","omittedrangerestriction"],
        ["|>>","omittedrangesubtraction"],
        ["<+","omittedrelationaloverriding"],
        ["><","omitteddirectproduct"],
        ["||","omittedparallelproduct"],
        ["/|\\","omittedtakefirstnelements"],
        ["\|/","omitteddropfirstnelements"]
    ]
    return res

def replace_omitted_symbols(x):
    osl = omitted_symbol_list()
    y = x + ""
    for item in osl:
        p = item[0]
        q = " " + item[1] + " "
        y = y.replace(p,q)
    return y

def recover_omitted_symbols(x):
    osl = omitted_symbol_list()
    y = x + ""
    for item in osl:
        p = " " + item[1] + " "
        q = item[0]
        y = y.replace(p,q)
    return y

# Split a line of B-code into tokens.
def b_code_split(x):

    y = '' + x
    y = replace_omitted_symbols(y)

    y = y.replace('(',' ( ')
    y = y.replace(')',' ) ')
    y = y.replace(',',' , ')
    y = y.replace('={',' = { ')
    y = y.replace('}=',' } = ')
    y = y.replace('{',' { ')
    y = y.replace('}',' } ')
    y = y.replace(';',' ; ')
    y = y.replace('-',' - ')
    y = y.replace('|',' | ')
    y = y.replace('~',' ~ ')

    # Replace "#x." with " # x . ", but omit "..". It is also applied to "!x.".
    y = y.replace('#',' # ')
    y = y.replace('!',' ! ')
    #y = y.replace('..',' domaintmpanno1007 ')
    y = y.replace('.',' . ')
    #y = y.replace(' domaintmpanno1007 ','..')

    y = recover_omitted_symbols(y)
    y = y.split()
    return y


   
# Replace all occurance of Token tp with Token tq. "blk" is a block of code.
def replace_token(blk, tp, tq):
    res = []
    for item in blk:
        x = b_code_split(item)
        for i in xrange(len(x)):
            if x[i] == tp:
                x[i] = ' ' + tq
            else:
                x[i] = ' ' + x[i]
        y = ''.join(x)
        res.append(y)
    return res

# Make a list of "init" variables.
def make_init_var_list(vlist):
    res = []
    for x in vlist:
        y = x + '_init'
        res.append(y)
    return res


# Get all operation names of a B-machine.
# mch --- A pretty-printed B-machine.
def get_all_ope_names(mch):
    ope_decls = get_all_ope_decl(mch)
    res = []
    for x in ope_decls:
        #y = get_first_token(x)
        #res.append(y)
        y = proc_opes_decl(x)
        res.append(y[0])
    return res
    


# Replace all occurance of a list of variables with their "init" variables.
# blk --- A block of code. vlist --- A list of variables.
def replace_var_with_init(blk,vlist):
    rlist = make_init_var_list(vlist)
    res = blk[:]
    for i in xrange(len(vlist)):
        res = replace_token(res,vlist[i],rlist[i])
    return res

# Get all pre-conditions from a B-Machine.
# mch --- A pretty-printed B-Machine.
def get_all_precond(mch):
    res = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        i = i + 1
    mchlen = len(mch)
    while i < mchlen:
        if get_first_token(mch[i]) == "PRE":
            j = i + 1
            y = ''
            while get_first_token(mch[j]) != "THEN":
                y = y + ' ' + mch[j]
                j = j + 1
            res.append(y)
            i = j + 1
        else:
            i = i + 1
    if res == []:
        print "Warning: No pre-condition found! Return None."
        return None
    return res

# Generation a B-disjunction of predicates.
# plist --- A list of predicates.
def gen_b_disj(plist):
    if plist == None:
        return None
    if len(plist) == 1:
        return plist[0]
    res = ""
    for p in plist:
        res = res + "( " + p + " ) or "
    res = res[0:len(res)-3]
    return res


# Get invariants of a B-Machine.
# mch --- A pretty-printed B-Machine.
def get_invariants(mch):
    res = []
    i = 0
    while get_first_token(mch[i]) != "INVARIANT":
        i = i + 1
    mchlen = len(mch)
    j = i + 1
    while j < mchlen:
        tt = get_first_token(mch[j])
        # Based on the syntax of <The B-book>, p.273.
        if tt == "ASSERTIONS": break
        if tt == "DEFINITIONS": break
        if tt == "INITIALIZATION": break
        if tt == "INITIALISATION": break
        if tt == "OPERATIONS": break
        if tt == "END": break
        res.append(mch[j])
        j = j + 1
    if res == []:
        print "Warning: No invariant found! Return [\'TRUE\']."
        return ['TRUE']
    return res


# Get enumerated sets of a B-Machine.
# mch --- A pretty-printed B-Machine.
def get_enum_sets(mch):
    res = []
    i = 0
    mchlen = len(mch)
    while i < mchlen:
        if get_first_token(mch[i]) == "SETS": break
        i = i + 1
    j = i + 1
    while j < mchlen:
        tt = get_first_token(mch[j])
        # Based on the syntax of <The B-book>, p.273.
        if "CONSTANTS" in tt: break
        if "VARIABLES" in tt: break
        if tt == "PROPERTIES": break
        if tt == "INVARIANT": break
        if tt == "ASSERTIONS": break
        if tt == "DEFINITIONS": break
        if tt == "INITIALIZATION": break
        if tt == "INITIALISATION": break
        if tt == "OPERATIONS": break
        if tt == "END": break
        res.append(mch[j])
        j = j + 1
    if res == []:
        return None
    return res

# Convert the code of enum sets to types.
def convert_enum_sets_to_types(esets):
    res = []
    for x in esets:
        y = x.replace(";","").replace(" ","")
        y = y.split("=")
        sname = y[0]
        svalue = y[1].replace("{","").replace("}","")
        svalue = svalue.split(",")
        res.append([sname,svalue])
    return res

# Get assertions of a B-Machine.
# mch --- A pretty-printed B-Machine.
def get_assertions(mch):
    res = []
    i = 0
    mchlen = len(mch)
    while i < mchlen:
        if get_first_token(mch[i]) == "ASSERTIONS": break
        i = i + 1
    j = i + 1
    while j < mchlen:
        tt = get_first_token(mch[j])
        # Based on the syntax of <The B-book>, p.273.
        if tt == "DEFINITIONS": break
        if tt == "INITIALIZATION": break
        if tt == "INITIALISATION": break
        if tt == "OPERATIONS": break
        if tt == "END": break
        y = mch[j]
        if y[-1] == ";":
            y = " ( " + y[0:len(y)-1] + " ) & "
        res.append(y)
        j = j + 1
    if res == []:
        print "Warning: No assertion found! Return \'None\'."
        return None
    res[-1] = "( " + res[-1] + ")"
    return res




# Convert a list of variables to a sequence.
def var_list_to_seq(vlist):
    res = ""
    for v in vlist:
        res = res + v + " , "
    res = res[0:len(res)-2]
    return res

# Generate a revision condition (in ANY-WHERE-THEN-END format).
# mch --- A pretty-printed B-machine.
# ex_list --- A list of extra conditions.
# rev_opt --- Revision Option.
def generate_revision_condition(mch, ex_list, rev_opt):
    var_list = get_var_names(mch)
    ope_var_list = get_all_ope_var(mch)[0]
    inv_list = get_invariants(mch)
    ass_list = get_assertions(mch)
    precon_list = get_all_precond(mch)
    precon_disj = gen_b_disj(precon_list)

    vseq = var_list_to_seq(var_list)
    rlist = make_init_var_list(var_list)
    rseq = var_list_to_seq(rlist)

    any_seq = rseq

    if ope_var_list != []:
        ope_var_list = list(set(ope_var_list)) # Remove duplicate variables.
        ope_vseq = var_list_to_seq(ope_var_list)
        any_seq = any_seq + " , " + ope_vseq

    rev_list = ["ANY %s WHERE"%any_seq]
    rev_list = rev_list + inv_list
    if not("Ass" in rev_opt) and not(ass_list == None):
        rev_list.append(" & ")
        rev_list = rev_list + ass_list
    if not(ex_list == []):
        rev_list.append(" & ")
        rev_list = rev_list + ex_list
    if not("Dead" in rev_opt) and precon_disj != None:
        rev_list.append(" & ( " + precon_disj + " )")
    rev_list = replace_var_with_init(rev_list,var_list)
    rev_list.append("THEN")
    rev_list.append("%s := %s"%(vseq,rseq))
    rev_list.append("END")

    return rev_list



# Generate an operation for detecting good states (i.e., states satisfying all invariants and assertions).
# mch --- A pretty-printed B-machine.
def generate_good_state_detection_operation(mch):
    var_list = get_var_names(mch)
    ope_var_list = get_all_ope_var(mch)[0]
    inv_list = get_invariants(mch)
    ass_list = get_assertions(mch)
    #precon_list = get_all_precond(mch)
    #precon_disj = gen_b_disj(precon_list)

    rev_list = ["PRE"]
    rev_list = rev_list + inv_list
    if not(ass_list == None):
        rev_list.append(" & ")
        rev_list = rev_list + ass_list
    #if precon_disj != None:
    #    rev_list.append(" & ( " + precon_disj + " )")
 
    rev_list.append("THEN")
    rev_list.append("skip")
    rev_list.append("END")

    return rev_list





# Generate a revision set machine.
# mch --- A pretty-printed B-machine.
# fst --- A faulty state.
# mcost --- The max cost.
# rev_opt --- Revision Option.
def generate_revision_set_machine(mch, fst, mcost, rev_opt):
    diff_cond = gen_sdr(fst,"absolute",mcost)
    rev_cond = generate_revision_condition(mch, [diff_cond], rev_opt)
    diff_ope = gen_ope_ss(fst, "absolute", mcost)
    
    res = []
    i = 0
    mchlen = len(mch)
    while i < mchlen:
        tt = get_first_token(mch[i])
        # Based on the syntax of <The B-book>, p.273.
        if tt == "INITIALIZATION": break
        if tt == "INITIALISATION": break
        res.append(mch[i])
        i = i + 1
    res.append("INITIALISATION")
    res = res + rev_cond
    res = res + diff_ope
    res.append("END")
    return res

 
# Replace the initialisation of a B-machine.
# mch --- A pretty-printed B-machine.
# init --- Initialisation.
def replace_initialisation(mch, init):

    res = []
    i = 0
    mchlen = len(mch)
    while i < mchlen:
        tt = get_first_token(mch[i])
        # Based on the syntax of <The B-book>, p.273.
        if tt == "INITIALIZATION": break
        if tt == "INITIALISATION": break
        res.append(mch[i])
        i = i + 1
    res.append("INITIALISATION")
    res = res + init
    while get_first_token(mch[i]) != "OPERATIONS":
        i = i + 1
    while i < mchlen:
        res.append(mch[i])
        i = i + 1
    return res

    
   
def generate_training_set_condition(mch):
    inv_list = get_invariants(mch)
    ass_list = get_assertions(mch)
    
    #precon_list = get_all_precond(mch)
    #precon_disj = gen_b_disj(precon_list)
    res = ""
    for x in inv_list:
        res = res + " " + x
    if ass_list == None: return res
    res = res + " & "
    for x in ass_list:
        res = res + " " + x
    
    return res



# Get all operations.
# mch --- A pretty-printed B-Machine.
def get_all_opes(mch):
    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        i = i + 1
    mchlen = len(mch)
    i = i + 1
    while i < mchlen:
        if mch[i].split() == []:
            i = i + 1
            continue
        if get_first_token(mch[i]) == "END": break
        y.append(mch[i])
        i = i + 1
        while mch[i].split() == []:
           i = i + 1
        y.append(mch[i])
        i = i + 1
        flag = 1
        while flag != 0:
            jt = get_first_token(mch[i])
            if jt == "END":
                flag = flag - 1
            elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "VAR" or jt == "LET":
                flag = flag + 1
            y.append(mch[i])
            i = i + 1 
    return y






# Generate a training set machine for DNNs.
# mch --- A pretty-printed B-machine.
# rev_opt --- Revision Option.
def generate_training_set_machine(mch, rev_opt):
    rev_cond = generate_revision_condition(mch, [], rev_opt)
    all_opes = get_all_opes(mch)
    
    res = []
    i = 0
    mchlen = len(mch)
    while i < mchlen:
        tt = get_first_token(mch[i])
        # Based on the syntax of <The B-book>, p.273.
        if tt == "INITIALIZATION": break
        if tt == "INITIALISATION": break
        res.append(mch[i])
        i = i + 1
    res.append("INITIALISATION")
    res = res + rev_cond
    res.append("OPERATIONS")
    res = res + all_opes
    res.append("END")
    return res

    






# Add a pre-condition condition to an operation.
# mch --- A pretty-printed B-Machine. ope --- The name of an operation which will be changed. cond --- A pre-condition.
def add_precond_to_mch(mch,ope,cond):
    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        y.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    opeflag = 0
    while i < mchlen:
        if not(ope in mch[i]) or not(proc_opes_decl(mch[i])[0] == ope):
            y.append(mch[i])
            i = i + 1
            continue
        opeflag = opeflag + 1
        if True:
            if get_first_token(mch[i+1]) == "skip":
                y.append(mch[i])
                y.append("PRE")
                y.append(cond)
                y.append("THEN")
                y.append("skip")
                if ";" in mch[i+1]:
                    y.append("END;")
                else:
                    y.append("END")
                i = i + 2

            elif get_first_token(mch[i+1]) == "PRE":
                y.append(mch[i])
                y.append(mch[i+1])
                y.append(cond + " &")
                i = i + 2
            else:
                flag = 1
                j = i + 2
                while flag > 0:
                    jt = get_first_token(mch[j])
                    if jt == "END":
                        flag = flag - 1
                    # Based on the syntax of <The B book.> pp 266
                    elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                        flag = flag + 1
                    j = j + 1
                y.append(mch[i])
                y.append("PRE")
                y.append(cond)
                y.append("THEN")
                k = i + 1
                while k < j:
                    y.append(mch[k])
                    k = k + 1
                if y[-1] == "END;":
                    y[-1] = "END"
                    y.append("END;")
                else:
                    y.append("END")
                i = j
    if opeflag != 1:
        print "Error: %d operations named %s found!"%(opeflag,ope)
        exit()
    return y



# Add a IF-THEN substitution to an operation.
# mch --- A pretty-printed B-Machine. ope --- The name of an operation which will be changed. subs --- A IF-THEN substitution.
def add_if_then_subs_to_mch(mch,ope,subs):
    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        y.append(mch[i])
        i = i + 1
    mchlen = len(mch)
    opeflag = 0
    while i < mchlen:
        if not(ope in mch[i]) or not(proc_opes_decl(mch[i])[0] == ope):
            y.append(mch[i])
            i = i + 1
            continue
        opeflag = opeflag + 1
        if True: #get_first_token(mch[i]) == ope:
            y.append(mch[i])
            j = i + 1
            jt = get_first_token(mch[j])
            while not(jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET"):
                y.append(mch[j])
                j = j + 1
                jt = get_first_token(mch[j])
            y.append(mch[j])
            j = j + 1
            flag = 1
     
            while flag > 0:
                jt = get_first_token(mch[j])
                if jt == "END":
                    flag = flag - 1
                # Based on the syntax of <The B book.> pp 266
                elif jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "SELECT" or jt == "VAR" or jt == "LET":
                    flag = flag + 1
                y.append(mch[j])
                j = j + 1
            k = len(y) - 1
            while get_first_token(y[k]) != "END":
                k = k - 1
            y = y[0:k] + [" ; ",subs] + y[k:len(y)]

            i = j
        #else:
        #    y.append(mch[i])
        #    i = i + 1
    if opeflag != 1:
        print "Error: %d operations named %s found!"%(opeflag,ope)
        exit()
    return y



# Get the first token of a sentence:
def get_first_token(x):
    y = '' + x
    y = y.replace('(',' ')
    y = y.replace(',',' ')
    y = y.replace('=',' ')
    y = y.replace(';',' ')
    y = y.replace('--\\>',' --\\> ')
    y = y.replace('-->', ' --> ')
    y = y.split()
    if len(y) > 0:
        return y[0]
    else:
        return None


# Get all operation declarations.
# mch --- A pretty-printed B-Machine.
def get_all_ope_decl(mch):
    y = []
    i = 0
    while get_first_token(mch[i]) != "OPERATIONS":
        i = i + 1
    mchlen = len(mch)
    i = i + 1
    while i < mchlen:
        if mch[i].split() == []:
            i = i + 1
            continue
        if get_first_token(mch[i]) == "END": break
        y.append(mch[i])
        i = i + 1
        while mch[i].split() == []:
           i = i + 1
        i = i + 1
        flag = 1
        while flag != 0:
            jt = get_first_token(mch[i])
            if jt == "END":
                flag = flag - 1
            elif jt == "SELECT" or jt == "BEGIN" or jt == "PRE" or jt == "IF" or jt == "CHOICE" or jt == "ANY" or jt == "VAR" or jt == "LET":
                flag = flag + 1
            i = i + 1 
    return y


# Get all operation variables:
# mch --- A pretty-printed B-Machine.
def get_all_ope_var(mch):
    x = get_all_ope_decl(mch)
    #y = []
    invar = []
    outvar = []
    for item in x:
        print item
        _,p,q = proc_trans_decl(item)
        invar = invar + p
        outvar = outvar + q
        #y = y + p
        #y = y + q
    return [invar,outvar]

# Process the declaration of an operation.
# In general, the declaration x is in one of the following formats:
# "ope ="
# "ope(in_1,in_2,...,in_n) ="
# "out_1,out_2,...,out_m <-- ope ="
# "out_1,out_2,...,out_m <-- ope(in_1,in_2,...,in_n) ="
def proc_trans_decl(x):
    # For historical reasons, "proc_opes_decl" is previous "proc_trans_decl".
    return proc_opes_decl(x)
def proc_opes_decl(x):
    lenx = len(x)
    invarlist = []
    outvarlist = []
    for i in xrange(lenx):
        if x[i:i+3] == "<--":
            y = x[0:i-1]
            y = y.replace(',',' ')
            outvarlist = y.split() 
            x = x[i+3:lenx]
            lenx = len(x)
            break
    x = x.replace('=',' ')
    x = x.replace(',',' ')
    x = x.replace('(',' ')
    x = x.replace(')',' ')
    x = x.split()
    opename = x[0]
    invarlist = x[1:len(x)]
    return opename, invarlist, outvarlist

# Convert a set of state labels to a B-sentence.
def label_to_sentence(x):
    if len(x) == 0:
        return "TRUE"
    else:
        res = ""
        for i in xrange(len(x)):
            if i % 2 == 0:
                res = res + x[i]
            else:
                res = res + " = %s & "%x[i]
        res = res[0:len(res)-3]
        return res


# Convert a B-sentence to a set of labels.
# It is an anti-function of "label_to_sentence(x)".
def sentence_to_label(x):
    if x == "TRUE":
        return []
    else:
        res = []
        y = x.split(' & ')
        for item in y:
            vs = item.split(' = ')
            res = res + vs
        return res

# Process state labels (a dot label --> a set of state labels):
def proc_state_label(x):
    if x != None:
        y = "" + x
        #y = y.replace('\|-\>',' , ')
        y = y.replace('\\n',' ')
        y = y.replace('\\{',' { ')
        y = y.replace('\\}',' } ')
        y = y.replace('{',' { ')
        y = y.replace('}',' } ')
        y = y.replace('"','')
        y = y.replace("'","")
        y = y.replace('=','')
        y = y.replace(',',' ')
        y = y.replace('\|-\>',',')
        y = y.replace('\\',' ')
    else:
        y = 'None'

    y = y.split()
    y = merge_set_label(y)

    return y

# Merge set labels:
def merge_set_label(y):
    i = 0
    maxlen = len(y)
    res = []
    while i < maxlen:
        if y[i] == '{':
            flag = 1
            j = i + 1
            sval = '{ '
            while flag > 0:
                if y[j] == '{':
                    flag = flag + 1
                    sval = sval + '{ '
                elif y[j] == '}':
                    flag = flag - 1
                    sval = sval + '}, '
                else:
                    sval = sval + y[j] + ' , '
                j = j + 1
            sval = sval.replace(', }',' }')
            sval = sval[0:len(sval)-2] 
            res.append(''.join(sval))
            
            i = j
        else:
            res.append(y[i])
            i = i + 1
    return res



# TF --- List of faulty transitions
# SREV --- List of candidate revision states
# Output --- List of revision repairs
def RevisionSynthesis(TF,SREV):
    res = []
    for x in TF:
        for y in SREV:
            s = ["revision"] + x + [y]
            res.append(s)
    return res



# TF --- List of deadlock transitions
# FL --- List of operations
# SINS --- List of candidate insertion states
# Output --- List of insertion repairs
def InsertionSynthesisForDeadlocks(TF,FL,SINS):
    res = []
    for x in TF:
        for y in SINS:
            i = int(random.random() * len(FL))
            s = ["insertion"] + x + [[x[2],FL[i],y]]
            res.append(s)
    return res

# TL --- List of existing transitions
# FL --- List of operations
# SG --- List of candidate goal states that satisfy a GOAL predicate
# Output --- List of insertion repairs
def InsertionSynthesisForGoalPredicates(TL,FL,SG):
    res = []

    prev_trans = ["GOAL","GOAL","GOAL"]

    # randomly generate an insertion from an existing pre-state to a goal state
    for x in TL:
        j = int(random.random() * len(SG))
        y = SG[j]

        i = int(random.random() * len(FL))
        s = ["insertion"] + prev_trans + [[x[0],FL[i],y]]
        res.append(s)

    # randomly generate an insertion from an existing post-state to a goal state
    for x in TL:
        j = int(random.random() * len(SG))
        y = SG[j]

        i = int(random.random() * len(FL))
        s = ["insertion"] + prev_trans + [[x[2],FL[i],y]]
        res.append(s)
    return res




# TF --- List of faulty transitions
# W --- Semantic model
# M --- Abstract machine file
# conffile --- configuration file
# wdir --- working directory
# Output --- List of revision repairs
def GeneticRevisionSynthesis(TF,W,M,conffile,wdir):
    cmd = "mkdir %s"%wdir
    os.system(cmd)

    SType = W.SType
    VList = W.VList
    res = []

    NC = read_config(conffile,"num_genetic_candidates","int")
    NG = read_config(conffile,"num_genetic_generations","int")
    NB = read_config(conffile,"num_genetic_breadth","int")
    NR = read_config(conffile,"num_genetic_result","int")

    res = []
    for x in TF:
        SREV = []
        for i in xrange(NC):
            s = GetRandomState(SType)
            SREV.append(s)

        #for s in SREV: print s
        #raw_input("sdfsa")

        """
        R = []
        for y in SREV:
            s = ["revision"] + x + [y]
            R.append(s)
        P = SemLearnLib.ScoreRevisionsUsingSemanticsModel(W,R,wdir)
        for i in xrange(len(R)):
            R[i].append(P[i])
        R.sort(key=lambda x:x[5],reverse=True)
        """
        R = RankModifications(x,SREV,W,conffile,wdir)
        R = R[0:min(len(R),NC)]
        #for y in R: print y
        print "org",R[0][5]
        #raw_input("afsad")

        S = []
        for y in R:
            S.append(y[4])

        U = S + []

        for i in xrange(NG):
            print "Epoch",i
            S1 = []
            for y in S:
                for j in xrange(NB):
                    yt = StateMutation(y,SType)
                    S1.append(yt)
                #print y
                #print yt

            # S2 --- results of crossover
            S2 = []
            for y in S:
                for j in xrange(NB):
                    z = S[int(random.random() * len(S))]
                    yt = StateCrossover(y,z)
                    S2.append(yt)
                

            S = S + S1 + S2
            S = list_union(S,[])
            
            RS = RankModifications(x,S,W,conffile,wdir)
            #for y in RS: print y
            #print len(RS)
            print RS[0][5]
            #raw_input("asdfas")
            S = []
            for j in xrange(min(NC,len(RS))):
                S.append(RS[j][4])
            U = U + S

 
        U = list_union(U,[])
        UR = RankModifications(x,U,W,conffile,wdir)
        U = []
        for j in xrange(min(len(UR),NR)):
            U.append(UR[j][4])

        for y in U:
            s = ["revision"] + x + [y]
            res.append(s)


    # Get all candidate states that satisfy revision requirements.
    SL = []
    for x in res:
        SL.append(x[4])
    SL = list_union(SL,[])
    print len(SL)
    #raw_input("num of states")


   
    init1u = RepSimpLib.initialise_vble_by_examples(VList,SL)
    with open(M) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]

    mch1u = replace_initialisation(mch,init1u)

    i = 0
    mcht = []
    while get_first_token(mch1u[i]) != "OPERATIONS":
        mcht.append(mch1u[i])
        i = i + 1
    mcht.append(mch1u[i])
    
    cond = generate_training_set_condition(mch)
    op = "verif = PRE %s THEN skip END"%cond
    mcht.append(op)
    mcht.append("END")

    fn = "%s/MT.mch"%wdir
    f = open(fn,"w")
    for x in mcht:
        f.write(x)
        f.write("\n")
    f.close()
    MT = fn

    for x in mcht: print x
    #raw_input("MCHT")

    DT = wdir + "/DT.txt"
    max_initialisations = len(SL) * 100
    max_operations = len(SL) * 100
    bscope = cond
    oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MT,max_initialisations,max_operations,bscope,DT)
    os.system(oscmd)

    sg = Bgenlib.BStateGraphForNN()
    sg.ReadStateGraph(DT)
    TL = sg.GetTransList()
    
    # VS --- set of states that satisfy requirements.
    VS = list_difference(RepSimpLib.extract_all_states(TLT),sg.GetStatesWithoutOutgoingTransitions(TL))

    resT = []
    for x in res:
        if not(x[4][2] in VS):
            continue
        resT.append(x)

    return resT



# TF --- List of faulty transitions
# W --- Semantic model
# FL --- list of operations
# M --- Abstract machine file
# conffile --- configuration file
# wdir --- working directory
# Output --- List of insertion repairs
def GeneticInsertionSynthesisForDeadlocks(TF,FL,W,M,conffile,wdir):
    cmd = "mkdir %s"%wdir
    os.system(cmd)

    SType = W.SType
    VList = W.VList
    res = []

    NC = read_config(conffile,"num_genetic_candidates","int")
    NG = read_config(conffile,"num_genetic_generations","int")
    NB = read_config(conffile,"num_genetic_breadth","int")
    NR = read_config(conffile,"num_genetic_result","int")

    res = []
    for x in TF:
        SINS = []
        for i in xrange(NC):
            s = GetRandomState(SType)
            op = FL[int(random.random() * len(FL))]
            SINS.append([x[2],op,s])

        R = RankInsertions(x,SINS,W,conffile,wdir)
        R = R[0:min(len(R),NC)]
        #for y in R: print y
        print "org",R[0][5]

        S = []
        for y in R:
            S.append([y[4][1]] + y[4][2])

        U = S + []

        for i in xrange(NG):
            print "Epoch",i
            S1 = []
            for y in S:
                for j in xrange(NB):
                    yt = StateMutation(y,[["Dist"] + FL] + SType)
                    S1.append(yt)
                #print y
                #print yt

            # S2 --- results of crossover
            S2 = []
            for y in S:
                for j in xrange(NB):
                    z = S[int(random.random() * len(S))]
                    yt = StateCrossover(y,z)
                    S2.append(yt)
                

            S = S + S1 + S2
            S = list_union(S,[])
            ST = []
            for y in S:
                ST.append([x[2],y[0],y[1:len(y)]])
            
            RS = RankInsertions(x,ST,W,conffile,wdir)
            #for y in RS: print y
            #print len(RS)
            print RS[0][5]
            #raw_input("asdfas")
            S = []
            for j in xrange(min(NC,len(RS))):
                S.append([RS[j][4][1]] + RS[j][4][2])
            U = U + S

 
        U = list_union(U,[])
        UT = []
        for y in U:
            UT.append([x[2],y[0],y[1:len(y)]])

        UR = RankInsertions(x,UT,W,conffile,wdir)
        U = []
        for j in xrange(min(len(UR),NR)):
            U.append([UR[j][4][1]] + UR[j][4][2])

        for y in U:
            s = ["insertion"] + x + [[x[2],y[0],y[1:len(y)]]]
            res.append(s)


    # Get all candidate states.
    SL = []
    for x in res:
        SL.append(x[4][2])
    SL = list_union(SL,[])
    print len(SL)
    #raw_input("num of states")


   
    init1u = RepSimpLib.initialise_vble_by_examples(VList,SL)
    with open(M) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]

    mch1u = replace_initialisation(mch,init1u)

    i = 0
    mcht = []
    while get_first_token(mch1u[i]) != "OPERATIONS":
        mcht.append(mch1u[i])
        i = i + 1
    mcht.append(mch1u[i])
    
    cond = generate_training_set_condition(mch)
    op = "verif = PRE %s THEN skip END"%cond
    mcht.append(op)
    mcht.append("END")

    fn = "%s/MT.mch"%wdir
    f = open(fn,"w")
    for x in mcht:
        f.write(x)
        f.write("\n")
    f.close()
    MT = fn

    for x in mcht: print x
    #raw_input("MCHT")

    DT = wdir + "/DT.txt"
    max_initialisations = len(SL) * 100
    max_operations = len(SL) * 100
    bscope = cond
    oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MT,max_initialisations,max_operations,bscope,DT)
    os.system(oscmd)

    sg = Bgenlib.BStateGraphForNN()
    sg.ReadStateGraph(DT)
    TL = sg.GetTransList()
    
    # VS --- set of states that satisfy requirements.
    VS = list_difference(RepSimpLib.extract_all_states(TLT),sg.GetStatesWithoutOutgoingTransitions(TL))

    resT = []
    for x in res:
        if not(x[4][2] in VS):
            continue
        resT.append(x)

    return resT





# TL --- List of existing transitions
# GP --- a GOAL predicate
# SG --- a set of candidate goal states, as the start pointing of search
# FL --- list of operations used by insertion repairs
# W --- Semantic model
# M --- Abstract machine file
# conffile --- configuration file
# wdir --- working directory
# Output --- List of insertion repairs
def GeneticInsertionSynthesisForGoalPredicates(TL,GP,SG,FL,W,M,conffile,wdir):
    cmd = "mkdir %s"%wdir
    os.system(cmd)

    SType = W.SType
    VList = W.VList
    res = []

    NC = read_config(conffile,"num_genetic_candidates","int")
    NG = read_config(conffile,"num_genetic_generations","int")
    NB = read_config(conffile,"num_genetic_breadth","int")
    NR = read_config(conffile,"num_genetic_result","int")

    res = []

    prev_trans = ["GOAL","GOAL","GOAL"]

    SINS = []
    for i in xrange(NC):
        x = TL[int(random.random() * len(TL))]
        s = SG[int(random.random() * len(SG))]
        op = FL[int(random.random() * len(FL))]
        if random.random() > 0.5:
            SINS.append([x[0],op,s])
        else:
            SINS.append([x[2],op,s])



    R = RankInsertions(prev_trans,SINS,W,conffile,wdir)
    R = R[0:min(len(R),NC)]
    #for y in R: print y
    print "org",R[0][5]

    S = []
    for y in R:
        S.append([y[4][1]] + y[4][2])

    U = S + []

    for i in xrange(NG):
        print "Epoch",i
        S1 = []
        for y in S:
            for j in xrange(NB):
                yt = StateMutation(y,[["Dist"] + FL] + SType)
                S1.append(yt)
            #print y
            #print yt

        # S2 --- results of crossover
        S2 = []
        for y in S:
            for j in xrange(NB):
                z = S[int(random.random() * len(S))]
                yt = StateCrossover(y,z)
                S2.append(yt)
                

        S = S + S1 + S2
        S = list_union(S,[])

        ST = []

        for z in SINS:
            y = S[int(random.random() * len(S))]
            for j in xrange(NB):
                ST.append([z[0],y[0],y[1:len(y)]])
            
        RS = RankInsertions(prev_trans,ST,W,conffile,wdir)
        S = []
        for j in xrange(min(NC,len(RS))):
            S.append([RS[j][4][1]] + RS[j][4][2])
        U = U + S

 
    U = list_union(U,[])
    UT = []

    for z in SINS:
        y = U[int(random.random() * len(U))]
        for j in xrange(NB):
            UT.append([z[0],y[0],y[1:len(y)]])

    UR = RankInsertions(prev_trans,UT,W,conffile,wdir)
    U = []
    for j in xrange(min(len(UR),NR)):
        U.append([UR[j][4][1]] + UR[j][4][2])

    for z in SINS:
        y = U[int(random.random() * len(U))]
        s = ["insertion"] + prev_trans + [[z[0],y[0],y[1:len(y)]]]
        res.append(s)


    # Get all candidate states.
    SL = []
    for x in res:
        SL.append(x[4][2])
    SL = list_union(SL,[])
    print len(SL)
    #raw_input("num of states")


   
    init1u = RepSimpLib.initialise_vble_by_examples(VList,SL)
    with open(M) as mchf:
        mch = mchf.readlines()
    mch = [x.strip() for x in mch]

    mch1u = replace_initialisation(mch,init1u)

    i = 0
    mcht = []
    while get_first_token(mch1u[i]) != "OPERATIONS":
        mcht.append(mch1u[i])
        i = i + 1
    mcht.append(mch1u[i])
    
    cond = generate_training_set_condition(mch) + " & ( %s )"%GP
    op = "verif = PRE %s THEN skip END"%cond
    mcht.append(op)
    mcht.append("END")

    fn = "%s/MT.mch"%wdir
    f = open(fn,"w")
    for x in mcht:
        f.write(x)
        f.write("\n")
    f.close()
    MT = fn

    for x in mcht: print x
    #raw_input("MCHT")

    DT = wdir + "/DT.txt"
    max_initialisations = len(SL) * 100
    max_operations = len(SL) * 100
    bscope = cond + " & ( %s )"%GP
    oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MT,max_initialisations,max_operations,bscope,DT)
    os.system(oscmd)

    sg = Bgenlib.BStateGraphForNN()
    sg.ReadStateGraph(DT)
    TLT = sg.GetTransList()
    
    # VS --- set of states that satisfy requirements.
    VS = list_difference(RepSimpLib.extract_all_states(TLT),sg.GetStatesWithoutOutgoingTransitions(TLT))

    resT = []
    for x in res:
        if not(x[4][2] in VS):
            continue
        resT.append(x)


    return resT




# T --- types of variables in states
def GetRandomState(T):
    s = []
    for x in T:
        if x[0] == "Set":
            v = []
            for j in xrange(1,len(x)):
                if random.random() > 0.5:
                    v.append(x[j])
            s.append(v)
        else:
            j = 1 + int(random.random() * (len(x) - 1))
            v = x[j]
            s.append(str(v))
    return s


# T --- faulty transition
# SREV --- List of candidate revision states
# W --- Semantic model
# conffile --- configuration file
# wdir --- working directory
# Output --- List of revision repairs
def RankModifications(T,SREV,W,conffile,wdir):
    x = T
    R = []
    for y in SREV:
        s = ["revision"] + x + [y]
        R.append(s)
    P = SemLearnLib.ScoreRevisionsUsingSemanticsModel(W,R,wdir)
    for i in xrange(len(R)):
        R[i].append(P[i])
    R.sort(key=lambda x:x[5],reverse=True)
    return R



# T --- deadlock transition
# SINS --- List of insertion repairs
# W --- Semantic model
# conffile --- configuration file
# wdir --- working directory
# Output --- List of revision repairs
def RankInsertions(T,SINS,W,conffile,wdir):
    x = T
    R = []
    for y in SINS:
        s = ["insertion"] + x + [y]
        R.append(s)
    P = SemLearnLib.ScoreInsertionsUsingSemanticsModel(W,R,wdir)
    for i in xrange(len(R)):
        R[i].append(P[i])
    R.sort(key=lambda x:x[5],reverse=True)
    return R


# X --- state
# T --- type of state
def StateMutation(X,T):
    i = int(len(X) * random.random())
    Y = X + []
    if T[i][0] == "Set":
        t = int((len(T[i])-1) * random.random())
        el = T[i][t+1]
        if el in Y[i]:
            #print "remove %s from %s"%(el,Y[i])
            Y[i] = Y[i] + []
            Y[i].remove(el)
        else:
            #print "add %s to %s"%(el,Y[i])
            Y[i] = Y[i] + [el]
            Y[i].sort()
    else:        
        t = int((len(T[i])-1) * random.random())
        Y[i] = str(T[i][t+1])
    return Y

# X --- state 1
# Y --- state 2
def StateCrossover(X,Y):
    Z = []
    for i in xrange(len(X)):
        if random.random() > 0.5:
            Z.append(X[i])
        else:
            Z.append(Y[i])
    return Z

# TF --- List of faulty transitions
# Output --- List of isolation repairs
def IsolationSynthesis(TF):
    res = []
    for x in TF:
        s = ["isolation"] + x
        res.append(s)
    return res



"""
# Compute MaxLabelLength.
maxlabellength = 0;
qq = pp.get_edge_list()
for i in xrange(len(qq)):
    x = qq[i].get_source()
    if x == 'root':
        edge_src = -1
    else:
        edge_src = int(x)
    edge_dest = int(qq[i].get_destination())
    y = qq[i].get_label()
    if y != None:
        y = y.replace('"','')
        y = y.replace("'","")
    else:
        y = 'None'
    if len(y) > maxlabellength:
        maxlabellength = len(y)

cfile.write("%d "%maxlabellength)

# Compute NumVble and MaxVbleLength.
numvble = 0;
maxvblelength = 0;
qq = pp.get_node_list()
for i in xrange(len(qq)):
    x = qq[i].get_name()
    if x == 'root':
        node_idx = -1
    elif x == 'graph':
        node_idx = -2
    else:
        node_idx = int(x)
    y = qq[i].get_label()
    y = proc_state_label(y)
    if len(y)/2 > numvble:
        numvble = len(y)/2

    for j in xrange(len(y)):
        if j % 2 == 0:
            if len(y[j]) > maxvblelength:
                maxvblelength = len(y[j])
cfile.write("%d %d\n"%(numvble,maxvblelength))


qq = pp.get_edge_list()
for i in xrange(len(qq)):
    x = qq[i].get_source()
    if x == 'root':
        edge_src = -1
    else:
        edge_src = int(x)
    edge_dest = int(qq[i].get_destination())
    y = qq[i].get_label()
    if y != None:
        y = y.replace('"','')
        y = y.replace("'","")
    else:
        y = 'None'


    cfile.write("%d %d %s\n"%(edge_src,edge_dest,y))


qq = pp.get_node_list()
for i in xrange(len(qq)):
    x = qq[i].get_name()
    if x == 'root':
        node_idx = -1
    elif x == 'graph':
        node_idx = -2
    else:
        node_idx = int(x)
    y = qq[i].get_label()
    y = proc_state_label(y)
    cfile.write("%d %d"%(node_idx,len(y)/2))
    for j in xrange(len(y)):
        if j % 2 == 1:
            try:
                t = int(y[j])
                cfile.write(" %d"%t)
            except ValueError:
                cfile.write(" %s"%y[j])
        else:
            
            cfile.write(" %s"%y[j])
    cfile.write("\n")


    # If the number of input parameters is 2, then skip the generation of isolation components and faulty states.
    if len(sys.argv) == 3:
        continue


    # Output isolation component:


    if qq[i].get_shape() == "doubleoctagon":
        x = qq[i].get_name()
        rr = pp.get_edge_list()
        for k in xrange(len(rr)):
            q = rr[k].get_destination()
            if q == x:
                p = rr[k].get_source()
                for u in xrange(len(qq)):
                    if qq[u].get_name() == p:
                        break
                break
        y = qq[u].get_label()
        y = proc_state_label(y)
        print y

        negstr = "[negation,pos(0,0,0,0,0,0),"
        conjstr = "[conjunct,pos(0,0,0,0,0,0),"
        equstr = "[equal,pos(0,0,0,0,0,0),"
        idenstr = "[identifier,pos(0,0,0,0,0,0),"
        intstr = "[integer,pos(0,0,0,0,0,0),"

        initstr = "initialisation(pos(0,0,0,0,0,0),sequence(pos(0,0,0,0,0,0),["
        assstr = "assign(pos(0,0,0,0,0,0),[identifier(pos(0,0,0,0,0,0),"

        resstr = ""
        init_resstr = ""

        for j in xrange(len(y)):
            
            if j % 2 == 1:
                try:
                    t = int(y[j])
                    tmpstr = tmpstr + intstr + "%d]]"%t
                    init_tmpstr = init_tmpstr + "[integer(pos(0,0,0,0,0,0),%d)])"%t
                except ValueError:
                    if y[j] == "TRUE":
                        t = "[boolean_true,pos(0,0,0,0,0,0)]]"
                        init_t = "[boolean_true(pos(0,0,0,0,0,0))])"
                    elif y[j] == "FALSE":
                        t = "[boolean_false,pos(0,0,0,0,0,0)]]"
                        init_t = "[boolean_false(pos(0,0,0,0,0,0))])"
                    else:
                        print ("Warning: Cannot analyse %s!"%y[j])
                        t = "%s-Error!"%y[j]
                        init_t = "%s-Error!"%y[j]
                    tmpstr = tmpstr + "%s"%t
                    init_tmpstr = init_tmpstr + "%s"%init_t
            else:
                tmpstr = equstr + idenstr + "%s],"%y[j]
                init_tmpstr = assstr + "%s)],"%y[j]
            if j >= 3 and j % 2 == 1:
                resstr = conjstr + resstr + "," + tmpstr + "]"
                init_resstr = init_resstr + "," + init_tmpstr
            elif j == 1:
                resstr = tmpstr
                init_resstr = init_tmpstr

        resstr = negstr + resstr + "]."
        init_resstr = initstr + init_resstr + "]))."

        print "Print the isolation component to: ", sys.argv[3]
        isocompfile = open(sys.argv[3],'w')
        print(resstr)
        isocompfile.write(resstr)

        print "Print the initialisation form of the isolation component to: ", sys.argv[3]
        isocompfile = open(sys.argv[3] + ".init",'w')
        print(init_resstr)
        isocompfile.write(init_resstr)



    # Output the faulty state:


    if qq[i].get_shape() == "doubleoctagon":
        x = qq[i].get_name()
        y = qq[i].get_label()
        y = proc_state_label(y)

        print "The faulty state is", y
        print "Print the faulty state to: ", sys.argv[4]
        fsfile = open(sys.argv[4],'w')
        for j in xrange(len(y)):
            fsfile.write(y[j] + '\n')

cfile.close()

if len(sys.argv) == 5:
    isocompfile.close()
    fsfile.close()
"""
"""
import os
from graphviz import Source
spfile = open('statespace.txt', 'r')
text=spfile.read()
print text[4]
Source(text)
#print text
"""
