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

# A general evaluator to evaluate B model quality.
# Usage: python [this script] [B model] [evaluation data] [configuration file] [result folder]

# =================================================

# e.g. python src/python/b_model_evaluation.py EvaluationCriteria/model/counter.mch EvaluationCriteria/model/eval.data EvaluationCriteria/model/config EvaluationCriteria/result/

sd = 777

if len(sys.argv) != 5:
    print "Error: The number of input parameters should be 5."
    print "Usage: python [this script] [B model] [evaluation data] [configuration file] [result folder]"
    exit(1)


M = sys.argv[1]
ED = sys.argv[2]
conffile = sys.argv[3]
resdir = sys.argv[4]

print "B Model:", M
print "Evaluation Data:", ED
print "Configuration File:", conffile
print "Result Folder:", resdir




print "Data Preparation..."


prob_changes = Bmch.read_config(conffile,"prob_changes","float")
mc_time_limit = Bmch.read_config(conffile,"mc_time_limit","int")
word_limit = Bmch.read_config(conffile,"word_limit","int")


max_num_words = word_limit

def div_sp(x,y):
    if abs(y) < pow(10,-100):
        if x * y >= 0:
            return 1.0
        else:
            return -1.0
    return (x * 1.0) / (y * 1.0)


cmd = "mkdir %s"%resdir
os.system(cmd)

s = resdir + "/model.mch"
cmd = "cp %s %s"%(M,s)
os.system(cmd)
M = s

s = resdir + "/evaluation.data"
cmd = "cp %s %s"%(ED,s)
os.system(cmd)
ED = s

fn = resdir + "/model_pp.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,M)
os.system(oscmd)
M = fn

print "Read evaluation data..."
f = open(ED,"r")
T_desired = []
flag = 0
for x in f.readlines():
    if "BEGIN_DESIRED_STATE_TRANSITIONS" in x:
        flag = 1
        continue
    if "END_DESIRED_STATE_TRANSITIONS" in x:
        flag = 2
    if flag == 0: continue
    if flag == 2: break
    if "FORMAT:" in x: continue
    y = eval(x)
    T_desired.append(y)
f.close()

S_desired = RepSimpLib.extract_all_states(T_desired)

G = []
f = open(ED,"r")
flag = 0
for x in f.readlines():
    if "BEGIN_GOALS" in x:
        flag = 1
        continue
    if "END_GOALS" in x:
        flag = 2
    if flag == 0: continue
    if flag == 2: break
    y = eval(x)
    G.append(y)
f.close() 

with open(M) as mchf:
    mch = mchf.readlines()
mch = [x.strip() for x in mch]


print "Deriving the state space of the B model..."

T_derived = resdir + "/T_derived.txt"
util_log_file = resdir + "/util.log"
max_initialisations = Bmch.read_config(conffile,"max_initialisations","int")
max_operations = Bmch.read_config(conffile,"max_operations","int")
bscope = Bmch.generate_training_set_condition(mch)
oscmd = "/usr/bin/time -o %s -v ./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(util_log_file,M,max_initialisations,max_operations,bscope,T_derived)
os.system(oscmd)

f = open(util_log_file,"r")
util_log = []
for x in f.readlines():
    util_log = util_log + x.replace("\t","").split("\n")
f.close()

sg = Bgenlib.BStateGraphForNN()
sg.ReadStateGraph(T_derived)
T_derived = sg.GetTransList()
S_derived = RepSimpLib.extract_all_states(T_derived)
FS_derived = sg.GetStatesWithoutOutgoingTransitions(T_derived)
TS_derived = Bmch.list_difference(S_derived,FS_derived)
VList = sg.GetVbleList()
SType = sg.GetSetTypeFromTransList(T_derived)



T_derived = Bmch.list_union(T_derived,[])
T_desired = Bmch.list_union(T_desired,[])

"""
for x in T_derived: print x
print len(T_derived)
ppppp
"""

FT_derived = []
TT_derived = []
for x in T_derived:
    if x[2] in FS_derived:
        FT_derived.append(x)
    else:
        TT_derived.append(x)


TF_derived = []
for x in T_derived:
    TF_derived.append(x[0] + [x[1]] + x[2])

TF_desired = []
for x in T_desired:
    TF_desired.append(x[0] + [x[1]] + x[2])
TF_ali = Bmch.align_two_sets_of_lists(TF_derived,TF_desired)

Total_Functional_Completeness = div_sp(len(Bmch.list_intersect(T_derived,T_desired)) * 1.0, len(T_desired))
print "Total_Functional_Completeness:", Total_Functional_Completeness

Partial_Functional_Completeness = div_sp(TF_ali[4] * 1.0, Bmch.size_of_a_set_of_lists(TF_desired))
print "Partial_Functional_Completeness:", Partial_Functional_Completeness


Total_Functional_Correctness = div_sp(len(Bmch.list_intersect(T_derived,T_desired)) * 1.0, len(T_derived))
print "Total_Functional_Correctness:", Total_Functional_Correctness


Partial_Functional_Correctness = div_sp(TF_ali[4] * 1.0, Bmch.size_of_a_set_of_lists(TF_derived))
print "Partial_Functional_Correctness:", Partial_Functional_Correctness




P_desired = []
for x in T_desired:
    P_desired.append(x[0]+x[2])
P_desired = Bmch.list_union(P_desired,[])

P_derived = []
for x in T_derived:
    P_derived.append(x[0]+x[2])
P_derived = Bmch.list_union(P_derived,[])

Total_Functional_Appropriateness = div_sp(len(Bmch.list_intersect(P_derived,P_desired)) * 1.0, len(P_desired))
print "Total_Functional_Appropriateness:", Total_Functional_Appropriateness

P_ali = Bmch.align_two_sets_of_lists(P_derived,P_desired)

Partial_Functional_Appropriateness = div_sp(P_ali[4] * 1.0, Bmch.size_of_a_set_of_lists(P_desired))

print "Partial_Functional_Appropriateness:", Partial_Functional_Appropriateness

Total_Model_Checking_CPU_Time = 0.0
for x in util_log:
    
    if "User time (seconds):" in x:
        y = x.replace("User time (seconds):","")
        Total_Model_Checking_CPU_Time = Total_Model_Checking_CPU_Time + float(y)
    if "System time (seconds):" in x:
        y = x.replace("System time (seconds):","")
        Total_Model_Checking_CPU_Time = Total_Model_Checking_CPU_Time + float(y)
    if "Maximum resident set size (kbytes):" in x:
        y = x.replace("Maximum resident set size (kbytes):","")
        Peak_Memory_Usage = float(y) / 1024



print "Total_Model_Checking_CPU_Time (s):", Total_Model_Checking_CPU_Time

print "Peak_Memory_Usage (MB):", Peak_Memory_Usage

Capacity = len(T_derived) + len(S_derived)

print "Capacity:", Capacity

Invariant_Satisfability = div_sp(len(TT_derived) * 1.0, len(T_derived))
print "Invariant_Satisfability:", Invariant_Satisfability

F_desired = []
for x in T_desired:
    F_desired.append(x[1])
F_desired = Bmch.list_union(F_desired,[])

F_derived = []
FF_derived = []
for x in T_derived:
    F_derived.append(x[1])
    if x[2] in FS_derived:
        FF_derived.append(x[1])
F_derived = Bmch.list_union(F_derived,[])
FF_derived = Bmch.list_union(FF_derived,[])
TF_derived = Bmch.list_difference(F_derived,FF_derived)

Availability = len(Bmch.list_intersect(TF_derived,F_desired)) * 1.0 / len(F_desired)


NDS_traced = []
TTmp = T_derived + []
TTmp.sort(key = lambda x:x[2])
for i in xrange(len(TTmp)):
    x = TTmp[i]
    for j in xrange(i+1,len(TTmp)):
        y = TTmp[j]
        if y[2] == x[2]:
            NDS_traced.append(x[2])
            break
DS_traced = Bmch.list_difference(S_derived,NDS_traced)

Accountability = div_sp(len(DS_traced) * 1.0, len(S_derived))
print "Accountability:", Accountability

#mc_scale = Bmch.read_config(conffile,"mc_scale","int")
mc_scale = 100

print "Deriving the state space of the B model using bounded model checking..."

TMC_derived = resdir + "/TMC_derived.txt"
max_initialisations = Bmch.read_config(conffile,"max_initialisations","int")
max_operations = Bmch.read_config(conffile,"max_operations","int")
bscope = Bmch.generate_training_set_condition(mch)
oscmd = "./../ProB/probcli %s -mc %d -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(M,mc_scale,max_initialisations,max_operations,bscope,TMC_derived)
os.system(oscmd)

sgMC = Bgenlib.BStateGraphForNN()
sgMC.ReadStateGraph(TMC_derived)
TMC_derived = sgMC.GetTransList()
SMC_derived = RepSimpLib.extract_all_states(TMC_derived)
FSMC_derived = sgMC.GetStatesWithoutOutgoingTransitions(TMC_derived)
TSMC_derived = Bmch.list_difference(SMC_derived,FSMC_derived)

FTMC_derived = []
TTMC_derived = []
for x in TMC_derived:
    if x[2] in FSMC_derived:
        FTMC_derived.append(x)
    else:
        TTMC_derived.append(x)




FMC_derived = []
for x in TMC_derived:
    if not(x[1] in FMC_derived):
        FMC_derived.append(x[1])



Modularity_List = []

for ope in FMC_derived:
    TMC_derived_ope = []
    for x in TMC_derived:
        if x[1] == ope:
            TMC_derived_ope.append(x)
    N_changes = int(len(TMC_derived_ope) * prob_changes)
    if N_changes == 0:
        N_changes = 1
 
    mch_changed_ope = mch + []

    # deletions:
    epid = "eval0"
    T_deleted_ope = []
    for i in xrange(N_changes):
        x = TMC_derived_ope[int(random.random() * len(TMC_derived_ope))]
        T_deleted_ope.append(x + ["isolation"])
    R_deletion_ope = []
    for x in T_deleted_ope:
        y = Bmch.atomic_modification_or_deletion_to_conditional_substitution(x,VList,epid)
        R_deletion_ope.append(y)

    RL = []
    for x in R_deletion_ope:
        RL.append([x[1],x[2]])

    if RL != []:
        mch_changed_ope = Bmch.apply_modifications_and_deletions(mch_changed_ope,ope,RL,VList,epid)
    #for x in mch_changed_ope: print x
    #pppp
   
    # insertions:
    T_inserted_ope = []
    for i in xrange(N_changes):
        p = SMC_derived[int(random.random() * len(SMC_derived))]
        q = Bmch.GetRandomState(SType)
        T_inserted_ope.append([p,ope,q])

    R_insertion_ope = RepSimpLib.AtomicReachabilityRepair(T_inserted_ope,VList)
    #for x in R_insertion_ope: print x
    rep = R_insertion_ope[0][1]
    mch_changed_ope = Bmch.apply_insertions(mch_changed_ope,ope,rep)
    #mch_changed_ope = Bmch.apply_S_change(mch_changed_ope,ope,rep)
    for x in mch_changed_ope: print x

    M_changed_ope = resdir + "/changed_%s.mch"%ope
    f = open(M_changed_ope,"w")
    for x in mch_changed_ope:
        f.write(x)
        f.write("\n")
    f.close()

    print "Deriving the state space of the B model with the changed operation %s..."%ope
    T_changed_ope = resdir + "/T_changed_%s.txt"%ope
    max_initialisations = Bmch.read_config(conffile,"max_initialisations","int")
    max_operations = Bmch.read_config(conffile,"max_operations","int")
    bscope_changed_ope = Bmch.generate_training_set_condition(mch_changed_ope)
    oscmd = "./../ProB/probcli %s -mc %d -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(M_changed_ope,mc_scale,max_initialisations,max_operations,bscope_changed_ope,T_changed_ope)
    os.system(oscmd)

    sg_changed_ope = Bgenlib.BStateGraphForNN()
    sg_changed_ope.ReadStateGraph(T_changed_ope)
    T_changed_ope = sg_changed_ope.GetTransList()
    S_changed_ope = RepSimpLib.extract_all_states(T_changed_ope)
    FS_changed_ope = sg_changed_ope.GetStatesWithoutOutgoingTransitions(T_changed_ope)
    TS_changed_ope = Bmch.list_difference(S_changed_ope,FS_changed_ope)

    T_changed_ope = Bmch.list_union(T_changed_ope,[])
 
    T_changed_ope_others = []
    for x in T_changed_ope:
        if not(x[1] == ope):
            T_changed_ope_others.append(x)

    TMC_derived_others = []
    for x in TMC_derived:
        if not(x[1] == ope):
            TMC_derived_others.append(x)

    S1 = Bmch.list_intersect(T_changed_ope_others,TMC_derived_others)
    S2 = Bmch.list_union(T_changed_ope_others,TMC_derived_others)
     
    Modularity_ope = div_sp(len(S1) * 1.0, len(S2))

    Modularity_List.append([ope,Modularity_ope,len(TMC_derived_ope)])

print "Modularity of each operation:",Modularity_List

Modularity = 0.0
for x in Modularity_List:
    Modularity = Modularity + x[1] * x[2]

Modularity = div_sp(Modularity, len(TMC_derived))

print "Modularity:", Modularity

Reusability = 1.0 - div_sp(len(FMC_derived) * 1.0, len(TMC_derived))
print "Reusability:", Reusability


    
N_changes = int(len(TMC_derived) * prob_changes)
if N_changes == 0:
    N_changes = 1
 
mch_changed = mch + []


# deletions:
epid = "eval0"
T_deleted = []
for i in xrange(N_changes):
    x = TMC_derived[int(random.random() * len(TMC_derived))]
    T_deleted.append(x + ["isolation"])
R_deletion = []
for x in T_deleted:
    y = Bmch.atomic_modification_or_deletion_to_conditional_substitution(x,VList,epid)
    R_deletion.append(y)

for ope in FMC_derived:
    RL = []
    for x in R_deletion:
        if x[0] == ope:
            RL.append([x[1],x[2]])
    if RL != []:
        mch_changed = Bmch.apply_modifications_and_deletions(mch_changed,ope,RL,VList,epid)


   
# insertions:
T_inserted = []
for i in xrange(N_changes):
    p = SMC_derived[int(random.random() * len(SMC_derived))]
    ope = FMC_derived[int(random.random() * len(FMC_derived))]
    q = Bmch.GetRandomState(SType)
    T_inserted.append([p,ope,q])

R_insertion = RepSimpLib.AtomicReachabilityRepair(T_inserted,VList)
for X in R_insertion:
    ope = X[0]
    rep = X[1]
    mch_changed = Bmch.apply_insertions(mch_changed,ope,rep)

    #mch_changed = Bmch.apply_S_change(mch_changed,ope,rep)
    
M_changed = resdir + "/changed.mch"
f = open(M_changed,"w")
for x in mch_changed:
    f.write(x)
    f.write("\n")
f.close()

print "Deriving the state space of the changed B model..."
T_changed = resdir + "/T_changed.txt"
max_initialisations = Bmch.read_config(conffile,"max_initialisations","int")
max_operations = Bmch.read_config(conffile,"max_operations","int")
bscope_changed = Bmch.generate_training_set_condition(mch_changed)
oscmd = "./../ProB/probcli %s -mc %d -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(M_changed,mc_scale,max_initialisations,max_operations,bscope_changed,T_changed)
os.system(oscmd)

sg_changed = Bgenlib.BStateGraphForNN()
sg_changed.ReadStateGraph(T_changed)
T_changed = sg_changed.GetTransList()
S_changed = RepSimpLib.extract_all_states(T_changed)
FS_changed = sg_changed.GetStatesWithoutOutgoingTransitions(T_changed)
TS_changed = Bmch.list_difference(S_changed,FS_changed)

T_changed = Bmch.list_union(T_changed,[])

CT_changed = Bmch.list_difference(Bmch.list_union(T_changed,T_deleted),T_inserted)

CFT_changed = []
CTT_changed = []
for x in CT_changed:
    if x[0] in FS_changed or x[2] in FS_changed:
        CFT_changed.append(x)
    else:
        CTT_changed.append(x)

S1 = Bmch.list_intersect(CT_changed,TMC_derived)
S2 = Bmch.list_union(CT_changed,TMC_derived)
print len(S1),len(S2) 
Functional_Analysability = 1.0 - div_sp(len(S1) * 1.0, len(S2))


S1 = Bmch.list_intersect(CFT_changed,FTMC_derived)
S2 = Bmch.list_union(CFT_changed,FTMC_derived)

Fault_Analysability = 1.0 - div_sp(len(S1) * 1.0, len(S2))

Testability = 1.0 - div_sp(min(Total_Model_Checking_CPU_Time,mc_time_limit) * 1.0, mc_time_limit)

print "Functional_Analysability:", Functional_Analysability

print "Fault_Analysability:", Fault_Analysability
print "Testability:", Testability

if G == []:
    Goal_Appropriateness = -1
else:
    gwdir = resdir + "/GOAL_counts/"
    GOALs_counts = Bmch.count_goals(M,G,conffile,gwdir)
    for x in GOALs_counts: print x
    N_GOALs = 0
    SN_GOALs = 0
    for x in GOALs_counts:
        N_GOALs = N_GOALs + x[1]
        SN_GOALs = SN_GOALs + min(x[1],x[2])
    Goal_Appropriateness = div_sp(SN_GOALs * 1.0, N_GOALs)

print "Goal_Appropriateness:", Goal_Appropriateness

N_words = Bmch.count_words(M,resdir)
Learnability = 1.0 - div_sp(min(max_num_words,N_words) * 1.0, max_num_words)
print "Learnability:", Learnability



# Here, we add two criteria for Reliability:

Fault_Tolerance = 1.0 - div_sp(len(CFT_changed) * 1.0, len(CT_changed))

print "Fault_Tolerance:", Fault_Tolerance

Recoverability = div_sp(len(Bmch.list_intersect(CTT_changed,TMC_derived)) * 1.0, len(TMC_derived))

print "Recoverability:", Recoverability



print ("*************** Evaluation Done! ****************")

summary_text = []

summary_text.append("*************** SUMMARY ****************")

summary_text.append("Total_Functional_Completeness: %s"%Total_Functional_Completeness)
summary_text.append("Partial_Functional_Completeness: %s"%Partial_Functional_Completeness)
summary_text.append("Total_Functional_Correctness: %s"%Total_Functional_Correctness)
summary_text.append("Partial_Functional_Correctness: %s"%Partial_Functional_Correctness)
summary_text.append("Total_Functional_Appropriateness: %s"%Total_Functional_Appropriateness)
summary_text.append("Partial_Functional_Appropriateness: %s"%Partial_Functional_Appropriateness)
summary_text.append("Total_Model_Checking_CPU_Time (s): %s"%Total_Model_Checking_CPU_Time)
summary_text.append("Peak_Memory_Usage (MB): %s"%Peak_Memory_Usage)
summary_text.append("Capacity: %s"%Capacity)
summary_text.append("Invariant_Satisfability: %s"%Invariant_Satisfability)
summary_text.append("Availability: %s"%Availability)
summary_text.append("Accountability: %s"%Accountability)
summary_text.append("Modularity of each operation: %s"%Modularity_List)
summary_text.append("Modularity: %s"%Modularity)
summary_text.append("Reusability: %s"%Reusability)
summary_text.append("Functional_Analysability: %s"%Functional_Analysability)
summary_text.append("Fault_Analysability: %s"%Fault_Analysability)
summary_text.append("Testability: %s"%Testability)
summary_text.append("Goal_Appropriateness: %s"%Goal_Appropriateness)
summary_text.append("Learnability: %s"%Learnability)
summary_text.append("Fault_Tolerance: %s"%Fault_Tolerance)
summary_text.append("Recoverability: %s"%Recoverability)

for x in summary_text: print x

resfile = resdir + "/RESULT"
f = open(resfile,"w")
for x in summary_text:
    f.write(x)
    f.write("\n")
f.close()



