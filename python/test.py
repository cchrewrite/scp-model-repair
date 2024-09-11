import os
import sys
import RepSimpLib
import Bmch

VL = ["Attr","Price"]
#VL = ["inpA" ,"inpB","inpc","out","outc","valid"]
conffile = "ReachabilityRepair/ReachRepData/config/SKCART_config"
wdir = "ReachabilityRepair/clustering/"

f = open("pos.exp","r")
x = f.readlines()[0]
PExp = eval(x)
f.close()

f = open("neg.exp","r")
x = f.readlines()[0]
NExp = eval(x)
f.close()

print PExp
print NExp

#SRL = RepSimpLib.CFGReachabilityRepairSimplification(PExp,NExp,VL,conffile,wdir)


f = open("SRL.tmp","r")
x = f.readlines()[0]
SRL = eval(x)
f.close()

MS = "MS_pp.mch"
with open(MS) as mchf:
    mch = mchf.readlines()
mch = [x.strip() for x in mch]

for X in SRL:
    op = X[0]
    rep = X[1]
    mch = Bmch.apply_S_change(mch,op,rep)
for X in mch: print X

f = open("sel_mch.mch","w")
for X in mch:
    f.write(X)
    f.write("\n")
f.close()

cmd = "python src/python/state_graph_comparison.py ReachabilityRepair/ReachRepData/machines/AA_House.mch sel_mch.mch ReachabilityRepair/comparison"
os.system(cmd)

raw_input("GGGGG")

pppppp

LP = []
LN = []
op = "Add_DVD"
for i in xrange(len(PExp)):
    if PExp[i][1] == op and not(PExp[i][0] in LP):
        LP.append(PExp[i][0])
PNExp = PExp + NExp
for i in xrange(len(PNExp)):
    if not(PNExp[i][0] in LP) and not(PNExp[i][0] in LN):
        LN.append(PNExp[i][0])

S = RepSimpLib.DeductivePreConditionSimplification(LP,LN,VL,conffile,wdir)

ppppppp
#S = RepSimpLib.DeductiveClustering(TLP,VL,conffile,wdir)
#for X in S: print X

#raw_input("DCDC")

#P = RepSimpLib.ComputeAlphaScore(S)

#for X in P: print X

#raw_input("CACA")

ppppp
cmd_list = []

# Repair Simplification (CIDR)
#cmd_list.append("python src/python/repair_simplification.py ILP_exp/AutoBusDetOrg.mch ILP_exp/AutoBusDetIsoRev.mch ILP_exp/res_tmp/")

# Fast B-repair
# cmd_list.append("python src/python/b_repair_fast.py ILP_exp/AutoBusOrg.mch NEW ILP_exp/oracle.empty ILP_exp/config ILP_exp/b_repair_fast_result/")

# Making a faulty B machine - Method A
#cmd_list.append("python src/python/make_faulty_machine_A.py ILP_exp/AutoBusDetIsoRev.mch 20 777 ILP_exp/machine_maker_result/")


"""
# Semantics Learning Test Scripts
#mchfile = "CSM.mch"
#mchfile = "Cruise_finite1.mch"
#mchfile = "Jukebox.mch"
#mchfile = "scheduler.mch"
#mchfile = "Locks.mch"
#mchfile = "CSE_Test2.mch" # This is a counter so that it takes very long time.
#mchfile = "IncrementalStatePackingTest.mch"
#mchfile = "Lift0.mch" # This is a counter so that it takes very long time.
#mchfile = "ParallelModelCheckTest.mch"
#mchfile = "POR_TwoThreads_WithSync.mch" # This is a counter so that it takes very long time.
#mchfile =  "QueryOperations.mch" # This is a counter so that it takes very long time.
#mchfile = "scheduler6.mch"
#mchfile = "club.mch"
#mchfile = "Jobshop.mch"
#mchfile = "Paperround.mch"
#mchfile = "progress.mch"
#mchfile = "Sortarray.mch"
wdir = "SemanticsLearningConference/dataset/selected"
#wdir = "SemanticsLearningConference/IJCAI_dataset_Others/selected/"
FL = os.listdir(wdir)
FL.sort()
for fn in FL:
    cmd_list = []
    if fn[len(fn)-4:len(fn)] != ".mch": continue
    x = "a"
    while x != "y" and x != "n":
        x = raw_input("Evaluating %s? (y/n): "%fn)
    if x != "y": continue
    mchfile = fn
    cmd_list.append("rm -r SemanticsLearningConference/faulty_model/")
    cmd_list.append("rm -r SemanticsLearningConference/b_repair_fast_result/")
    cmd_list.append("python src/python/make_faulty_machine_A.py %s/%s 10 777 SemanticsLearningConference/faulty_model/"%(wdir,mchfile))
    cmd_list.append("python src/python/b_repair_fast.py SemanticsLearningConference/faulty_model/result.mch NEW SemanticsLearningConference/faulty_model/answer.txt SemanticsLearningConference/config SemanticsLearningConference/b_repair_fast_result/")
    cmd_list.append("python src/python/state_graph_comparison.py SemanticsLearningConference/b_repair_fast_result/result.mch %s/%s SemanticsLearningConference/b_repair_fast_result/comparison/"%(wdir,mchfile))
    for x in cmd_list:
        os.system(x)
    cmd_list = []
"""

"""
# Fast B-repair Evaluation - small model.
cmd_list.append("python src/python/make_faulty_machine_A.py ILP_exp/AutoBusDetIsoRev.mch 20 777 ILP_exp/machine_maker_result/")
cmd_list.append("python src/python/b_repair_fast.py ILP_exp/machine_maker_result/result.mch NEW ILP_exp/machine_maker_result/answer.txt ILP_exp/config ILP_exp/b_repair_fast_result/")
"""
"""
# Fast B-repair Evaluation - big model and different seeds.
num_faults = 200
num_seeds = 1
cmd_list.append("rm -r ILP_exp/b_repair_fast_summary/")
cmd_list.append("mkdir ILP_exp/b_repair_fast_summary/")

i = 0 
for i in xrange(num_seeds):
    sd = 777 + i
    cmd_list.append("rm -r ILP_exp/machine_maker_result/")
    cmd_list.append("rm -r ILP_exp/b_repair_fast_result/")
    cmd_list.append("python src/python/make_faulty_machine_A.py ILP_exp/House_Answer.mch %d %d ILP_exp/machine_maker_result/"%(num_faults,sd))
    cmd_list.append("python src/python/b_repair_fast.py ILP_exp/machine_maker_result/result.mch NEW ILP_exp/machine_maker_result/answer.txt ILP_exp/config_House_Answer ILP_exp/b_repair_fast_result/")
    cmd_list.append("cp ILP_exp/b_repair_fast_result/summary ILP_exp/b_repair_fast_summary/%d.summary"%i)

cmd_list.append("python src/python/count_results_b_repair_fast.py ILP_exp/b_repair_fast_summary/")
"""

for x in cmd_list:
    os.system(x)

