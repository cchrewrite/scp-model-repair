import os
import sys
import time

cmd = "ulimit -v 6000000"
os.system(cmd)
cmd = "ulimit -f 500000000"
os.system(cmd)

cmd_list = []

for sd in xrange(778,888):
    num_goals = 100
    rdepth = 1

    datadir = "ReachabilityRepair/ReachRepData"
    FL = os.listdir(datadir + "/machines/")
    FL.sort()


    for sem_mdl_type in ["SKCART","BNBayes","LR","SVM","MLP"]:
        for fn in FL:
            #fn = "BinomialCoefficientConcurrent.mch"
            #fn = "BridgeTransitions.mch"

            if fn[len(fn)-4:len(fn)] != ".mch": continue

            mch_id = fn[0:len(fn)-4]

            x = "a"
            while x != "y" and x != "n":
                x = raw_input("Evaluating %s with %s and %d goals?(y/n): "%(fn,sem_mdl_type,num_goals))
            if x != "y": continue

            mchfile = datadir + "/machines/" + fn
            conffile = datadir + "/config/" + sem_mdl_type + "_config"
            resdir = "ReachabilityRepair/reachability_repair_results"
            mdldir = resdir + "/mdl"
            logdir = resdir + "/log"
            logfile = logdir + "/%s_%s_ng%d_rd%d_sd%d.RESULT"%(mch_id,sem_mdl_type,num_goals,rdepth,sd)

            cmd_list = []
            cmd_list.append("mkdir %s"%(resdir))
            
            
            cmd_list.append("rm -r %s"%(mdldir))
            cmd_list.append("mkdir %s"%(mdldir))
            cmd_list.append("python src/python/make_faulty_machine_M.py %s %d %d %d %s/faulty_model"%(mchfile,num_goals,rdepth,sd,mdldir))
            
            cmd_list.append("python src/python/b_reachability_repair_fast.py %s/faulty_model/result.mch %s/faulty_model/goal.txt NEW %s/faulty_model/answer.txt %s %s/repaired_model/"%(mdldir,mdldir,mdldir,conffile,mdldir))
            cmd_list.append("python src/python/state_graph_comparison.py %s/repaired_model/result.mch %s %s/comparison/"%(mdldir,mchfile,mdldir))

            cmd_list.append("mkdir %s"%(logdir))
            cmd_list.append("cp %s/repaired_model/summary %s"%(mdldir,logfile))
            for x in cmd_list:
                os.system(x)
            cmd_list = []
            time.sleep(3)


