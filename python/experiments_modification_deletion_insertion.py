import os
import sys
import time

cmd_list = []

for sd in xrange(778,788):

    num_faults = 100

    datadir = "GeneticModificationExperiments/AbsMacModData"
    FL = os.listdir(datadir + "/machines/")
    FL.sort()

    for fn in FL:
        #fn = "BinomialCoefficientConcurrent.mch"
        #fn = "BridgeTransitions.mch"
        #fn = "club.mch"

        if fn[len(fn)-4:len(fn)] != ".mch": continue

        mch_id = fn[0:len(fn)-4]

        for sem_mdl_type in ["SKCART"]: #["SKCART","Silas","BNBayes","LR","SVM","MLP"]:
            #sem_mdl_type = "SKCART"

            x = "a"
            while x != "y" and x != "n":
                x = raw_input("Evaluating %s with %s and %d faults?(y/n): "%(fn,sem_mdl_type,num_faults))
            if x != "y": continue

            mchfile = datadir + "/machines/" + fn
            conffile = datadir + "/config/" + sem_mdl_type + "_config"
            resdir = "GeneticModificationExperiments/abstract_machine_modification_results"
            mdldir = resdir + "/mdl"
            logdir = resdir + "/log"
            logfile = logdir + "/%s_%s_%d_sd%d.RESULT"%(mch_id,sem_mdl_type,num_faults,sd)

            cmd_list = []

            """            
            cmd_list.append("mkdir %s"%(resdir))
            cmd_list.append("rm -r %s"%(mdldir))
            cmd_list.append("mkdir %s"%(mdldir))
            cmd_list.append("python src/python/make_faulty_machine_A.py %s %d %d %s/faulty_model"%(mchfile,num_faults,sd,mdldir))

            cmd_list.append("python src/python/repair_reachability.py %s/faulty_model/result.mch %s/faulty_model/goal.txt NEW %s/faulty_model/answer.txt %s %s/repaired_model/"%(mdldir,mdldir,mdldir,conffile,mdldir))

            """

            cmd_list.append("python src/python/repair_deadlock_states.py %s/faulty_model/result.mch NEW %s/faulty_model/answer.txt %s %s/repaired_model/"%(mdldir,mdldir,conffile,mdldir))

            #cmd_list.append("python src/python/repair_invariant_violations.py %s/faulty_model/result.mch NEW %s/faulty_model/answer.txt %s %s/repaired_model/"%(mdldir,mdldir,conffile,mdldir))

            
            #cmd_list.append("python src/python/state_graph_comparison.py %s/repaired_model/result.mch %s %s/comparison/"%(mdldir,mchfile,mdldir))

            cmd_list.append("mkdir %s"%(logdir))
            cmd_list.append("cp %s/repaired_model/summary %s"%(mdldir,logfile))
            for x in cmd_list:
                os.system(x)
            cmd_list = []
            time.sleep(3)


