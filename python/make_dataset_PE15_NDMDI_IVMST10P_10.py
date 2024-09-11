import os
import sys
import time

cmd_list = []


for sd in xrange(1001,1015):

    per_faults = 1

    datadir = "TOSEM_Experiments/PE15_NDMDI_IVMST%dP_10"%per_faults
    FL = os.listdir(datadir + "/material/")
    FL.sort()

    for fn in FL:
        
        if fn[len(fn)-4:len(fn)] != ".mch": continue

        mch_id = fn[0:len(fn)-4]

        if True:
            x = "y"
            while x != "y" and x != "n":
                x = raw_input("Making a test case for %s --- Seed = %d --- Percentage of Faulty State Transitions is %d/100? (y/n): "%(fn,sd,per_faults))
            if x != "y": continue

            mchfile = datadir + "/material/" + fn

            resdir = datadir + "/dataset"
            mdldir = resdir + "/%s_NDMDI_IVMST%dP_SD%d"%(mch_id,per_faults,sd)

            cmd_list = []
            cmd_list.append("mkdir %s"%(resdir))
            
            cmd_list.append("rm -r %s"%(mdldir))
            cmd_list.append("mkdir %s"%(mdldir))
            cmd_list.append("python src/python/make_faulty_machine_NDMDI.py %s %d %d %s/faulty_model"%(mchfile,per_faults,sd,mdldir))
            cmd_list.append("cp %s/faulty_model/result.mch %s/model.mch"%(mdldir,mdldir))
            cmd_list.append("cp %s/faulty_model/goal_predicates.txt %s/goal.pred"%(mdldir,mdldir))
            cmd_list.append("cp %s/faulty_model/eval.data %s/eval.data"%(mdldir,mdldir))
            cmd_list.append("rm -r %s/faulty_model"%(mdldir))

            for x in cmd_list:
                os.system(x)
            cmd_list = []
            time.sleep(3)


