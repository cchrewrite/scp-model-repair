import os
import sys
import time

cmd_list = []

datadir = "TOSEM_Experiments/ALTC_36"
resdir = "TOSEM_Experiments/result_ALTC_36"

SUBL = [["AMS","House"],["CMS","CourseManagementSystem"],["LCS","Lift"],["TPA","TennisPlayer"]]

for sem_mdl_type in ["SKCART"]: #["SKCART","LR","ResNet"]:#["LR"]: #["ResNet"]: #["SKCART"]: #["SKCART","Silas","BNBayes","LR","SVM","MLP"]:

    for subj in SUBL:

        fdir = subj[0]
        fn = subj[1] + ".mch"

        mch_id = fn[0:len(fn)-4]

        for fault_type in ["IV1","IV5","IV10","AV1","AV5","AV10","DL1","DL5","DL10"]:
            # SKCART set1
            #if not(sem_mdl_type == "SKCART"): continue
            #if not([fdir,fault_type] in [["AMS","IV5"],["AMS","AV5"],["AMS","AV10"],["AMS","DL1"],["AMS","DL5"],["AMS","DL10"],["LCS","IV5"],["LCS","IV10"],["LCS","AV10"],["LCS","DL1"],["LCS","DL5"],["LCS","DL10"],["TPA","IV5"],["TPA","AV5"],["TPA","AV10"],["TPA","DL1"],["TPA","DL5"],["TPA","DL10"],["CMS","IV1"],["CMS","IV5"],["CMS","IV10"],["CMS","DL1"],["CMS","DL5"],["CMS","DL10"]]): continue
            
            # SKCART set2
            if not(sem_mdl_type == "SKCART"): continue
            if not([fdir,fault_type] in [["AMS","IV5"],["AMS","DL1"],["LCS","IV5"],["LCS","IV10"],["LCS","AV10"],["TPA","AV5"],["TPA","AV10"],["TPA","DL1"],["TPA","DL5"],["TPA","DL10"],["CMS","IV1"],["CMS","IV10"],["CMS","DL1"]]): continue

            # ResNet set1
            #if not(sem_mdl_type == "ResNet"): continue
            #if not([fdir,fault_type] in [["AMS","IV10"],["AMS","AV1"],["AMS","AV5"],["AMS","AV10"],["AMS","DL1"],["AMS","DL5"],["AMS","DL10"],["LCS","IV5"],["LCS","IV10"],["LCS","AV5"],["LCS","AV10"],["LCS","DL1"],["LCS","DL5"],["LCS","DL10"],["TPA","IV5"],["TPA","IV10"],["TPA","AV5"],["TPA","AV10"],["TPA","DL1"],["TPA","DL5"],["TPA","DL10"],["CMS","IV1"],["CMS","IV5"],["CMS","IV10"],["CMS","AV1"],["CMS","AV5"],["CMS","AV10"],["CMS","DL1"],["CMS","DL5"],["CMS","DL10"]]): continue

            # ResNet set2
            #if not(sem_mdl_type == "ResNet"): continue
            #if not([fdir,fault_type] in [["AMS","DL1"],["AMS","DL5"],["AMS","DL10"],["LCS","AV10"],["LCS","DL1"],["LCS","DL5"],["LCS","DL10"],["TPA","IV5"],["TPA","IV10"],["TPA","AV10"],["TPA","DL1"],["TPA","DL5"],["TPA","DL10"],["CMS","IV1"],["CMS","IV10"],["CMS","AV1"],["CMS","AV5"],["CMS","AV10"],["CMS","DL1"],["CMS","DL5"],["CMS","DL10"]]): continue


            # LR set1
            #if not(sem_mdl_type == "LR"): continue
            #if [fdir,fault_type] in [["AMS","IV5"],["AMS","IV10"],["AMS","AV10"],["AMS","DL5"],["LCS","AV1"]]: continue

            # LR set2
            #if not(sem_mdl_type == "LR"): continue
            #if [fdir,fault_type] in [["AMS","IV5"],["AMS","IV10"],["AMS","AV10"],["AMS","DL5"],["AMS","DL10"],["LCS","IV5"],["LCS","IV10"],["LCS","AV1"],["LCS","AV10"],["LCS","DL1"],["LCS","DL5"],["LCS","DL10"]]: continue

            x = "a"
            while x != "y" and x != "n":
                x = raw_input("Repairing %s --- Fault Type --- %s --- Semantic Model is %s? (y/n): "%(fn,fault_type,sem_mdl_type))
            if x != "y": continue

            correct_mchfile = datadir + "/dataset/%s/Answer_%s"%(fdir,fn)
            faulty_mchfile = datadir + "/dataset/%s/%s_%s"%(fdir,fault_type,fn)
            eval_data = datadir + "/dataset/%s/eval.data"%(fdir)

            conffile = datadir + "/config/" + sem_mdl_type + "_config"

            rprdir = resdir + "/repaired_%s_%s_%s"%(sem_mdl_type,fdir,fault_type)


            cmd_list = []
            cmd_list.append("mkdir %s"%(resdir))
            
            cmd_list.append("rm -r %s"%(rprdir))
            cmd_list.append("mkdir %s"%(rprdir))
            

            # Model Repair
            cmd_list.append("python src/python/repair_invariant_violations.py %s NEW NONE %s %s"%(faulty_mchfile,conffile,rprdir))

            # Model Comparison
            cmd_list.append("python src/python/state_graph_comparison.py %s/result.mch %s %s/comparison/"%(rprdir,correct_mchfile,rprdir))

 
            #evaldir = rprdir + "/eval"
            #cmd_list.append("rm -r %s"%(evaldir))
            #cmd_list.append("mkdir %s"%(evaldir))

            # Quality Evaluation - Faulty Model
            #cmd_list.append("python src/python/b_model_evaluation.py %s/source.mch %s/eval.data %s %s/faulty_model"%(rprdir,mdldir,conffile,evaldir))
            #cmd_list.append("cp %s/faulty_model/RESULT %s/source.eval"%(evaldir,rprdir))
            
            # Quality Evaluation - Repaired Model
            #cmd_list.append("python src/python/b_model_evaluation.py %s/result.mch %s/eval.data %s %s/repaired_model"%(rprdir,mdldir,conffile,evaldir))
            #cmd_list.append("cp %s/repaired_model/RESULT %s/result.eval"%(evaldir,rprdir))

            # Quality Comparison
            #cmd_list.append("python src/python/b_model_evaluation_comparison.py %s/source.eval %s/result.eval %s/comparison.eval"%(rprdir,rprdir,rprdir))


            for x in cmd_list:
                os.system(x)
            cmd_list = []
            time.sleep(3)


