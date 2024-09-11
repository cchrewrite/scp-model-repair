import os
import sys
import time

cmd_list = []


for sd in xrange(1001,1011):

    per_faults = 1

    datadir = "TOSEM_Experiments/PE15_NDMDI_IVMST%dP_10"%per_faults
    resdir = "TOSEM_Experiments/result_journal_PE15_NDMDI_IVMST%dP_10"%per_faults
    FL = os.listdir(datadir + "/material/")
    FL.sort()

    for fn in FL:
        
        if "Binomial" in fn:
            continue

        if fn[len(fn)-4:len(fn)] != ".mch": continue

        mch_id = fn[0:len(fn)-4]

        if "Cruise" in mch_id: continue
        if "GSM" in mch_id: continue
        if "Incremental" in mch_id: continue
        if "TestBZTT3" in mch_id: continue

        for sem_mdl_type in ["MLP4L","MLPAE4L","FNNRBB16L","SKCART","Random"]: #["SKCART","Silas","BNBayes","LR","SVM","MLP"]:
        #for sem_mdl_type in ["Random","BNBayes","LR","SVM"]:

            
            x = "y"
            while x != "y" and x != "n":
                x = raw_input("Simplifying %s --- Seed = %d --- Percentage of Faulty State Transitions is %d/100 --- Semantic Model is %s? (y/n): "%(fn,sd,per_faults,sem_mdl_type))
            if x != "y": continue

            original_mchfile = datadir + "/material/" + fn
            mdlname = "%s_NDMDI_IVMST%dP_SD%d"%(mch_id,per_faults,sd)

            mdldir = datadir + "/dataset/" + mdlname

            conffile = datadir + "/config/" + sem_mdl_type + "_config"

            rprdir = resdir + "/repaired_%s_%s"%(sem_mdl_type,mdlname)


            cmd_list = []
            #cmd_list.append("mkdir %s"%(resdir))
            
            cmd_list.append("rm -r %s/recon"%(rprdir))
            cmd_list.append("mkdir %s/recon"%(rprdir))
            

            # Repair Simplification
            cmd_list.append("python src/python/b_repair_reconstitution.py %s/goal_reachability_repair/source.mch %s/result.mch %s %s/recon"%(rprdir,rprdir,conffile,rprdir))
 
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


