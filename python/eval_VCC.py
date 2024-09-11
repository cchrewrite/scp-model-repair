import os
import sys
import time

cmd_list = []


for sd in xrange(1001,1015):

    per_faults = 10

    datadir = "EvaluationCriteria/Cruise_Eval/dataset"
    resdir = "EvaluationCriteria/Cruise_Eval/result"
    FL = os.listdir(datadir)
    FL.sort()

    os.system("mkdir %s"%(resdir))

    for fn in FL:
        
        if fn[len(fn)-4:len(fn)] != ".mch": continue

        mch_id = fn[0:len(fn)-4]


        for sem_mdl_type in ["SKCART"]: #["SKCART","Silas","BNBayes","LR","SVM","MLP"]:

            x = "a"
            while x != "y" and x != "n":
                x = raw_input("Evaluating %s --- Seed = %d --- Percentage of Missing States is %d/100? (y/n): "%(fn,sd,per_faults))
            if x != "y": continue

            mchfile = datadir + "/" + fn

            conffile = datadir + "/config"
            evalfile = datadir + "/eval.data"

            cmd_list = []
      
            evaldir = resdir + "/eval_%s"%mch_id
            #cmd_list.append("rm -r %s"%(evaldir))
            cmd_list.append("mkdir %s"%(evaldir))

            # Quality Evaluation - Repaired Model
            cmd_list.append("python src/python/b_model_evaluation.py %s %s %s %s"%(mchfile,evalfile,conffile,evaldir))
            #cmd_list.append("cp %s/repaired_model/RESULT %s/result.eval"%(evaldir,rprdir))

            # Quality Comparison
            #cmd_list.append("python src/python/b_model_evaluation_comparison.py %s/source.eval %s/result.eval %s/comparison.eval"%(rprdir,rprdir,rprdir))


            for x in cmd_list:
                os.system(x)
            cmd_list = []
            time.sleep(3)


