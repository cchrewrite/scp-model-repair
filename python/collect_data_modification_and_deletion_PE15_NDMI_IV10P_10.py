import os
import sys
import time

cmd_list = []

per_faults = 5

coldir = "TOSEM_Experiments/result_PE15_collection"
os.system("mkdir %s"%coldir)
logfile = coldir + "/COLLECTION_LOG_PE15_NDMI_IV%dP_10"%per_faults
logtxt = []

for sd in xrange(1001,1015):

    datadir = "TOSEM_Experiments/PE15_NDMI_IV%dP_10"%per_faults
    resdir = "TOSEM_Experiments/result_PE15_NDMI_IV%dP_10"%per_faults
    
    FL = os.listdir(datadir + "/material/")
    FL.sort()

    for fn in FL:
        

        if fn[len(fn)-4:len(fn)] != ".mch": continue

        mch_id = fn[0:len(fn)-4]

        for sem_mdl_type in ["SKCART","Random","BNBayes","LR","SVM"]: #["SKCART","Silas","BNBayes","LR","SVM","MLP"]:
        #for sem_mdl_type in ["Random","BNBayes","LR","SVM"]:

            
            x = "y"
            while x != "y" and x != "n":
                x = raw_input("Collecting Results of %s --- Seed = %d --- Percentage of Faulty State Transitions is %d/100 --- Semantic Model is %s? (y/n): "%(fn,sd,per_faults,sem_mdl_type))
            if x != "y": continue

            original_mchfile = datadir + "/material/" + fn
            mdlname = "%s_NDMI_IV%dP_SD%d"%(mch_id,per_faults,sd)

            mdldir = datadir + "/dataset/" + mdlname

            conffile = datadir + "/config/" + sem_mdl_type + "_config"

            rprdir = resdir + "/repaired_%s_%s"%(sem_mdl_type,mdlname)

            resprefix = coldir + "/%s_%s"%(sem_mdl_type,mdlname)

            if os.path.exists(rprdir) == False:
                loginfo = "Result directory does not exist: %s"%rprdir
                print loginfo
                logtxt.append(loginfo)
                continue
            else:
                loginfo = "Result directory exists: %s"%rprdir
                print loginfo
                logtxt.append(loginfo)

            cmd_list = []

            # collecting results of source machines
            if sem_mdl_type == "SKCART":

                seval = rprdir + "/source.eval"
                if os.path.exists(seval) == False:
                    loginfo = "Result does not exist: %s"%seval
                    print loginfo
                    logtxt.append(loginfo)
                else:
                    tf = resprefix.replace("SKCART","SOURCE") + ".eval"
                    cmd_list.append("cp %s %s"%(seval,tf))
                    loginfo = "Result exists: %s"%seval
                    print loginfo
                    logtxt.append(loginfo)

            # collecting reconstitution results
            if sem_mdl_type == "SKCART" and per_faults == 1:
                recon = rprdir + "/recon/RESULT"
                if os.path.exists(recon) == False:
                    loginfo = "Result does not exist: %s"%recon
                    print loginfo
                    logtxt.append(loginfo)
                else:
                    tf = resprefix + ".recon"
                    cmd_list.append("cp %s %s"%(recon,tf))
                    loginfo = "Result exists: %s"%recon
                    print loginfo
                    logtxt.append(loginfo)

            # collecting results of repaired machines

            if True:
                reval = rprdir + "/result.eval"
                if os.path.exists(reval) == False:
                    loginfo = "Result does not exist: %s"%reval
                    print loginfo
                    logtxt.append(loginfo)
                else:
                    tf = resprefix + ".eval"
                    cmd_list.append("cp %s %s"%(reval,tf))
                    loginfo = "Result exists: %s"%reval
                    print loginfo
                    logtxt.append(loginfo)

                rsum = rprdir + "/summary"
                if os.path.exists(rsum) == False:
                    loginfo = "Result does not exist: %s"%rsum
                    print loginfo
                    logtxt.append(loginfo)
                else:
                    tf = resprefix + ".summary"
                    cmd_list.append("cp %s %s"%(rsum,tf))
                    loginfo = "Result exists: %s"%rsum
                    print loginfo
                    logtxt.append(loginfo)





#TOSEM_Experiments/result_PE15_NDMI_IV5P_10/repaired_SKCART_ADD4_NDMI_IV5P_SD1001/result.eval            
#TOSEM_Experiments/result_PE15_NDMI_IV1P_10/repaired_SKCART_ADD4_NDMI_IV1P_SD1001/recon/RESULT

            for x in cmd_list:
                os.system(x)
            cmd_list = []
            

logtxt.sort()
logf = open(logfile,"w")
for x in logtxt:
    logf.write(x)
    logf.write("\n")
logf.close()

