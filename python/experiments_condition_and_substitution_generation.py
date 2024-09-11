import os
import sys
import time

cmd_list = []

datadir = "Cond_And_Subs_Generation_Experiments/datasets"
resdir = "Cond_And_Subs_Generation_Experiments/results"

FL = os.listdir(datadir)
FL.sort()

for fid in FL:
    if fid == "config":
        continue
    
    sdir = resdir + "/" + fid
    cmd = "mkdir " + sdir
    os.system(cmd)

    for nx in ["25"]:

        ddir = datadir + "/%s/%s/"%(fid,nx)
        sfile = ddir + "/state-transitions.data"
        vlfile = ddir + "/state-variables.data"
        conffile = datadir + "/config"

        wdir = resdir + "/tmp"
        cmd = "rm -r " + wdir
        os.system(cmd)
        cmd = "mkdir " + wdir
        os.system(cmd)

        
        DL = os.listdir(ddir)
        DL.sort()
        
        for tf in DL:
            if not("state-transitions-" in tf):
                continue
            if not(".data" in tf):
                continue
            ope = tf.replace("state-transitions-","").replace(".data","")
    
            tfile = ddir + "/" + tf
            cmd = "python src/python/condition_and_substitution_generation.py %s %s %s %s %s"%(tfile,sfile,vlfile,conffile,wdir)
            os.system(cmd)

            cmd = "cp %s/result.txt %s/result-%s-%s.txt"%(wdir,sdir,ope,nx)
            os.system(cmd)
            cmd = "cp %s/summary.txt %s/summary-%s-%s.txt"%(wdir,sdir,ope,nx)
            os.system(cmd)

            time.sleep(5)

