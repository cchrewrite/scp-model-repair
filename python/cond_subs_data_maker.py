import os
import sys
import time

cmd_list = []

datadir = sys.argv[1]
resdir = sys.argv[2]

FL = os.listdir(datadir)
FL.sort()

for fn in FL:
    if fn[len(fn)-4:len(fn)] != ".mch":
        continue
    fid = fn[0:len(fn)-4]
    
    sdir = resdir + "/" + fid
    cmd = "mkdir " + sdir
    os.system(cmd)

    mchfile = datadir + "/" + fn
    
    cmd = "python src/python/make_datasets_of_condition_and_substitution_generation.py %s %s"%(mchfile,sdir)
    os.system(cmd)

    time.sleep(5)

