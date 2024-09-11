import sys
import Bmch
import os
import time
import Bgenlib
import random
import RepSimpLib
import SemLearnLib
from nnet.nnetlib import *
from Cartlib import *
from NBayes import *
from SKCART import *
import numpy
import logging
import pickle
import time

# ==================================================

# Making a dataset for the condition and substitution generation algorithm.
# Usage: python [this script] [source abstract machine] [semantics model] [result folder]

# =================================================

start_time = time.time()

if len(sys.argv) != 3:
    print "Error: The number of input parameters should be 2."
    print "Usage: python [this script] [source abstract machine] [result folder]"
    exit(1)



M = sys.argv[1]
resdir = sys.argv[2]

cmd = "mkdir %s"%resdir
os.system(cmd)

print "Source Abstract Machine:", M
print "Result Folder:", resdir

print "Computing the state space..."

MR = M

fn = resdir + "/MR_pp.mch"
oscmd = "./../ProB/probcli -pp %s %s"%(fn,MR)
os.system(oscmd)
MR = fn

with open(MR) as mchf:
    mch = mchf.readlines()
mch = [x.strip() for x in mch]

D = resdir + "/D.txt"
max_initialisations = 1000000
max_operations = 1000000
bscope = Bmch.generate_training_set_condition(mch)
oscmd = "./../ProB/probcli %s -model_check -df -p MAX_DISPLAY_SET -1 -p MAX_INITIALISATIONS %d -p MAX_OPERATIONS %d -nodead -scope \"%s\" -spdot %s -c"%(MR,max_initialisations,max_operations,bscope,D)
os.system(oscmd)

sg = Bgenlib.BStateGraphForNN()
sg.ReadStateGraph(D)
TL = sg.GetTransList()
VList = sg.GetVbleList()
    
print "Writing data to files."

# prop is the percentage of data/.
for prop in [25, 50, 100]:

    sdir = resdir + "/" + str(prop) + "/"
    cmd = "mkdir " + sdir
    os.system(cmd)

    SL = TL + []
    random.shuffle(SL)
    SL = SL[0:int(len(SL) * prop * 1.0 / 100)]

    sfile = sdir + "/" + "state-variables.data"
    f = open(sfile,"w")
    f.write(str(VList))
    f.write("\n")
    f.close()

    SL.sort(key = lambda x:x[1])
    sfile = sdir + "/" + "state-transitions.data"
    f = open(sfile,"w")
    for x in SL:
        f.write(str(x))
        f.write("\n")
    f.close()

    FL = []
    for x in SL:
        if not(x[1] in FL):
            FL.append(x[1])

    for F in FL:
        sfile = sdir + "/state-transitions-" + F + ".data"
        f = open(sfile,"w")
        for x in SL:
            if x[1] != F:
                continue
            f.write(str(x))
            f.write("\n")
        f.close()

print "Done!"
