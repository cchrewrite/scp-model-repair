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

# The condition and substitution generation algorithm.
# Usage: python [this script] [covered state transitions] [state space] [variable list] [configuration file] [result folder]

# =================================================

start_time = time.time()

if len(sys.argv) != 6:
    print "Error: The number of input parameters should be 5."
    print "Usage: python [this script] [covered state transitions] [state space] [variable list] [configuration file] [result folder]"
    exit(1)



tfile = sys.argv[1]
sfile = sys.argv[2]
vlfile = sys.argv[3]
conffile = sys.argv[4]
resdir = sys.argv[5]

print "Covered State Transitions:", tfile
print "State Space:", sfile
print "Variable List:", vlfile
print "Configuration File:", conffile
print "Result Folder:", resdir

print "Data Preparation..."

cmd = "mkdir " + resdir
os.system(cmd)

sdir = resdir + "/generation"
cmd = "mkdir " + sdir
os.system(cmd)

tf = sdir + "/covered-state-transitions.data"
cmd = "cp %s %s"%(tfile,tf)
os.system(cmd)
tf = open(tf,"r")
TL = []
for x in tf.readlines():
    y = x.replace("\n","")
    y = eval(y)
    TL.append(y)
tf.close()

sf = sdir + "/state-space.data"
cmd = "cp %s %s"%(sfile,sf)
os.system(cmd)
sf = open(sf,"r")
SL = []
for x in sf.readlines():
    y = x.replace("\n","")
    y = eval(y)
    SL.append(y)
sf.close()
for x in SL: print x

vlf = sdir + "/variable-list.data"
cmd = "cp %s %s"%(vlfile,vlf)
os.system(cmd)
vlf = open(vlf,"r")
VL = []
x = vlf.readlines()[0]
y = x.replace("\n","")
VL = eval(y)
vlf.close()
for x in VL: print x

SRF = RepSimpLib.CFGReachabilityRepairSimplification(TL,SL,VL,conffile,sdir)

resf = resdir + "/result.txt"
resf = open(resf,"w")

for x in SRF:
    resf.write(x[0])
    resf.write("\n")
    for y in x[1]:
        z = str(y).replace("/* CFG substitution */","")
        resf.write(z)
        resf.write("\n")

    print "==========="
    print x[0]
    for y in x[1]:
        print y
    
resf.close()

resf = resdir + "/result.txt"
resf = open(resf,"r")
CSL = []
rflag = False
for x in resf.readlines():
    if rflag == False:
        rflag = True
        continue
    y = x.replace("\n","")
    y = eval(y)
    CSL.append(y)
resf.close()

# NCL --- number of condition literals
NCL = 0
# NAS --- number of atomic substitutions
NAS = 0
for x in CSL:
    CL = RepSimpLib.count_number_of_literals_in_DNF(x[0])
    print CL
    NCL = NCL + CL
    SL = RepSimpLib.count_number_of_atomic_substitutions(x[1])
    print SL
    NAS = NAS + SL

# N0 --- size of original transitions
N0 = len(TL)*len(VL)*2

# NCSC --- number of condition-substitution constructs
NCSC = len(CSL)

print NCL,NAS,N0

end_time = time.time()
elapsed_time = end_time - start_time

print "Number of condition-substitution constructs is %d."%NCSC
print "Number of condition literals is %d."%NCL
print "Number of atomic substitution is %d."%NAS
print "Size of original transitions is %d."%N0
print "Elapsed Time (s): %f."%elapsed_time

fn = resdir + "/summary.txt"
logf = open(fn,"w")
logf.write("Number of condition-substitution constructs is %d.\n"%NCSC)
logf.write("Number of condition literals is %d.\n"%NCL)
logf.write("Number of atomic substitution is %d.\n"%NAS)
logf.write("Size of original transitions is %d.\n"%N0)
logf.write("Elapsed Time (s): %f.\n"%(elapsed_time))
logf.close()

