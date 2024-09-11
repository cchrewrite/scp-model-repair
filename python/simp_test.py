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

 
f = open("TL.tmp","r")
TL = eval(f.readlines()[0])
f.close()
f = open("R.tmp","r")
R = eval(f.readlines()[0])
f.close()
f = open("VList.tmp","r")
VList = eval(f.readlines()[0])
f.close()

wdir = "TOSEM_Experiments/model_repair_results/mdl/repaired_model/repair_simplification/"
conffile = "TOSEM_Experiments/model_repair_results/mdl/repaired_model/config"
epoch = 0
epid = "ep%d"%epoch
VPF = ["_pre_ep0",""]

if Bmch.read_config(conffile,"apply_CFG_simplification","bool") == True:
    RepSimpLib.CFGModificationSimplification(R,TL,VList,VPF,conffile,wdir)
else:
    ppp



