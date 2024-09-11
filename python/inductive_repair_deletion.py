
import Bgenlib
import sys
import RepSimpLib
import os


if len(sys.argv) != 5:
    print "Error: The number of input parameters should be 4."
    exit(1)

print "Positive Example File: ", sys.argv[1]
print "Negative Example File: ", sys.argv[2]
print "Variable List: ",sys.argv[3]
print "Result Folder: ", sys.argv[4]

pfn = sys.argv[1]
nfn = sys.argv[2]
vlfn = sys.argv[3]
resdir = sys.argv[4]

pf = open(pfn,"r")
nf = open(nfn,"r")

pexp = []
nexp = []
for x in pf.readlines():
    pexp.append(eval(x)[0])
for x in nf.readlines():
    nexp.append(eval(x)[0])

vlf = open(vlfn,"r")
vble_list = []
for x in vlf.readlines():
    u = x.replace("\n","")
    vble_list.append(u)

if pexp == []:
    cfn = resdir + "/result.cond"
    cf = open(cfn,"w")
    cf.write("No Repair.\n")
    cf.close()
    exit()
 
def boolconv(P):
    for i in xrange(len(P)):
        for j in xrange(len(P[i])):
            if P[i][j] == 'T': P[i][j] = 'TRUE'
            elif P[i][j] == 'F': P[i][j] = 'FALSE'
            else:
                print "ERROR!!!"
                ppp
    return P

SType = RepSimpLib.get_types_for_progol(pexp + nexp)

#print pexp
#print nexp
#pppp

for i in xrange(len(pexp)):
    pexp[i] = RepSimpLib.rewrite_state_from_python_to_prolog(pexp[i])
for i in xrange(len(nexp)):
    nexp[i] = RepSimpLib.rewrite_state_from_python_to_prolog(nexp[i])


# [[['TV', 'SONY_PS3', 'Microwave', 'Air_Cond', 'Oven'], '5'], 'Remove_Oven', [['TV', 'SONY_PS3', 'Microwave', 'Air_Cond'], '4']]
#SType = sg.GetSetTypeFromTransList(RepSimpLib.convert_pn_examples_to_translist(pexp) + RepSimpLib.convert_pn_examples_to_translist(nexp))

prog = ["% Isolation simplification using Aleph.\n"]
prog_d = ["% Directives.\n"]
prog_b = ["% Background Knowledge Section.\n"]
prog_f = ["% Positive Examples Section.\n"]
prog_n = ["% Negative Examples Section.\n"]
prog_e = ["% End Section.\n"]



prog_b.append(":-begin_bg.\n")
prog_f.append(":-begin_in_pos.\n")
prog_n.append(":-begin_in_neg.\n")

prog.append("% Module loading.\n")
prog.append(":- use_module(library(aleph)).\n\n")
prog.append("% Aleph initialization.\n")
prog.append(":- aleph.\n\n")

S = [
  # i --- layers of new variables.
  ":- set(i,5).\n",
  # clauselength --- the maximum length of clause. Default is 4.
  ":- set(clauselength,10).\n",
  # minpos --- the minimum number of positive examples to be covered by a learnt clause
  # ":- set(minpos,2).\n",
  # splitvars --- variables are independent of each other
  ":- set(splitvars,true).\n",
  #":- set(explore,true).\n :- set(depth,100).\n :- set(splitvars,true).\n",
]
prog_d = prog_d + S

S = RepSimpLib.make_iso_example_decl(pexp[0])
prog_d = prog_d + S

#S = RepSimpLib.make_type_defs()
#prog_b = prog_b + S + [""]
S = RepSimpLib.make_general_rules()
prog_b = prog_b + S + [""]
S = RepSimpLib.make_is_bset_rule(SType,5)
prog_b = prog_b + S + [""]

"""
S = [
  "% Special Rules.",
  "is_bset([bS0,bS1,bS2,bS3,bS4,bS5]).",
  "is_bset([bS0,bS1,bS2,bS3,bS4]).",
  "is_bset([bS0,bS1,bS2,bS3]).",
]
prog_b = prog_b + S + [""]
"""
S = RepSimpLib.make_iso_examples(pexp,"pos")
prog_f = prog_f + S + [""]

S = RepSimpLib.make_iso_examples(nexp,"neg")
prog_n = prog_n + S + [""]



prog_d.append("\n")
prog_b.append(":-end_bg.\n\n")
prog_f.append(":-end_in_pos.\n\n")
prog_n.append(":-end_in_neg.\n\n")
prog_e.append(":-aleph_read_all.\n")
prog_e.append(":-induce.\n")
rfn = resdir + "/result.rule"
prog_e.append(":-aleph:write_rules(\"%s\").\n"%rfn)

prog = prog + prog_d + prog_b + prog_f + prog_n + prog_e


for x in prog: print x

plfn = resdir + "/simp.pl"
f = open(plfn,"w")
for x in prog:
    f.write(x)
    f.write("\n")
f.close()


#cmd = "swipl -g \"[tttr], nl, halt\""
cmd = "swipl -g \"consult(\'%s\'), nl, halt\""%plfn
os.system(cmd)

rl = RepSimpLib.read_aleph_rules(rfn)
for x in rl: print x

cond = ""
#vble_list.append("havefun")
for x in rl:
    y = RepSimpLib.rule_to_cond(x,vble_list,resdir)
    y = "not ( " + y + " )"
    cond = cond + y + " & "

cond = cond[0:len(cond)-3]
print cond

cfn = resdir + "/result.cond"
cf = open(cfn,"w")
cf.write(cond)
cf.write("\n")
cf.close()

"""
ppppppppppppppp

prog = prog + [":- set(h,1000)?\n:- set(r,100000000)?\n"]

S = RepSimpLib.make_example_decl(pexp[0],"iso_cond")
prog = prog + S + [""]

# iso_cond(A,B) :- bset_bg(A,B).
S = RepSimpLib.make_examples(pexp,"iso_cond","pos")
prog = prog + S + [""]

S = RepSimpLib.make_examples(nexp,"iso_cond","neg")
prog = prog + S + [""]


S = RepSimpLib.progol_essential()
prog = prog + S + [""]

S = RepSimpLib.make_all_in_relations(SType,3)
prog = prog + S + [""]

S = RepSimpLib.make_all_bsubst_relations(SType,6)
prog = prog + S + [""]


S = RepSimpLib.make_is_bool_relations()
prog = prog + S + [""]

f = open("tttr.pl","w")
for x in prog:
    f.write(x)
    f.write("\n")
f.close()

cmd = "./../../progol/source/progol tttr.pl"
os.system(cmd)
"""
