#Program to extract primary and secondary sequences of A chains only from ss.txt
# This program extracts A chain details only from ss.txt file, however users can extract all chain details by making a small modification to the codes
#ss.txt file can be downloaded from RCSB-PDB website. It contains primary and secondary(assigned by DSSP) of all proteins in RCSB-PDB
import os
import re
import math
import glob

with open('ss.txt','r') as f:
    haystack = f.read()

if not haystack:
    sys.exit("Could not read haystack data")

#with open('ss.txt','r') as f1:
i=0
always_print=False
needle='[0-9A-Z]:A:'

for line1 in haystack.splitlines():
	if (re.search(needle,line1)):
		#print(i,'found')
		i=i+1
		always_print= True
		#print(line1, end='')
		print(line1)
		#print("\n")
		#f2.write(line1)
	elif '>' not in line1 and always_print:
		print(line1)
		#print(i)
		#f2.write(line1)
	elif '>' in line1:
		always_print = False


