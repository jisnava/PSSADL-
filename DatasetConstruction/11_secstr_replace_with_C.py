#Program to fill the gaps in secondary structure with Unknown type:C
# Final done on March 17 2020
import os
import re
import math
import glob

with open('9_new_line_removed_from_8.txt','r') as f:
    haystack = f.read()

if not haystack:
    sys.exit("Could not read haystack data")

#with open('ss.txt','r') as f1:
i=0
s=0
always_print=False
now_print = False
needle='[0-9A-Z]:A:sequence'

needle1='[0-9A-Z]:A:secstr'
#for line in f1:
	#needle = line.strip()
	#print(needle)
	#print("\n")

for line1 in haystack.splitlines():
	print("\n")
	x=range(len(line1))
	for i in x:
		if(line1[i]!=' '):
			print(line1[i],end='')
		else:
			print("C",end='')
print("\n")

