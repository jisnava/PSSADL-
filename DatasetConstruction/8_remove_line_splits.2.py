#Program to remove line gaps in ss.txt
# So one primary sequence in one line, its secondary is another line and so on
import os
import re
import math
import glob

with open('7_a_secondary_extraction.txt','r') as f:
    haystack = f.read()

if not haystack:
    sys.exit("Could not read haystack data")

#with open('ss.txt','r') as f1:
i=0
s=0
always_print=False
now_print = False
needle='[0-9A-Z]:A:sequence'
#needle2= '^\d'
needle1='[0-9A-Z]:A:secstr'
needle2='^[0-9A-Z]{4}\n'
#for line in f1:
	#needle = line.strip()
	#print(needle)
	#print("\n")
for line1 in haystack.splitlines():
	

	if (re.search(needle2,line1)):
		always_print = False
		now_print= False
		#print(line1)
	elif (re.search(needle,line1)):
		
		s=0
		always_print= True
		now_print = False
		#print(line1, end='')
		print("\n")
		print(line1)
		
	elif '>' not in line1 and always_print:
		print(line1, end='')
		#s=s+len(line1)
		#print(s)
		#f2.write(line1)
	elif (re.search(needle1,line1)):
		now_print= True
		always_print = False
		print("\n")
		#print(line1)
	elif '>' not in line1 and now_print and (not(re.search(needle2,line1))):
		if (not(re.search(needle2,line1))):
			print(line1, end='')
		else:
			continue
	elif '>' not in line1 and (re.search(needle2,line1)):
		always_print = False
		now_print= False
		#print("\n")


