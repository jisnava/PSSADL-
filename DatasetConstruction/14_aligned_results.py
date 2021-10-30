#Program to print primary of both and aligned result to a single file

import os
import re
import math
import glob
from pathlib import Path
#import amino
directory_path1='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_COORD/'
#directory_path2='/home/jisna/Jisna/STRUCTURE_ASSIGNMENT/A_chain_result/'
x=0
string_1=''
string_2=''
string_3=''
for file in os.listdir(directory_path1):
	#print(file)
	id=file.split('.')[0]
	print(id)
	f= open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_COORD/'+file,"r")
	lines=f.readlines()
	k=len(lines)
	L=[]
	string_1=lines[2]
	#for i in range(0,x):
	print(string_1,end='')
	#print("\n")

	with open('18_16_aligned_results.txt','r') as f, open('12_Our_and_their_A_chain_only_new_line_removed.txt','r') as fl:
		line1 = f.readlines()
		linel = fl.readlines()
	

	if not line1:
	    sys.exit("Could not read haystack data")

	#with open('ss.txt','r') as f1:
	i=0
	j=0
	always_print=True
	now_print = False
	needle=''+id+''
	needle2= '^\d'
	needle1='[0-9A-Z]:A:secstr'
	x=len(line1)
	y=len(linel)
	#print(x)
	for i in range(x):
		#print("Hello")
		if (re.search(needle,line1[i])):
			string_2=line1[i+1]
			for j in range(y):
				if (re.search(needle,linel[j])):
					#print("Hai")
					string_3=linel[j+2]
					break
			break
	
	print(string_2,end='')
	print(string_3)
	

