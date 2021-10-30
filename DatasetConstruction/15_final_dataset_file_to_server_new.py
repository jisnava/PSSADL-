#Final dataset construction program
# Takes each protein details(single letter primary and coordinate details) and aligned information.
# And assigns secondary structure details to the tertiary coordinates from the aligned information
import os
import re
import math
import glob
from pathlib import Path
#import amino
directory_path1='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_T_CA_COORD/'
#directory_path2='/home/jisna/Jisna/STRUCTURE_ASSIGNMENT/A_chain_result/'
x=0
string_1=''
string_2=''
string_3=''
lines=''
for file in os.listdir(directory_path1):
	#print(file)
	id=file.split('.')[0]
	print(id)
	f= open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_T_CA_COORD/'+file,"r")
	lines=f.readlines()
	#z=len(lines)
	L=[]
	string_1=lines[2]


	with open('18_17_aligned_with_three_sequence.txt','r') as f:
		line1 = f.readlines()
	fl=open('18_new_file.txt','a+')
	
	
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
    	
	for i in range(x):
		#print("Hello")
		if (re.search(needle,line1[i])):
			try:
				string_2=line1[i+2]
				string_3=line1[i+3]
			except:
				continue
	#print(string_2,end='')
	#print(string_3,end='')
	len_string_2=len(string_2)
	z=min(len(string_3), len(string_1), len(string_2))+3
	m=z-3
	if (m > 500):
		continue
	fl.writelines("\n<ID>\n")
	fl.writelines(''.join(id))
	fl.writelines("\n<PRIMARY>\n")
	fl.writelines(''.join(string_1))
	fl.writelines("<PRIMARY_DSSP>\n")
	fl.writelines(''.join(string_2))
	fl.writelines("<SECONDARY_DSSP>\n")	
	fl.writelines(''.join(string_3))	
	
	#print(len(string_2))
	fl.writelines("<LEN>\n")
	fl.writelines(''.join(str(m)))
	fl=open('18_new_file.txt','a+')
	fin=open('18_new_file_in.txt','a+')
	fout=open('18_new_file_out.txt','a+')
	fl.writelines("\n<RESIDUE_NAME TERTIARY_COORDINATES \\n SECONDARY_STRUCTURE >\n")	
	for k in range(4,z):
		AA_name=lines[k][0]
		#print(AA_name)
		#print(j,len_string_2)
		##while(j<len_string_2):
		r=min(len(string_2),len(string_3))
		while(j < r-1):
			if (AA_name==string_2[j]):
				try:
					model_in=(lines[k]+"\n")
					fin.writelines(''.join(model_in))
					model_out=(string_3[j]+"\n")
					fout.writelines(''.join(model_out))
					model=(lines[k],string_3[j]+"\n")
					fl.writelines(''.join(model))
					#print(lines[k],string_3[j])
					#print(string_3[j])
					j=j+1
					break
				except:
					print("extra primaries")
			else:
				j=j+1
	fl.writelines("<END>")
	fin.writelines("END\n")
	fout.writelines("END\n")
	#break

    		
#print(L)	
	


