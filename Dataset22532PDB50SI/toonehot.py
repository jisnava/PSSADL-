#This code is for converting the files to standard lengths 500
#Each file is of 500X23 format
#Amino acid names are onehot encoded 

import os
import re
import math
import glob
from pathlib import Path
#import amino
single_aa_id={'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11,'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20}
secondary_class={'G':0, 'H':0, 'I':0, 'E':1, 'B':1, 'T':2, 'S':2, 'C':2}
directory_path1='/home/jisna1729/Assignment/Benchmarking_ASSIGNMENT/Check_set_Roosafeed/JUNE21/'
#directory_path2='/home/jisna/Jisna/STRUCTURE_ASSIGNMENT/A_chain_result/'
x=0
count=0
#model =[]
#letter = [int(0) for _ in range(20)]
#class_ = [int(0) for _ in range(8)]
#letter[val_] = 1
f= open('/home/jisna1729/Assignment/Benchmarking_ASSIGNMENT/Check_set_Roosafeed/JUNE21/check_new_file_outNEW.txt',"r")
#newfile = open('/home/jisna/Jisna/3_class_structure_assignment/18_new_file_r-1_combined_single_line.txt',"a+")
#f1= open('/home/jisna/Jisna/3_class_structure_assignment/3_class_output_9021.txt',"r")
#out_file=open('3_class_input_9021.txt',"a+")
lines=f.readlines()
#lines1=f1.readlines()
#print(len(lines))
#print(len(lines1))
i=0
j=0
while i<len(lines):
	#j=0
	
	val=list(lines[i].split())
	#print(len(val))
	if "END" not in lines[i]:
		
		sec=lines[i].split("\n")[0]
		single_class=secondary_class[sec]
		#print(single_class)
		class_ = [int(0) for _ in range(3)]
		class_[single_class] = 1
		
		print(*class_)
		
		i=i+1
		
	elif "END" in lines[i]:
		
		i=i+1
		print("END")
	else:

		i=i+1

		
print('Written')

		