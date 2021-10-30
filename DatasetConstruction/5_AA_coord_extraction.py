#Program to extract Amino acid name(single letter name) from a pdb fle and to write it along with its coordinates
import os
import math
import glob
import amino
from pathlib import Path
#import amino
directory_path1='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_A/'
directory_path2='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_COORD/'
i=0
mapping=amino.aminomapping()
for file in os.listdir(directory_path1):
	print(file)
	id=file.split('.')[0]
	#filename = os.fsdecode(file)
	f= open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_A/'+file,"r")
	lines=f.readlines()
	L=[]
	newfile = open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_COORD/' + file, "a+")
	newfile.writelines(id)
	newfile.writelines("\n<PRIMARY>\n")
	for j in (lines):
		#currentline=j.split()
		try:
			aa_name=j[0:3]
			single_letter=mapping[aa_name]
			#print(single_letter)
			newfile.writelines(''.join(single_letter))
		except:
			print("None ")
			print(j)
	newfile.writelines("\n<TERTIARY>\n")
	for j in (lines):
		#currentline=j.split()
		try:
			aa_name=j[0:3]
			single_letter=mapping[aa_name]
			#model=(single_letter,"\t", j[4:13],"\t", currentline[7],"\t", currentline[8])
			model=(single_letter,"\t", j[4:13],"\t", j[14:22],"\t", j[23:31])
			newfile.writelines(''.join(model))
			#newfile.writelines("\n")
			#newfile.writelines(''.join(single_letter+'\t'+j[4:13]+' '+j[14:22]+' '+j[23:31]))
			#newfile.writelines("\n")
		except:
			print("again None ")
			print(j)
	id=file.split('.')[0]
	print(id)
	
print("Written")

