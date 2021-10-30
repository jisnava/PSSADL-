#Extracts only CA ATOM RECORD line from 'pdb' file
#This is an extra file, user can run only 4_CA_extraction.py. If any spacing issues comes they can try this as well(otherwise no need to run)
import os
import math
import glob
from pathlib import Path
#import amino
directory_path1='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB/'
directory_path2='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_INDEX_BASED/'
i=0
for file in os.listdir(directory_path1):
	print(file)
	#filename = os.fsdecode(file)
	f= open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB/'+file,"r")
	lines=f.readlines()
	L=[]
	for j in (lines):
		currentline=j.split()
		try:
			aa_name=j[0:4]
			atom_name=currentline[2]
			#print(j[0:4]+j[12:16]+aa_name+atom_name)
			if ((aa_name=="ATOM") and (atom_name =="CA")):
			#L.append(j)
				newfile = open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_INDEX_BASED/' + file, "a+")
				newfile.writelines(''.join(j))
				#print(j)
			else:
				continue
		except:
			print("do nothing")

		
	
print("Written")

