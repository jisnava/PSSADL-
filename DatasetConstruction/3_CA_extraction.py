#Extracts only CA ATOM RECORD line from the protein 'pdb' files
import os
import math
import glob
from pathlib import Path
#import amino
directory_path1='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB/'
directory_path2='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA/'
i=0
for file in os.listdir(directory_path1):
	print(file)
	#filename = os.fsdecode(file)
	f= open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB/'+file,"r")
	lines=f.readlines()
	L=[]
	newfile = open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA/' + file, "a+")
	for j in (lines):
		#currentline=j.split()
		try:
			
			if ((str(j[0:4])=='ATOM') and (str(j[13:15]) =='CA')):
				L.append(j)
				#print(j)
				newfile.writelines(''.join(j))
		except:
			print("do nothing")
	
print("Written")

