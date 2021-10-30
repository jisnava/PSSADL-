#Extracts only CA ATOM line from 'pdb' file
#Do extract details of only first chain(any chain name)
import os
import math
import glob
from pathlib import Path
#import amino
directory_path1='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA/'
directory_path2='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_A/'
i=0
for file in os.listdir(directory_path1):
	print(file)
	#filename = os.fsdecode(file)
	f= open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA/'+file,"r")
	lines=f.readlines()
	L=[]
	K=[]
	try:
		#currentline=lines[0].split()
		#print(lines[0])
		#aa_name=currentline[3]
		chain_name_old=lines[0][21:22]
		
		res_num_old=lines[0][22:26]
	except:
		print("do nothing")
	newfile1 = open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_A/' + file + "".join("_") +  "".join(chain_name_old), "a+")
	
	for j in (lines):
		
		try:
			
			chain_name=j[21:22]
			#print(chain_name)
			res_num=j[22:26]
			if ( (chain_name==chain_name_old) and (res_num >= res_num_old)):
				#L.append(j)
				#print(j)
				#if (str(j[16:17])=='B' or str(j[16:17])=='C'): However user can avoid specific chain names if needed
					#continue
				#else:
				newfile1.writelines(''.join(j[17:20]+' '+j[30:38]+' '+j[38:46]+' '+j[46:54]+'\n'))
				res_num_old=res_num
				chain_name_old=chain_name
		except:
			print("do nothing")
	
print("Written")

