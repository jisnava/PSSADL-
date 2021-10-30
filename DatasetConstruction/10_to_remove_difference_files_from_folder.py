#Calculate difference between the files: protein_ids_A.txt and file with those ids extractes from current secondary structure file(matching_ids.txt#output of extract protein ids.txt)
#Remove those protein files whose secondary structure details are not there in ss.txt
import os
import math
import glob
import shutil
from pathlib import Path
#import amino
directory_path1='/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/18_PDB_CA_COORD/'
#directory_path2='/home/jisna/Jisna/lockdown_data_March2020/trial_str_ASSIGN//7_PDB_Result_chainwise_A/'
i=0
for file in os.listdir(directory_path1):
    print(file)
    f_name=file.split('.')
    with open('14_difference_A_chain_IDs.txt','r') as f1: 
        for needle in (line.strip() for line in f1):
            if ((f_name[0] ==needle)):
                os.remove(directory_path1+file)
            else:
                print("nothing")
	
			
print("Written")

