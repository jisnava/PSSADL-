# To remove line gaps 
#This you can simply remove and not necessarily with this code
import re
import os
import math
import glob
import amino
from pathlib import Path
f= open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/8_line_splits_removed_from_7.txt',"r")
newfile = open('/home/jisna/Jisna/18_FULL_LENGTH_DATA_FOR_ASSIGNMENT/9_new_line_removed_from_8.txt', "a+")
lines=f.readlines()
needle='^\n'
for line in lines:
	if (re.search(needle,line)):
		continue
	elif "\r" not in line:
		newfile.writelines(''.join(line))
#print("Written")	

