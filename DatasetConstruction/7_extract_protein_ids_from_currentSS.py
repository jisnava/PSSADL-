
#This is needed because some of the proteins in our PDB folder doesen't have secondary structure in ss.txt, but it name appears in new ss.txt.
#(This creates problem and this can be avoided in program 9(secondary extraction)). 
#Now we are extracting id's from new ss.txt and from our A folder and find the difference and remove those from our folder-everything) 

import re
with open('7_a_secondary_extraction.txt','r') as f1:
	lines_f = f1.readlines()
	i=0
	for i in range(len(lines_f)):
		#print(lines_f[i][1:5])
		wanted=str(lines_f[i][0:5])
		if '>' in wanted: #since the format is like this
				print(lines_f[i][1:5])
				#break
				#i=i+3
			#else: 
				#print("Not found")

	#print("Written")
