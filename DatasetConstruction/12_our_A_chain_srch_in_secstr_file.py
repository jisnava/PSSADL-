#Here we are keeping primary and secondary of the protein files under consideration
import re
import re
with open('11_GAP_WITH_C_NEW_LINE_REMOVED.txt','r') as f:
   lines_f = f.readlines()
   #print(len(lines_f))

if not lines_f:
    sys.exit("Could not read 9_A_CHAIN_SECSTR.txt")

with open('Pisces_40ID_A_CHAIN_ID_only.txt','r') as f1,open('11_GAP_WITH_C_NEW_LINE_REMOVED.txt','r') as f:
	
	#print(len(f1.readline()))
	for needle in (line.strip() for line in f1):
		i=0
		#print(needle)
		for i in range(len(lines_f)):
			#print(lines_f[i][1:5])
			wanted=str(lines_f[i][1:5])
			if needle in wanted:
				print(lines_f[i]+lines_f[i+1]+lines_f[i+2])
				break
				#i=i+3
			#else: 
				#print("Not found")

	#print("Written")