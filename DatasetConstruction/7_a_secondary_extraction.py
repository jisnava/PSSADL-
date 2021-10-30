#Program to extract primary and secondary data from ss.txt(secondary structure information from PDB) of a given protein Ids 
#Of the proteins we have choosen as non redundant, we have those protein id's with A chain only


with open('6_A_CHAIN_SECSTR.txt','r') as f:
    haystack = f.read()

if not haystack:
    sys.exit("Could not read haystack data")

with open('1_Pisces_40ID_A_CHAIN_ID_only.txt','r') as f1:
	i=0
	j=0
	always_print=False
	for needle in (line.strip() for line in f1):
	#for line in f1:
		#needle = line.strip()
		print(needle)
		#print("\n")
		for line1 in haystack.splitlines():
			if needle in line1:
				#print(i,'found')
				i=i+1
				always_print= True
				#print(line1, end='')
				print(line1)
				#print("\n")
				#f2.write(line1)
			elif '>' not in line1 and always_print:
				print(line1)
				#print(i)
				j=j+1
				#f2.write(line1)
			elif '>' in line1:
				always_print = False
			





"""
print("Hai")
with open('ss.txt','r') as f, open('mylist_50_1442','r') as f1, open('secondary_struct_out.txt','w') as f2:
	always_print=False
	f11=f1.readlines()
	for line1 in f11:
		print(line1)
		always_print=False
		#pid=str(line1)
		for line in f:
			if  '1KQ6' in line:
				always_print= True
				print(line, end='')
				f2.write(line)
			elif '>' not in line and always_print:
				print(line, end='')
				f2.write(line)
			else:
				always_print = False
		
			

print("Hai")
with open('ss.txt','r') as f, open('mylist_50_1442','r') as f1, open('secondary_struct_out.txt','w') as f2:
	always_print=False
	f11=f1.readlines()
	for line1 in f11:
		print(line1)
		always_print=False
		#pid=str(line1)
		for line in f:
			if  '1KQ6' in line:
				always_print= True
				print(line, end='')
				f2.write(line)
			elif '>' not in line and always_print:
				print(line, end='')
				f2.write(line)
			else:
				always_print = False
		
"""		
		
