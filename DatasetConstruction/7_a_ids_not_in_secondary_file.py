#Prints those PDB ID's whose secondary details are not present in 6_A_CHAIN_SECSTR.txt( A chain details extracted fron ss.txt)

with open('6_A_CHAIN_SECSTR.txt','r') as f:
    haystack = f.read()

if not haystack:
    sys.exit("Could not read haystack data")

with open('1_Pisces_40ID_A_CHAIN_ID_only.txt','r') as f1:
	i=0
	j=0
	always_print=False
	for needle in (line.strip() for line in f1):
		if needle not in haystack:
			print(needle)





