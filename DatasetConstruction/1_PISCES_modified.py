#This code will extract only PDB ID's needed for further processing(From PICSES output)

with open('PISCES_1.txt','r') as f, open('PISCES_11_40ID_ID_only.txt', 'w') as f2:
	lines1=f.readlines()
	for j in (lines1):
		currentline=j.split(' ')
		
		if j[4:5] =='A':
			f2.write(j[0:4]+'\n')
f2.close()
