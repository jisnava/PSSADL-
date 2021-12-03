#Code to test individual files from the test set using the saved model weights
#imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed,Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from keras.models import load_model
import numpy as np
import time
#start = time.time()

def userformat_to_numpytextformat(pathfirsttext): #Convert to text for numpy format
    f = open(pathfirsttext, 'r')
    t = open('./numpytext_format_sys.txt', 'w')      #output file name
    classes = {'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11,'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20}
    def make_one_hot(n):
        st = ''
        i = 1
        while (i <= 20):
            if(i == n):
                st += '1'
            else:
                st += '0'
            if(i != 20):
                st += ' '
            i += 1
        return st

    def fill_z(n):
        i = 1
        st = ''
        while (i <= n):
            st += make_one_hot(0)
            st += ' 0 0 0 0 0 0'
            if(i != n):
                st += '\n'
            i += 1
        return st

    line_count = 0

    while True:
        line = f.readline()
        #line_end = o.readline()
        if not line:
            break

        if 'END' in line:
            if(line_count < 500):
                t.write(fill_z(500 - line_count))
                t.write('\n')
            line_count = 0
        
        else:
            line_count += 1
            out = line.replace("\n", "").replace("\t", "").split(' ')
            #print(out)
            c = classes[out[0]]
            t.write(make_one_hot(c))
            for x in range(1, len(out)):
                y = str(out[x]).strip()
                if(y != ''):
                    t.write(" " + str(y))
                    #print(y)
            t.write("\n")


def gettext(filepath): 
  txt = open(filepath, encoding='utf-8-sig')
  batches = 1           
  l = 0            
  x = []     
  y = []     
  while True:
    line = txt.readline()
    if (not line):
      break 
    w = line.split(' ')
    for i in w[:20]:
      #print(type(i))
      x.append(int(i))
    for c in w[20:23]:
      x.append(float(c)) 
    l += 1
    if l == 500:
      l = 0
  x = np.array(x)
  x2 = x.reshape(batches, 500, 23)
  x2s = batches * 500 * 23
  x2 = x.reshape(x2s)
  #np.save('/bench_sample.npy', x2)
  return x2
 

def testing_protein(numpyfile, act_len): #Make predictions
    x_test=np.load(numpyfile)
    #-----------------------------------------------------------------------------------------
    #Loading test set (189 proteins) from the mounted Google drive OR you can load this from your current working directory
    #x_test=np.load('/content/drive/MyDrive/CNN_for_STR_ASSIGNMENT_DATA_and_CODES/bench_193x500x23_xyzjune21.npy')
    #y_test=np.load('/content/drive/MyDrive/CNN_for_STR_ASSIGNMENT_DATA_and_CODES/bench_labels_193x500x3_xyzjune21.npy')
    #print("x_test: " + str(x_test.shape))
    #x_test and y_test represents benchmarked set throughout the code(coordinates and labels respectively)
    #----------------------------------------------------------------------------------------
    #Reshape the test set-to match the expected dimensions for first CNN layer
    x_test = x_test.reshape(1, 500, 23, 1)
    #y_test = y_test.reshape(189, 500, 3)
    #-----------------------------------------------------------------------------------------
    #Load the saved model(2DCNN-BLSTM)
    #!pip install keras #if needed
    new_model = load_model('./my_model_full9624Accuracy') #saved model
    #new_model.summary() # To see the model summary
    #resultbenchmarkCNNBLSTM = new_model.evaluate(x_test, y_test) # Accuracy on benchmark set(Test set I)- Evaluation for entire test set(not for individual samples)
    #----------------------------------------------------------------------------------------
    #Evaluation on a single protein
    #print(x_test[0].shape)
    ti = 2 #third protein in test set
    test_protein=x_test[0].reshape(1, 500, 23, 1)
    #print(test_protein.shape)
    preds = new_model.predict(test_protein) #Take a single protein to test and reshape it
    label_index = np.argmax(preds, axis=2)
    labels = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    labels = np.array(labels)
    labels = labels.reshape(3, 3)
    #print(labels)
    act_len=act_len-1
    results=[]
    #print("Mismatch in classes: " + str(x) + "\nLength of sequence: " + str(t))
    #print("Length of sequence: " + str(t))
    #print("Accuracy: " + str(((t-x)/t)*100) + "%")
    for k in range(0,act_len):
      if (label_index[0][k])==0:
        #print("H",end=' ')
        results.append("H")
      if (label_index[0][k])==1:
        #print("E",end=' ')
        results.append("E")
      if (label_index[0][k])==2:
        #print("C",end=' ')
        results.append("C")
    return(results)

userformat_to_numpytextformat('./text_format_sample_1.txt')
fp=open('./text_format_sample_1.txt','r')
lenf=len(fp.readlines())
#print(lenf)
x3=gettext('./numpytext_format_sys.txt')
np.save('./bench_sample_2_sys.npy', x3)  #saving in numpy format
res=testing_protein('./bench_sample_2_sys.npy',lenf)
print("\n")
print("The assignments for the given file are:")
print(' '.join(res))
#end = time.time()
#print({end-start})

