# 17CS30020
# K. Snehal Reddy 
# 2 
# python 17CS30020_2.py

import pandas as pd 
import numpy as np 

#making the dataset pandas compatible
full = []
with open('data2_19.csv', 'r') as file:
    data = file.readlines()

for i in range(len(data)):
    data[i] = data[i].replace("\"","")
    full.append(data[i].strip('\n').split(","))

data = pd.DataFrame(data=full[1:], columns=full[0])
data = data.astype(int)

full_test = []
with open('test2_19.csv', 'r') as file:
	data_1 = file.readlines()

for i in range(len(data_1)):
    data_1[i] = data_1[i].replace("\"","")
    full_test.append(data_1[i].strip('\n').split(","))

test_data = pd.DataFrame(data=full_test[1:], columns=full_test[0])
test_data = test_data.astype(int)


########################################

#training

# print("training !")
alpha = 1 # laplace_smoothing_constant
	
class_priors = {}
cond_prob = {}

for cl in data.D.unique():
	data_c = data[data['D']==cl]
	class_priors[cl] = (data_c.shape[0] / data.shape[0])

	cond_attr = []
	for attr in data_c.columns[1:]:
		cnts = data_c[attr].value_counts()

		for i in range(1, 6):
			if i not in cnts.keys():
				cnts[i] = 0

		# laplacian smoothing
		prob = (cnts.sort_index()+alpha) / (data_c.shape[0]+alpha*len(cnts))
		cond_attr.append(prob)

	cond_prob[cl]= (cond_attr)


########################################

#testing

# print("testing !")

pred_prob = np.ones((test_data.shape[0], len(test_data.D.unique())))
for index, row in test_data.iterrows():
	for cl in test_data.D.unique():	
		data_c = data[data['D']==cl]
		for i, dat in enumerate(row[1:]):			
			pred_prob[index][cl] *= ( cond_prob[cl][i][dat])
		pred_prob[index][cl]*=class_priors[cl]
			# print(pred_prob[index][cl])

pred_label = np.argmax(pred_prob, axis=1)
# print(pred_label)
correct = 0
for i in range(len(pred_label)):
	if(pred_label[i] == test_data['D'][i]):
		correct+=1

print("predicted labels - ")
print(pred_label)
print("Test accuracy - "+str(correct*100/len(pred_label)))
