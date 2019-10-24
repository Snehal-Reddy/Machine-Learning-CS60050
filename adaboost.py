import csv
import numpy as np #for array handling
import math
import pprint #for formatted text

def counter(data,att,variable,target_variable):
	den = 0
	num = 0
	# print(len(data[list(data.keys())[-1]]) , att,len(data[att]))
	for i in range(len(data[list(data.keys())[-1]])):
		if(data[att][i] == variable):
			den+=1
			if(data[list(data.keys())[-1]][i] == target_variable):
				num+=1
	return num,den

def attribute_entropy(data,attribute):
	y = np.unique(np.array(data[list(data.keys())[-1]]))
	variables = np.unique(np.array(data[attribute]))
	entropy2 = 0
	non_zero = np.finfo(float).eps
	for variable in variables:
		entropy = 0
		for target_variable in y:
			num,den = counter(data,attribute,variable,target_variable)
			# print("num",num, den)
			fraction = num/(den+non_zero)
			entropy += -fraction*np.log2(fraction+non_zero)
		fraction2 = den/len(data)
		entropy2 += -fraction2*entropy
	# print(attribute, abs(entropy2))
	return abs(entropy2)

def partition(data,node,value):
	# print("before",data.keys(),node,value)
	new = {}
	for i in (list(data.keys())):
		new[i] = []
	output = list(data.keys())[-1] 
	for j in range(len(data[output])):
		if(data[node][j] == value):
			for i in  (list(data.keys())):
				new[i].append(data[i][j])
	# print("after",len(new[list(new.keys())[-1]]))
	del new[node]		
	return new

def Tree(data,attributes,tree = None): 
	gain = []
	output = list(data.keys())[-1]   
	node_entropy = 0
	all_ = np.array(data[output])
	unique, counts = np.unique(all_, return_counts=True)
	for i in range(len(unique)):
		fraction = counts[i]/len(data[output])
		node_entropy += -fraction*np.log2(fraction)

	# print(data.keys(),attributes)
	if(len(attributes)==0):
		clValue,counts = np.unique( np.array( data[list(data.keys())[-1]] ),return_counts=True)	
		_class = clValue[np.argmax(counts)]
		# print("UUUUUU",counts)
		return _class

	# print(attributes)
	for key in attributes:
		gain.append(node_entropy-attribute_entropy(data,key))
	# print(gain)
	node =  list(data.keys())[:-1][np.argmax(gain)]
	# attributes.remove(node)
	vals = np.unique(np.array(data[node]))  

	# print(attributes,node,vals)
	if tree is None:                    
		tree={}
		tree[node] = {} 
	
	for value in vals:	
		prev_clValue,prev_counts = np.unique( np.array( data[list(data.keys())[-1]] ),return_counts=True)		
		prev_class = prev_clValue[np.argmax(prev_counts)]
		# print(value)
		new_data = partition(data,node,value)
		# attributes_new = attributes.copy()
		# attributes_new.remove(node)
		if(len(new_data[list(new_data.keys())[-1]])==0):
			tree[node][value] = prev_class
			continue

		clValue,counts = np.unique(np.array(new_data[list(new_data.keys())[-1]]),return_counts=True)                        
		
		if len(counts)==1:
			tree[node][value] = clValue[0]                                                    
		else:        
			tree[node][value] = Tree(new_data,list(new_data.keys())[:-1]) 
				   
	return tree
  
  
data = {}
header = []
first = 1
with open('data3_19.csv','rt')as f:
	data_ = csv.reader(f)
	for row in (data_):
		if (first):
			for col in (row):
				data[col] = []
				header.append(col)
			first = 0
			continue
		for i,col in enumerate(row):
			data[header[i]].append(col)

attributes = list(data.keys())[:-1]

accuracy = []
n = 1000 #number of sampling data points
probabilities = np.full(len(data[attributes[0]]), 1/len(data[attributes[0]]))
tree_lis = []
alpha_lis = []
num = 0

# three iterations
for iter in range(3):
	draw = np.random.choice(np.arange(len(data[attributes[0]])), n, p=probabilities)
	data_1 = {}
	for i in data.keys():
		data_1[i] = np.take(data[i],draw)
	tree_lis.append(Tree(data_1,attributes))
	# pprint.pprint(tree, indent=4)
	epsilon = 0
	wrong = []
	for i in draw:
		pred = tree_lis[num]
		while(pred!='yes' and pred!='no'):
			pred = pred[list(pred.keys())[0]][ data[list(pred.keys())[0]][i] ] 
		# print(pred, data[list(data.keys())[-1]][i])
		if(pred!=data[list(data.keys())[-1]][i]):
			epsilon+=probabilities[i]
			wrong.append(i)
	# print(epsilon)
	alpha = np.log((1-epsilon)/epsilon)/2
	alpha_lis.append(alpha)
	probabilities = probabilities*np.exp(alpha)
	for i in wrong:
		probabilities[i] = probabilities[i]*np.exp(-2*alpha)

	probabilities = probabilities/np.sum(probabilities)
	num+=1


#############################
#testing

data = {}
for i in header:
	data[i] = []
with open('test3_19.csv','rt')as f:
	data_ = csv.reader(f)
	for row in (data_):
		for i,col in enumerate(row):
			data[header[i]].append(col)

incorrect = 0

#for all test cases
for i in range(len(data[list(data.keys())[0]])):
	
	y = 0
	n = 0
	#test with the three decision trees
	for j in range(3):
		pred = tree_lis[j]
		while(pred!='yes' and pred!='no'):
			pred = pred[list(pred.keys())[0]][ data[list(pred.keys())[0]][i] ] 
		if(pred=='yes'):
			y+=alpha_lis[j]
		else:
			n+=alpha_lis[j]
	
	if(y>n):
		pred = 'yes'
	else:
		pred = 'no'
	# print(y,n)
	if(pred!=data[list(data.keys())[-1]][i]):
		incorrect+=1

print("Accuracy = "+str(100 - 100*(incorrect)/len(data[list(data.keys())[0]])))
