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
with open('data1_19.csv','rt')as f:
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
tree = Tree(data,attributes)
pprint.pprint(tree, indent=4)



			

			

	 