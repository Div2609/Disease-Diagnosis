# Disease-Diagnosis
# Medical Diagonis  🌡
## Code 💻
We took the dataset from kaggel, Disease Symptom Dataset, then we filled nan in the columns were no symptom was present, then our dataset looks like this:
![](https://github.com/Ananyaiitbhilai/Assignment1c/blob/main/images/Screenshot%202022-02-21%20at%2012.39.05%20AM.png)<br>

We printed the unique symptoms from symptom-severity csv file and applied label binarizer to each symptom. Here we used label binarzer, to convert each symptom into a binary matrix, e.g. itching refers to [1,0,0,.....,0] , skin_rashes refers to [0,1,0,0,.....,0] vector. We have 132 symptoms in the dataset. This implies we have length of array as 132.

We have y_vec that has the list of diseases, and x_vec has the list of symptoms. For a corresponding value of x_vec there is a y_vec i.e. a disease. for instance:
![](https://github.com/Ananyaiitbhilai/Assignment1c/blob/main/images/Screenshot%202022-02-21%20at%201.09.48%20AM.png)<br>

It is being interpreted as:
![](https://github.com/Ananyaiitbhilai/Assignment1c/blob/main/images/Screenshot%202022-02-21%20at%201.14.05%20AM.png)<br>


We splitted the data into testing and training(85%).
We used inbuit decision tree function and saved the trained model in a pickel file. We got an accuracy of 100%, and the following confusion matrix.
![](https://github.com/Ananyaiitbhilai/Assignment1c/blob/main/images/Screenshot%202022-02-21%20at%201.19.12%20AM.png)<br>
![](https://github.com/Ananyaiitbhilai/Assignment1c/blob/main/images/Screenshot%202022-02-21%20at%201.19.27%20AM.png)<br>


Here even if the order of symptoms are changed i.e. order is shufffled, same disease is predicted. for e.g.
![](https://github.com/Ananyaiitbhilai/Assignment1c/blob/main/images/Screenshot%202022-02-21%20at%201.21.18%20AM.png)<br>
![](https://github.com/Ananyaiitbhilai/Assignment1c/blob/main/images/Screenshot%202022-02-21%20at%201.24.51%20AM.png)<br>


Are you concerned about your health, we present the one stop easy solution to check for the disease you might be suffering from, at your finger tips. Our model has an accuracy of 100%. You have to enter the symptoms you are having, based on a decision tree, more symptoms you enter, more is the tree traversed and you get more precise prediction of disease. <br>


## Our code of decision Tree
  
  ```
  ###defining entropy
def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy
  
##Info Gain

def InfoGain(data,split_attribute_name,target_name="Disease"):
    total_entropy = entropy(data[target_name])
    vals,counts = np.unique(data[split_attribute_name],return_counts=True)
    #cal the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).
                                dropna()[target_name])for i in range(len(vals))])
    
    #formula for information gain
    Information_Gain = total_entropy-Weighted_Entropy
    return Information_Gain

def ID3(data,originaldata,features,target_attribute_name="Disease",
        parent_node_class=None):
    #If all target_values have the same value,return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    #if the dataset is empty
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],
                                                                           return_counts=True)[1])]
    
    #If the feature space is empty
    elif len(features) == 0:
        return parent_node_class 

    #If none of the above condition holds true grow the tree

    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],
                                                                           return_counts=True)[1])]

    #Select the feature which best splits the dataset
    item_values = [InfoGain(data,feature,target_attribute_name)for feature in features] #Return the infgain values
    best_feature_index = np.argmax(item_values)
    best_feature = features[best_feature_index]

    #Create the tree structure
    tree = {best_feature:{}}

    #Remve the feature with the best info gain
    features = [i for i in features if i!= best_feature]

    #Grow the tree branch under the root node

    for value in np.unique(data[best_feature]):
        value = value
        sub_data = data.where(data[best_feature]==value).dropna()
        #call the ID3 algotirthm
        subtree = ID3(sub_data,df,features,target_attribute_name,parent_node_class)
        #Add the subtree
        tree[best_feature][value] = subtree
    return(tree)

#Predict
def predict(query,tree,default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
               result = tree[key][query[key]]
            except:
               return default

            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
##check the accuracy

def train_test_split(df):
    training_data = df.iloc[:3936].reset_index(drop=True)
    testing_data = df.iloc[3936:].reset_index(drop=True)
    return training_data,testing_data
training_data = train_test_split(df)[0]
testing_data = train_test_split(df)[1]

def test(data,tree):
   queries = data.iloc[:,:-1].to_dict(orient="records")
   predicted = pd.DataFrame(columns=["predicted"])

   #calculation of accuracy

   for i in range(len(data)):
       predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0)
   print("The Prediction accuracy is:",(np.sum(predicted["predicted"]==data["Disease"])/len(data))*100,'%')
  
#Train the tree,print the tree abnd predict the accuracy
tree = ID3(training_data,training_data,training_data.columns[:-1])
pprint(tree)
test(testing_data,tree)
  
  ```
