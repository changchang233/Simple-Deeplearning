# -*- coding: utf-8 -*-

import operator
import pickle
import numpy as np
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

    
digits = load_digits()

n_samples, n_dimensionality=digits.data.shape
n_classes= digits.target_names.size

X= digits.data
y= digits.target


X_train, X_test, y_train, y_test= train_test_split(
        X, y, random_state=0)

error_set= np.zeros((10,), dtype = np.int)

def FuntionalityOne():
    
    '''
    print the Funtinality one's requirements
       print the number or data entries, nuber of classes,
             the number of data entries for each classes,
             the minmum and maximum values for each feature,
             train dataset and test dataset split
    '''
    
    print('*******************************************************************')
    print('Number of data entries is: ', n_samples)
    print('Dimensionality of one sample  is ',n_dimensionality)
    print('Number of feature is ', n_classes)
    print('The size of train dataset is: ', X_train.shape)
    print('The size of test dataset size is: ', X_test.shape)
    print('One data in the dataset show in array is')
    data= X_train[0]
    print(data.reshape(8,8))
    print('All of numbers has the light and black block,thus the maximum value',
          ' is: ',np.max(data))
    print('the minmum value is',np.min(data))
    
    for i in range(0,10):
           print('The number of data entries for Class ',
           i,' is: ', sum(y==i));
    print('*******************************************************************')
    
    return;


class rewrite:
    
    def Test(self, X_train, X_test, y_train, y_test, k):
        '''
        fetch one data from the dataset that need to indentify in order
        and send the data to mehtod KNN to indentify it's number value
        compare the return result with the ordinary label then get the accuray rate
        
        '''
      
        i=0 #use it to record the number of train data
        rightCount= 0
        for sample in X_train:
           label=  y_train[i]
           result= rewrite().KNN(sample, X_test, y_test, k ) 
           i=i+1
           if (result == label): rightCount += 1
           else: 
              rewrite().errorLocation(label)
        #print each classes' error identify 
        for a in range(0,10):
               print('The number of error identifying data of class ', a, ' is ',
               error_set[a])
        
        print('The number of total error identifying is', i-rightCount)
        print('The number of data in this test is ', i)
        print('The accuracy rate in this trainning is', rightCount/i)
    
        error_set.fill(0)#reset the error list

        return;
    
    def errorLocation(self, error_location):
        '''
        It use numpy to record several misidentifications in class0..9
        the numpy location's value crossponding to the class's value
        '''
        
        for i in range(0,10):
            if (i== error_location): 
                value= error_set[i]+1 
                error_set[i]=value
        return;


    def KNN(self, train_Sample, test_dataset, test_label, k):
        '''
        return the identify result
        
        the implementation of the algorithm K nearest neighbor
		'''
		
        diff_matrix = train_Sample - test_dataset# calculate the diff of two dataset
        sqare_diff_mat = diff_matrix ** 2 #calculate the diff square
        sqare_distances = sqare_diff_mat.sum(axis=1)#get the Euclidean distance's quare
        distances = sqare_distances ** 0.5#Get the distance from sample to the test point

        sorted_index = distances.argsort()
            
        class_count = {}#store the label
        
        for i in range(k):
                near_label = test_label[sorted_index[i]]# get the data label
                class_count[near_label] = class_count.get(near_label, 0) + 1#Count the number of occurrences of each data label
      
        #sort the label according to the occurrence number
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1),
                                reverse=True)
        
   
        return sorted_class_count[0][0]#Returns the label that appears the most
    
    

def rewriteModel():
    '''
    Ask user input 1 or other number to choose to load the model or
    do the training
    After training all the data, save the model
    '''
    print("Now do the rewrite model")
    print('Please input 1 to load the saved model, other input will',
          'run the model, then save the model')
    choice=input()
    
    print('Now you input ', choice)
    
    if choice=='1':
           pickle_load=open('rewrite_model.pickle', 'rb')
           load_model=pickle.load(pickle_load)
           
           print('the loaded training set is: ')
           load_model.Test(X_train, X_test, y_train, y_test, 5)
           
           print('')
           print('the loaded test set is: ')
           load_model.Test(X_test,X_train, y_test, y_train, 5)
           
           
    else:
       rewrite_model=rewrite()
       
       print('firstly do the train dataset')
       rewrite_model.Test(X_train, X_test, y_train, y_test, 5)
       print('')
       print('then do the test dataset')
       rewrite_model.Test(X_test,X_train, y_test, y_train, 5)
       
       
       file = open('rewrite_model.pickle', 'wb')
       pickle.dump(rewrite_model, file)
       file.close()
   
    return;
    
def libraryModel():
    '''
    Ask user input 1 or other number to choose to load the model or
    do the library function
    After training all data, save the model
    '''
    
    print("Now do the library model") 
    print('Please input 1 to load the saved model, other input will',
          'run the model, then save the model')
    print('')
    choice=input()
    
    print('Now you input ', choice)
    
    if choice=='1':
       print('Now load the model')
       pickle_load=open('library_model.pickle', 'rb')
       load_model=pickle.load(pickle_load)
       
       print('the accuracy rate for model training set is: ', 
             load_model.score(X_train, y_train))
       
       print('Now the accuracy rate for model test set is: ',
               load_model.score(X_test, y_test))
    else:
        print('Now do the library function')
        model_library=KNeighborsClassifier()
        model_library.fit(X_train,y_train)
        result_one= model_library.predict(X_test)
        
        print('Now the accuracy rate for training set is: ',
          accuracy_score(y_test, result_one))
        
        model_library.fit(X_test,y_test)
        result_two= model_library.predict(X_test)
        
        print('Now the accuracy rate for test set is: ',
               accuracy_score(y_test, result_two))
        #save the model
        file = open('library_model.pickle', 'wb')
        pickle.dump(model_library, file)
        file.close()
        
    
    return;

def FuntionalityFive():
    
    '''
    it does the cunctionality 5
    allow user to query the models by changing the input
    input 1 to do the rewrite model
    input 2 to do the library model
    other input will exit the program
    '''
    
    print('')
    print('Please input 1 to query rewrite model')
    print('And input 2 to query the library model')
    print('If you input other number, the program will be end')
    
    choice=input()
   
    print('Now you input ', choice, ' as your choice')
    
    if choice=="1": 
        rewriteModel();
    elif choice=="2": 
        libraryModel();
    else:
        sys.exit();
        
    FuntionalityFive()
    return;


FuntionalityOne()

print('')

FuntionalityFive()








