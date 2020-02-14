Python 3 is used for this project.

#Constructs graph from train.txt and saves as networkx pickle file
construct_graph_save_to_txt()


#Reads graph from Graph.gpickle file and generates positive and negative samples from 
#given data and saves positive samples in positive_samples.csv and negative sample in 
#negative_samples.csv files.
randomly_select_positive_negative_edges()


#Reads positive_sample.csv and negative_samples.csv and generates feature values and saves #in train_dataset.csv and also generates feature values for edges in test-public.txt and #saves in test_dataset.csv 
extract_features()

#Reads train_dataset.csv and test_dataset.csv. train_dataset.csv is used to build models. #Edges in test_dataset.csv will be predicted with probability of having an edge. #Algorithms used are Random Forest and Bagging.
get_all_classifier_result()

#This is used to evaluate the model using 10-fold cross validation
validation()