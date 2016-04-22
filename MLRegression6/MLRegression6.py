import pandas as pd
import numpy as np
from sklearn import linear_model
import random
import math

def loadFile(dataFile):
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
    ds = pd.read_csv(dataFile, dtype=dtype_dict)
    return ds

def get_numpy_data(df, features, output):
    df['constant'] = 1 # add a constant column 

    # prepend variable 'constant' to the features list
    features = ['constant'] + features

    # Filter by features
    fm = df[features]
    y = df[output]
   
    # convert to numpy matrix/vector whatever...
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html
    features_matrix = fm.as_matrix()
    output_array = y.as_matrix()

    return(features_matrix, output_array)

def predict_output(feature_matrix, weights):
    result = feature_matrix.dot(weights.T)
    return result

def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix / norms
    return(normalized_features, norms)


def euclideanDist(a, b):
    c = a-b
    sqSum = c.T.dot(c)
    return np.sqrt(sqSum)

def compute_distances(features_instances, features_query):
    diff = features_instances - features_query
    sqSum = np.sum(diff**2, axis=1)
    distances = np.sqrt(sqSum)
    return distances

def k_nearest_neighbors(k, feature_train, features_query):
    # get the distances in order
    distances = compute_distances(feature_train, features_query)
    
    # numpy.argsort is a better choice here
    # numpy.argsort(a, axis=-1, kind='quicksort', order=None)[source]
    # Returns the indices that would sort an array.

    n = len(distances)
    # use the decorate, sort, undecorate pradigm with tuples
    neighbors = [(distances[i], i) for i in range (0, n)]
    neighbors = sorted(neighbors)
    neighbors = [neighbors[j][1] for j in range (0, k)]

    return neighbors

def predict_output_of_query(k, features_train, output_train, features_query):
    neighbors = k_nearest_neighbors(k, features_train, features_query)
    total = np.sum([output_train[i] for i in neighbors])
    return total/k

def predict_output(k, features_train, output_train, features_query):
    predictions = []
    n = len(features_query)
    for i in range(0, n):
        predictions.append(predict_output_of_query(k, features_train, output_train, features_query[i]))
    return predictions

def main():
    # Use the house data
    print("\n\nUsing training data...")
    train_data = loadFile('kc_house_data_small_train.csv')

    # Had to look at the iPython notebook, people did not put the info for the Python users! :(
    # https://github.com/anindya-saha/Machine-Learning-with-Python/blob/master/Coursera-Machine-Learning-Regression/week-6-local-regression-assignment-graphlab.ipynb
    features = ['bedrooms',  
                'bathrooms',  
                'sqft_living',  
                'sqft_lot',  
                'floors',
                'waterfront',  
                'view',  
                'condition',  
                'grade',  
                'sqft_above',  
                'sqft_basement',
                'yr_built',  
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']
    output = 'price'
    (train_matrix, train_output) = get_numpy_data(train_data, features, output)
    (features_train, train_norms) = normalize_features(train_matrix)

    print("\n\nUsing test data...")
    test_data = loadFile('kc_house_data_small_test.csv')
    (test_matrix, test_output) = get_numpy_data(test_data, features, output)
    features_test = test_matrix / train_norms

    print("Query house: ", features_test[0])
    print("Training house #10;", features_train[9])
    print("Euclidean distance: ", euclideanDist(features_test[0], features_train[9]))

    minIndex = None
    minDist = None

    for i in range(0, 10):
        dist = euclideanDist(features_test[0], features_train[i])
        #print(dist)
        if minDist == None or  dist < minDist:
            minDist = dist
            minIndex = i

    print("The closest training house is (zero-based index): ", minIndex)

    # 1-NN regression
    diff = features_train[0:] - features_test[0]

    # Try the non matrix way
    #n = len(features_train)
    #diff = [features_train[i] - features_test[0] for i in range(0, n)]

    print("Testing a result of -0.0934339605842: ", diff[-1].sum())

    # Sum of squared feature differences for all training houses.
    sqSum = np.sum(diff**2, axis=1)
    print("Testing for equality: ", np.sum(diff**2, axis=1)[15], np.sum(diff[15]**2))

    distances = compute_distances(features_train[0:] , features_test[0])
    print("distances[100] should contain 0.0237082324496: ", distances[100])

    # 3rd house in the train set is the query house
    distances = compute_distances(features_train[0:], features_test[2])

    minIndex = None
    minDist = None
    n = len(distances)
    for i in range(0, n):
        dist = distances[i]
        #print(dist)
        if minDist == None or  dist < minDist:
            minDist = dist
            minIndex = i

    print("The closest training house is to features_test[2] (zero-based index): ", minIndex)
    print("Predicted value of features_test[2]: ", train_output[minIndex])

    # k-NN regression
    kNearest = k_nearest_neighbors(4, features_train[0:], features_test[2])
    print("The closest k(4) training houses to features_test[2] (zero-based index)): ", kNearest)
    price = predict_output_of_query(4, features_train[0:], train_output, features_test[2])
    print("The predicted price based on 4 nearest: ", price)

    # Predict the prices of the first 10 in the test set
    predictions = predict_output(10, features_train[0:], train_output, features_test[0:10])
    print("The predicted price based on 10 nearest for the first 10: ", predictions)
    # np.argmin is a btter choice
    lowest = np.argsort(predictions)
    print("The house with the lowest predicted value (zero-based index): ", lowest[0])
    print("The values of this house: ", predictions[lowest[0]])

    # Choosing the best k from the validation set
    kSet = [i for i in range(1, 16)]
    
    # Load the validation set
    print("\n\nUsing validation data...")
    validation_data = loadFile('kc_house_data_validation.csv')
    (validation_matrix, validation_output) = get_numpy_data(validation_data, features, output)
    features_validation = validation_matrix / train_norms

    rssSet = []
    for k in kSet:
        predictions = predict_output(k, features_train[0:], train_output, features_validation[0:])
        diff = predictions - validation_output
        rss = diff.T.dot(diff)
        rssSet.append(rss)
    
    minRssI = np.argmin(rssSet)
    print("The minimum rss index (zero-based): ", minRssI)
    k = kSet[minRssI]
    print("The value of k is: ", k)

    # Find RSS on the test data
    predictions = predict_output(k, features_train[0:], train_output, features_test[0:])
    diff = predictions - test_output
    rss = diff.T.dot(diff)
    print("The rss value is over the test set: ", rss)
main()