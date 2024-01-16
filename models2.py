#!/usr/bin/env python
# coding: utf-8

# # Setup

# ## Import Statements

# In[477]:


import pandas as pd
import numpy as np
import math
import random
from sklearn.metrics import accuracy_score


# # Classes

# ## Decision Tree - DecisionTree

# In[480]:


class DecisionTree:
    
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, x_train, y_train):
        self.tree = self._grow_tree(x_train, y_train)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if num_samples < 1:
            return {'class': 1}
        
        if num_features == 0:
            return {'class' : np.argmax(class_counts)}

        # If only one class in the node or maximum depth reached, create a leaf node
        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return {'class': unique_classes[0]}

        # Find the best split based on information gain
        best_feature, means, stds = self._best_split(X, y, class_counts)
        
        feature_means = []
        feature_stds  = []
        for k in means.keys():
            feature_means.append(means[k][best_feature])
            feature_stds.append(stds[k][best_feature])
            

        # If no split is found, create a leaf node
        if best_feature is None:
            return {'class': unique_classes[np.argmax(class_counts)]}

        
        left_part  = X[X[:,best_feature]==0]
        right_part = X[X[:,best_feature]==1]
        
        left_data  = np.concatenate((left_part[:, :best_feature],left_part[:,best_feature+1:]), axis=1)
        right_data = np.concatenate((right_part[:, :best_feature],right_part[:,best_feature+1:]), axis=1)
        
        # Recursively grow the left and right subtrees
        left_subtree  = self._grow_tree(left_data , y[X[:,best_feature] == 0], depth + 1)
        right_subtree = self._grow_tree(right_data, y[X[:,best_feature] == 1], depth + 1)
        
        
        # Return the current node
        return {
            'feature_index': best_feature,
            'left': left_subtree,
            'right': right_subtree,
            'mean': feature_means,
            'std' : feature_stds
        }

    def _best_split(self, X, y, class_counts):
        num_samples, num_features = X.shape
        
        class_priors = class_counts/np.sum(class_counts)
        # class_priors = np.reshape(class_priors.size,1)
        
        subsets = {}
        feature_mean_vals_subset = {}
        feature_std_vals_subset = {}
        
        for i in range(class_priors.size):
            subsets[i] = X[y==i]
            feature_mean_vals_subset[i] = np.mean(subsets[i], axis=0)
            feature_std_vals_subset[i] = np.std(subsets[i], axis=0)
        
                
        posterior_per_observation = {}
        #result = {}
        #result = {key : np.zeros(subsets[key].shape) for key in subsets}
        
        for key, values in subsets.items():
            
            posterior_per_observation[key] = class_priors[key] * self.likelihood_normal_dist(feature_mean_vals_subset[key], feature_std_vals_subset[key], subsets[key])
            result = np.zeros(subsets[key].shape)
            
            for i in range(class_priors.size):
                
                result += class_priors[i] * self.likelihood_normal_dist(feature_mean_vals_subset[i], feature_std_vals_subset[i], subsets[key])
            result += 1e-5
            posterior_per_observation[key] = posterior_per_observation[key] / result
        
        #for key in posterior_per_observation:
        #    posterior_per_observation[key] = posterior_per_observation[key]/result[key]
        
        
        posterior_per_feature = {j : np.sum(posterior_per_observation[j], axis = 0)/subsets[j].shape[0] for j in range(len(posterior_per_observation.keys()))}
        entropy_dict = {i : [posterior_per_feature[j][i]] for i in range(num_features) for j in range(len(subsets.keys()))}
        
        for i in entropy_dict:
            entropy_dict[i] = np.array(entropy_dict[i]) 
        
        # current entropy is a dict with features as keys and corresponding entropy mapped as values
        current_entropy = self._entropy(entropy_dict)

        entropy_feature_array = []
        
        for x in current_entropy:
            entropy_feature_array.append(current_entropy[x])

        entropy_feature_array = np.array(entropy_feature_array)
        best_feature = np.nanargmin(entropy_feature_array)
        best_threshold = current_entropy[best_feature]
        
        return best_feature, feature_mean_vals_subset, feature_std_vals_subset    
    
    
    # This function returns a dict with features as keys containing entropy 
    def _entropy(self, class_posterior_given_feature_dict):        
        # make sure class_posterior_given_feature_dict has features as keys and lists(prob for each class) as values 
        # make sure the indexed values are numpy arrays to make sure elements are multiplied and we dont get syntax error(python lists error)
        entropy = {}
        
        for feature in class_posterior_given_feature_dict:
            entropy[feature] = -np.sum(class_posterior_given_feature_dict[feature] * np.log2(class_posterior_given_feature_dict[feature] + 1e-10))
        
        return entropy
    
    def likelihood_normal_dist(self, mean_i, std_i, X_i):
        return np.exp(-1*((X_i-mean_i)**2)/2*std_i**2)/(std_i*math.sqrt(2*math.pi))

    def predict(self, train_data, test_data, unseen_data):
        x_train, y_train = train_data[:,:-1], train_data[:,-1]
        x_test, y_test = test_data[:,:-1], test_data[:,-1]
        x_unseen, y_unseen = unseen_data[:,:-1], unseen_data[:,-1]
        
        y_train_pred = np.array([self._predict_tree(x, self.tree) for x in x_train])
        y_test_pred = np.array([self._predict_tree(x, self.tree) for x in x_test])
        y_unseen_pred = np.array([self._predict_tree(x, self.tree) for x in x_unseen])
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        return y_test, y_test_pred, y_unseen, y_unseen_pred, train_accuracy

    def _predict_tree(self, x, node):
        if 'class' in node:
            return node['class']
        
        if self.likelihood_normal_dist(node['mean'][0], node['std'][0], x[node['feature_index']]) >= self.likelihood_normal_dist(node['mean'][1], node['std'][1], x[node['feature_index']]):
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])


# ## Random Forest - RandomForest

# In[487]:


class RandomForest:
    
    def __init__(self, max_depth=None, num_trees=5):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.cfs = []

    def fit(self, X, y):
        train_data_sets, target_sets = self._divide_data(X, y, self.num_trees)
        for i in range(self.num_trees):
            train_data = train_data_sets[i]
            targets = target_sets[i]
            self.tree = self._grow_tree(train_data, targets)
            self.cfs.append(self.tree)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if num_samples < 1:
            return {'class': 0}

        # If only one class in the node or maximum depth reached, create a leaf node
        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return {'class': unique_classes[0]}

        p = X.shape[1]
        m = math.ceil(np.sqrt(p))
        indices_chosen = np.random.randint(p, size = m)
        data = np.zeros(X.shape)
        for i in range(len(indices_chosen)):
            data[:, i] = X[:, indices_chosen[i]]

        # Find the best split based on information gain
        best_feature, means, stds = self._best_split(data, y, class_counts)
        
        feature_means = []
        feature_stds  = []
        for k in means.keys():
            feature_means.append(means[k][best_feature])
            feature_stds.append(stds[k][best_feature])
            

        # If no split is found, create a leaf node
        if best_feature is None:
            return {'class': unique_classes[np.argmax(class_counts)]}

        
        left_part  = X[X[:,best_feature]==0]
        right_part = X[X[:,best_feature]==1]
        
        left_data  = np.concatenate((left_part[:, :best_feature],left_part[:,best_feature+1:]), axis=1)
        right_data = np.concatenate((right_part[:, :best_feature],right_part[:,best_feature+1:]), axis=1)
        
        # Recursively grow the left and right subtrees
        left_subtree  = self._grow_tree(left_data , y[X[:,best_feature] == 0], depth + 1)
        right_subtree = self._grow_tree(right_data, y[X[:,best_feature] == 1], depth + 1)
        
        
        # Return the current node
        return {
            'feature_index': best_feature,
            'left': left_subtree,
            'right': right_subtree,
            'mean': feature_means,
            'std' : feature_stds
        }

    def _best_split(self, X, y, class_counts):
        num_samples, num_features = X.shape
        
        class_priors = class_counts/np.sum(class_counts)
        # class_priors = np.reshape(class_priors.size,1)
        
        subsets = {}
        feature_mean_vals_subset = {}
        feature_std_vals_subset = {}
        
        for i in range(class_priors.size):
            subsets[i] = X[y==i]
            feature_mean_vals_subset[i] = np.mean(subsets[i], axis=0)
            feature_std_vals_subset[i] = np.std(subsets[i], axis=0)
        
                
        posterior_per_observation = {}
        #result = {}
        #result = {key : np.ones(subsets[key].shape) for key in subsets}

        for key, values in subsets.items():
            
            posterior_per_observation[key] = class_priors[key] * self.likelihood_normal_dist(feature_mean_vals_subset[key], feature_std_vals_subset[key], subsets[key])
            result = np.zeros(subsets[key].shape)
            
            for i in range(class_priors.size):
                
                result += class_priors[i] * self.likelihood_normal_dist(feature_mean_vals_subset[i], feature_std_vals_subset[i], subsets[key])
            result += 1e-5
            posterior_per_observation[key] = posterior_per_observation[key] / result
        
        posterior_per_feature = {j : np.sum(posterior_per_observation[j], axis = 0)/subsets[j].shape[0] for j in range(len(posterior_per_observation.keys()))}
        entropy_dict = {i : [posterior_per_feature[j][i]] for i in range(num_features) for j in range(len(subsets.keys()))}
        
        for i in entropy_dict:
            entropy_dict[i] = np.array(entropy_dict[i]) 
        
        # current entropy is a dict with features as keys and corresponding entropy mapped as values
        current_entropy = self._entropy(entropy_dict)

        entropy_feature_array = []
        
        for x in current_entropy:
            entropy_feature_array.append(current_entropy[x])

        entropy_feature_array = np.array(entropy_feature_array)
        best_feature = np.nanargmin(entropy_feature_array)
        best_threshold = current_entropy[best_feature]
        
        return best_feature, feature_mean_vals_subset, feature_std_vals_subset

    # This function returns a dict with features as keys containing entropy 
    def _entropy(self, class_posterior_given_feature_dict):        
        # make sure class_posterior_given_feature_dict has features as keys and lists(prob for each class) as values 
        # make sure the indexed values are numpy arrays to make sure elements are multiplied and we dont get syntax error(python lists error)
        entropy = {}
        
        for feature in class_posterior_given_feature_dict:
            entropy[feature] = -np.sum(class_posterior_given_feature_dict[feature] * np.log2(class_posterior_given_feature_dict[feature] + 1e-5))
        
        return entropy
    
    def likelihood_normal_dist(self, mean_i, std_i, X_i):
        return np.exp(-1*((X_i-mean_i)**2)/2*std_i**2)/(std_i*math.sqrt(2*math.pi))

    def predict(self, test_data, unseen_data):
        x_test, y_test = test_data[:,:-1], test_data[:,-1]
        x_unseen, y_unseen = unseen_data[:, :-1], unseen_data[:, -1]
        
        test_preds = []
        unseen_preds = []
        rf_preds_test = []
        rf_preds_unseen = []
        for cf in self.cfs:
            rf_preds_test.append(np.array([self._predict_tree(x, cf) for x in x_test]))
            rf_preds_unseen.append(np.array([self._predict_tree(x, cf) for x in x_unseen]))
        t_preds_test = np.array(rf_preds_test)
        t_preds_unseen = np.array(rf_preds_unseen)
        
        for j in range(t_preds_test.shape[1]):
            votes=0
            for i in range(t_preds_test.shape[0]):
                votes += t_preds_test[i][j]
            if votes>2:
                test_preds.append(1)
            else:
                test_preds.append(0)
                
        for j in range(t_preds_unseen.shape[1]):
            votes=0
            for i in range(t_preds_unseen.shape[0]):
                votes += t_preds_unseen[i][j]
            if votes>2:
                unseen_preds.append(1)
            else:
                unseen_preds.append(0)
        return y_test, np.array(test_preds), y_unseen, np.array(unseen_preds)
            

    def _predict_tree(self, x, node):
        if 'class' in node:
            return node['class']
        
        if self.likelihood_normal_dist(node['mean'][0], node['std'][0], x[node['feature_index']]) >= self.likelihood_normal_dist(node['mean'][1], node['std'][1], x[node['feature_index']]):
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])
        
    def _divide_data(self, X, y, T):
        set_size = X.shape[0] // T
        n = X.shape[0]
        
        samples = []
        for i in range(T):
            indices = np.random.choice(n, set_size, replace=False)

            samples.append(list(indices))  

        # Split the dataset based on the random indices
        training_data_sets = [ X[samples[i], :] for i in range(len(samples))]
        targets = [ y[samples[i]]for i in range(len(samples))]

        return np.array(training_data_sets), np.array(targets)


# # Models

# Each function defined below is for different classifiers that work in the following way
# -  Take training, testing and unseen datasets as parameters
# -  Split the feature matrix x and labels y for training, testing and unseen sets
# -  Trains the model using training data
# -  Make predictions on the testing, unseen data
# -  Return the actual and predicted labels for the testing, unseen set

# ## Multiple Linear Regression - linreg()

# In[465]:


# Function for multiple linear regression with epochs and learning rate
def linreg(train_data, test_data, unseen_data, eta=0.001, epochs=5000):
    
    # Split feature matrix and target vectors
    x_train, y_train = train_data.drop('target', axis=1), train_data['target']
    x_test, y_test = test_data.drop('target', axis=1), test_data['target']
    x_unseen, y_unseen = unseen_data.drop('target', axis=1), unseen_data['target']
    
    # Initialize weights
    w = np.zeros(x_train.shape[1])

    # Training loop for epochs with learning rate eta
    for epoch in range(epochs):
        y_train_pred = np.dot(x_train, w)
        gradient = np.dot(x_train.T, (y_train_pred - y_train)) / len(y_train)

        # Gradient clipping for stability
        gradient = np.clip(gradient, -1, 1)
        
        w -= eta * gradient
        
    # Convert continuous predictions to binary class labels with thresholding
    predict = lambda x: np.where(np.dot(x, w) >= 0, 1, 0)
    y_test_pred = predict(x_test)
    y_unseen_pred = predict(x_unseen)
    
    # Get training accuracy to be used for ensembling
    y_train_pred = predict(x_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Return actual and predicted test, unseen labels and also the training accuracy
    return y_test, y_test_pred, y_unseen, y_unseen_pred, train_accuracy


# ## Logistic Regression - logreg()

# In[462]:


# Function for logistic regression with 5000 epochs and 0.001 learning rate
def logreg(train_data, test_data, unseen_data, eta = 0.001, epochs = 5000):
    
    # Split feature matrix and target vectors
    x_train, y_train = train_data.drop('target', axis=1), train_data['target']
    x_test, y_test = test_data.drop('target', axis=1), test_data['target']
    x_unseen, y_unseen = unseen_data.drop('target', axis=1), unseen_data['target']
    
    # Define sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))   
    
    # Initialize weights
    w = np.random.rand(x_train.shape[1]) * 2e-4 - 1e-4

    # Training loop for 5000 epochs with learning rate of 0.01
    for _ in range(epochs):
        y_train_pred = sigmoid(np.dot(x_train, w))
        w -= eta * (np.dot(x_train.T, (y_train_pred - y_train)) / len(y_train))
        
    # Get predictions for testing, unseen set
    predict = lambda x: np.where(sigmoid(np.dot(x, w)) >= 0.5, 1, 0)
    y_test_pred = predict(x_test)
    y_unseen_pred = predict(x_unseen)
    
    # Get training accuracy to be used for ensembling
    y_train_pred = predict(x_train)
    train_accuracy = accuracy_score(y_train,y_train_pred)
    
    # Return actual and predicted test,unseen labels and also the training accuracy
    return y_test, y_test_pred, y_unseen, y_unseen_pred, train_accuracy


# ## Linear Discriminant Analysis - lda()

# In[366]:


# Function for LDA
def lda(train_data, test_data, unseen_data):
    
    # Split feature matrix and target vectors
    x_train, y_train = train_data.drop('target', axis=1), train_data['target']
    x_test, y_test = test_data.drop('target', axis=1), test_data['target']
    x_unseen, y_unseen = unseen_data.drop('target', axis=1), unseen_data['target']
    
    # Compute mu for features for class 0,1
    mu = np.array([x_train[y_train == i].mean(axis=0) for i in np.unique(y_train)])
    
    # Compute covariance matrices sigma 
    sigma = [np.cov(x_train[y_train == i], rowvar=False, bias=True) for i in np.unique(y_train)]
    
    # Compute between class scatter matrix
    sb = np.outer(mu[0]-mu[1], mu[0]-mu[1])
    
    # Compute within class scatter matrix
    sw = sigma[0] + sigma[1]
    
    # Get the sw-1sb matrix
    sm = np.dot(np.linalg.pinv(sw),sb)
    
    # Get eigenvalues and eigenvectors
    eigval,eigvec = np.linalg.eig(sm)
    
    # Get w as eigvec with largest eigval
    w = eigvec[:, np.argmax(np.abs(eigval))]
    
    # Get z for training and testing
    z_train = x_train.dot(w)

    # Get training post-projection mu for classes
    mu0 = np.mean(z_train[y_train == 0])
    mu1 = np.mean(z_train[y_train == 1])

    
    # Function to predict class labels
    def predict(x_data, w, mu0, mu1):
        
        # Get z for data
        z_data = x_data.dot(w)

        # Return predictions classified using training means 
        return np.where(np.abs(z_data - mu1) > np.abs(z_data - mu0), 0, 1)

    # Get predictions for all datasets
    y_train_pred = predict(x_train, w, mu0, mu1)
    y_test_pred = predict(x_test, w, mu0, mu1)
    y_unseen_pred = predict(x_unseen, w, mu0, mu1)
    
    # Get training accuracy
    train_accuracy = accuracy_score(y_train,y_train_pred)
      
    # Return actual and predicted test,unseen labels and also the training accuracy
    return y_test, y_test_pred, y_unseen, y_unseen_pred, train_accuracy


# ## Naive Bayes - nb()

# In[354]:


# Function for naive bayes classifier
def nb(train_data,test_data, unseen_data):
    
    # Split feature matrix and target vectors
    x_train = train_data.drop(['target'], axis=1)
    x_test = test_data.drop(['target'], axis=1)
    x_unseen = unseen_data.drop(['target'], axis=1)
    y_train = train_data['target']
    y_test = test_data['target']
    y_unseen = unseen_data['target']
    
    
    """Step 1: Calculate class priors P(y)"""
    
    # Initialize variable to store class priors
    p_y = y_train.value_counts(normalize=True)
    
    
    """Step 2: Calculate likelihoods P(x|y)"""
    
    # Initialize variables to store means and variances for each class and feature
    means = {}
    variances = {}
    
    # Calculate means and variances for each class and feature
    for curr_class in p_y.index:
        means[curr_class] = x_train[y_train == curr_class].mean()
        variances[curr_class] = x_train[y_train == curr_class].var()
    
    # Function to calculate Gaussian probability density
    def gaussian_pdf(x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
    
    
    """Step 3. Classify sets to get P(y|x) predictions"""
    
    # Function to predict class labels
    def predict(x_data, class_priors, means, variances):
        
        # Initialize variable to store predictions for any dataset passed
        y_data_pred = []
        
        # Loop through rows of x_data
        for i, row in x_data.iterrows():
            
            # Initialize variable to hold the class scores
            p_yx_curr = {}
            
            # Go through all the classes in class priors 
            for curr_class in class_priors.index:

                # Add log(P(y)) for the current class prediction
                p_yx_curr[curr_class] = np.log(class_priors[curr_class])

                # Go through all the features in testing set
                for ft in x_data.columns:

                    # Get mean and variance for current class and feature
                    mean = means[curr_class][ft]
                    var = variances[curr_class][ft]

                    # Add log likelihood to the current P(y|x) for every class and ft for current row in x_test
                    # Add small constant to avoid log(0)
                    p_yx_curr[curr_class] += np.log(gaussian_pdf(row[ft], mean, var) + 1e-6)

            # Choose the most likely class and append to the prediction
            y_data_pred.append(max(p_yx_curr, key=p_yx_curr.get))
        
        # Return the predicted labels
        return y_data_pred
    
    # Get predictions for all datasets
    y_train_pred = predict(x_train, p_y, means, variances)
    y_test_pred = predict(x_test, p_y, means, variances)
    y_unseen_pred = predict(x_unseen, p_y, means, variances)
    
    # Get training accuracy
    train_accuracy = accuracy_score(y_train,y_train_pred)
 
    
    # Return actual and predicted test,unseen labels and also the training accuracy   
    return y_test,y_test_pred, y_unseen, y_unseen_pred,train_accuracy 


# ## Decision Tree Learning - dtl()

# In[481]:


def dtl(train_data, test_data, unseen_data, max_depth = 4):
    x_train, y_train = train_data.iloc[:,:-1], train_data.iloc[:,-1]
    tree = DecisionTree(max_depth)
    tree.fit(x_train.values, y_train.values)
    y_test, y_test_pred, y_unseen, y_unseen_pred, train_accuracy = tree.predict(train_data.values, test_data.values, unseen_data.values)
    return y_test, y_test_pred, y_unseen, y_unseen_pred, train_accuracy


# ## All Models Ensemble Learning - el_all()

# In[491]:


def el_all(train_data, test_data, unseen_data):
    
    # Classifiers to use for simple ensemble learning
    classifiers = [linreg,logreg, lda, nb, dtl]
    
    # Inbuilt function to return weighted vote
    def weighted_vote(predictions, weights):
        return np.round(np.average(predictions, axis=0, weights=weights)).astype(int)
    
    # Initialize lists to store actual/predicted labels for all classifiers
    all_y_test, all_y_test_pred = [], []
    all_y_unseen, all_y_unseen_pred = [], []
    
    # Initialize list to store training accuracy for all classifiers to be used as weights
    all_training_accuracy = []

    # Go through each classifier
    for clf in classifiers:
        
        print(f"Ensembling {clf.__name__ }")
        
        # Get outputs from each classifier
        y_test, y_test_pred, y_unseen, y_unseen_pred, train_accuracy = clf(train_data, test_data, unseen_data)
        
        # Append the testing actual/predicted labels
        all_y_test.append(y_test)
        all_y_test_pred.append(y_test_pred)
        
        # Append the unseen actual/predicted labels
        all_y_unseen.append(y_unseen)
        all_y_unseen_pred.append(y_unseen_pred)
        
        # Append the training accuracy for this classifier
        all_training_accuracy.append(train_accuracy)
    
    
    # Normalize the training accuracies to use as weights
    all_training_accuracy = np.array(all_training_accuracy)
    normalized_weights = all_training_accuracy / all_training_accuracy.sum()

    # Compute weighted votes for the final predictions
    final_test_pred = weighted_vote(np.array(all_y_test_pred), normalized_weights)
    final_unseen_pred = weighted_vote(np.array(all_y_unseen_pred), normalized_weights)
    
    # Define dummy veriable dummy_train_accuracy to be consistent with rest of the models
    dummy_train_accuracy = 0

    return all_y_test[0], final_test_pred, all_y_unseen[0], final_unseen_pred, dummy_train_accuracy


# ## Linear Models Ensemble Learning - el_lin()

# In[488]:


def el_lin(train_data, test_data, unseen_data):
    
    # Classifiers to use for simple ensemble learning
    classifiers = [linreg,lda, dtl]
    
    # Inbuilt function to return weighted vote
    def weighted_vote(predictions, weights):
        return np.round(np.average(predictions, axis=0, weights=weights)).astype(int)
    
    # Initialize lists to store actual/predicted labels for all classifiers
    all_y_test, all_y_test_pred = [], []
    all_y_unseen, all_y_unseen_pred = [], []
    
    # Initialize list to store training accuracy for all classifiers to be used as weights
    all_training_accuracy = []

    # Go through each classifier
    for clf in classifiers:
        
        print(f"Ensembling {clf.__name__ }")
        
        # Get outputs from each classifier
        y_test, y_test_pred, y_unseen, y_unseen_pred, train_accuracy = clf(train_data, test_data, unseen_data)
        
        # Append the testing actual/predicted labels
        all_y_test.append(y_test)
        all_y_test_pred.append(y_test_pred)
        
        # Append the unseen actual/predicted labels
        all_y_unseen.append(y_unseen)
        all_y_unseen_pred.append(y_unseen_pred)
        
        # Append the training accuracy for this classifier
        all_training_accuracy.append(train_accuracy)
    
    
    # Normalize the training accuracies to use as weights
    all_training_accuracy = np.array(all_training_accuracy)
    normalized_weights = all_training_accuracy / all_training_accuracy.sum()

    # Compute weighted votes for the final predictions
    final_test_pred = weighted_vote(np.array(all_y_test_pred), normalized_weights)
    final_unseen_pred = weighted_vote(np.array(all_y_unseen_pred), normalized_weights)
    
    # Define dummy veriable dummy_train_accuracy to be consistent with rest of the models
    dummy_train_accuracy = 0

    return all_y_test[0], final_test_pred, all_y_unseen[0], final_unseen_pred, dummy_train_accuracy


# ## Random Forest Ensemble Learning - el_rf()

# In[490]:


def el_rf(train_data, test_data, unseen_data, max_depth=3, num_trees=3):
    x_train, y_train = train_data.iloc[:,:-1], train_data.iloc[:,-1]
    rf = RandomForest(max_depth, num_trees)
    rf.fit(x_train.values, y_train.values)
    y_test, y_test_preds, y_unseen, y_unseen_preds = rf.predict(test_data.values, unseen_data.values)
    return y_test, y_test_preds, y_unseen, y_unseen_preds, 0

