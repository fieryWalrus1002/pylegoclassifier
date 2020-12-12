import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from scipy import ndimage
from skimage import morphology
from skimage import exposure
import os
from math import pi
from math import isnan
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score


class NaiveBayes:
    # P(c|x) = P(x|c) * P(c) / P(x)
    # P(x|x) is the posterior probability
    # P(x|c) is the likelihood
    # P(c) is the class prior probability, or the prob of c occuring indpendently. 
    # P(x) is the predictor prior probability, or the prob of x occuring independently
    
    def fit(self, features, target):
        # define class variables
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.feature_nums = features.shape[1]
        self.rows = features.shape[0]
        
        # calculate statistics for all those features
        self.calc_statistics(features, target)
        
        # prior is the random chance of drawing a particular class based on its proportion in the dataset
        self.prior = self.calc_prior(features, target)
        
              
    def get_predictions(self, input_vector):
        predictions = []
        
        for i in range(len(input_vector)):
            result = self.calc_posterior((input_vector.iloc[i,:]))
            predictions.append(result)
        return predictions
     

    def predict(self, observation):
        #call the calc_posterior function on the observation
        pred_class = self.calc_posterior(observation)
        return pred_class
        
        
    def calc_statistics(self, features, target):
        # calculate mean, variance for each column and convert to numpy array
        self.mean = features.groupby(target).apply(np.mean).to_numpy()
        self.var = features.groupby(target).apply(np.var).to_numpy()
        return self.mean, self.var
    
    
    def calc_prior(self, features, target):
        # this is the probability of picking one of a class at random from the dataset
        self.prior = (features.groupby(target).apply(lambda x: len(x)/self.rows).to_numpy())
        return self.prior
    
    
    def calc_posterior(self, x):
        # this is the probability, post evidence
        # x is a numpy array
        # x is feature vector for one observation 
                
        # make a list that we will add each classes posterior prob to
        posteriors = []
        
        # iterate through the classes
        for i in range(0, self.count):
            # for each class look at the prior probability for the class
            prior = self.prior[i]
            
            # calculate the conditional probability for the 
            conditional = np.sum(self.gaussian_density(i, x))
            posterior = prior + conditional
            #  print(f"i = {i}, prior = {prior}, conditional = {conditional}, posterior = {posterior}")
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]
        
        
    def gaussian_density(self, class_idx, x):
        # calc probability from gaussian denssityy fucntion (normal dist)
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        # this part sucked and I had a typo that cost me hours
        numerator = np.exp(-((x-mean)**2 / (2 * var)))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
        
    
    def pdf(self, x, mean, stdev):
        # calculate probability density function
        exponent = np.exp(-((x-mean)**2 / (2*stdev**2)))
        return exponent * (1/(np.sqrt(2*np.pi)*stdev))

        
    def get_accuracy(self, test, predictions):
        correct = 0
        for i in range(len(test)):
            if test.iloc[i] == predictions[i]:
                correct += 1
        return (correct / float(len(test)))


# TODO: read these and see how it works        
# https://www.mathworks.com/help/matlab/matlab_external/matlab-arrays-as-python-variables.html        
# https://www.mathworks.com/help/matlab/matlab_external/passing-data-to-python.html        

class ImageSegmenter:
    def bg_segmentation(self, image):
        
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # create an hsv mask for red colors
        hsv_mask = cv.inRange(hsv_image, 
                             (0, 0, 100),
                             (360, 255, 255)).astype(np.uint8)
        hsv_mask = np.where(hsv_mask > 0, 1, 0).astype(np.uint8)

        # median filter
        hsv_mask = ndimage.median_filter(hsv_mask, size=(3, 3)).astype(np.uint8)

        # binary dilation
        hsv_mask = morphology.binary_dilation(hsv_mask, np.ones((6, 6))).astype(np.uint8)

        # erode the mask
        hsv_mask = morphology.erosion(hsv_mask, morphology.disk(5))

        # apply the mask
        image_2 = cv.bitwise_and(image, image, mask =hsv_mask).astype(np.uint8)

        return image_2
    
        
    def dummy_method(self, a):
        if type(a) is np.ndarray:
            result = "object is a numpy.ndarray"
            return result
        else:
            result = "object is a " + str(type(a)) + "and I'm gonna have a hard time with that"
            return result
                  


#imseg = ImageSegmenter()
#nb = NaiveBayes()
#print(imseg.dummy_method("I'm a carrot!"))
