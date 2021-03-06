
# import the needed packages
import pickle
from sklearn import preprocessing
import time
from os import listdir
from os.path import isfile, join
from random import randint, uniform
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
from skimage.filters import sobel


# set random seed
np.random.seed(26)

# the NaiveBayes classifier I wrote for assignment 6 in BSYSE_530, modified a little for this purpose
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
            
# this exists only for my testing purposes
class MatlabSurrogate():
    def __init__(self):
        self.state_of_mind = "Badass."
        
        
    def acquire_kinect_image(self, filename):
        # give this function a filename, and it will load that image with opencv
        # this will be a BGR format, because that is how opencv rolls
        kinect_image = cv.imread(filename)
        print(f"kinect has acquired the image with shape = {kinect_image.shape}")
        return kinect_image
    
    
    # function to display images resized, using opencv
    def imshow(self, image, imdiv = 4):
        imdiv = int(imdiv)
        w, h = int(image.shape[1]/imdiv), int(image.shape[0]/imdiv)
        cv.namedWindow("output", cv.WINDOW_NORMAL)
        cv.resizeWindow("output", (w, h))
        cv.imshow("output", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    
# I should probably have one image processing class that takes in a single image and then spits out a dataframe that could be used for prediction
# replaces ImageSegmenter
class ImageProcess():
    def __init__(self):
        print("image processor activated! use 'process_image_to_df()' to get back a pandas df")
        self.black_lower = (0, 0, 0)
        self.black_upper = (179, 255, 30)
        self.hsv_lower = (0, 0, 0)
        self.hsv_upper = (179, 255, 90)
#         self.black_lower = (0, 0, 203)
#         self.black_upper = (43, 255, 255)
#         self.hsv_lower = (0, 0, 70)
#         self.hsv_upper = (179, 34, 255)
# NOT mask for lego_imgs[14]
# hsv_lower = (0,0,0)
# hsv_upper = (179,234,77)

    

    def dummy_method(self, a):
        if type(a) is np.ndarray:
            result = "object is a numpy.ndarray, this is perfect. Is the image RGB order or BGR?"
            return result
        else:
            result = "object is a " + str(type(a)) + "and I'm gonna have a hard time with that"
            return result
      
    
        
    def bg_segmentation(self, image, mode="hsv", show_img=False):
                
            # create an hsv mask for red colors
            hsv_mask = cv.inRange(cv.cvtColor(image, cv.COLOR_BGR2HSV), 
                                 self.hsv_lower,
                                 self.hsv_upper).astype(np.uint8)
             
            # use this as a NOT mask
            hsv_mask = np.where(hsv_mask > 1, 0, 1).astype(np.uint8)
  
            hsv_mask = ndimage.gaussian_filter(hsv_mask, sigma=1)

            # erode the mask
            hsv_mask = morphology.erosion(hsv_mask, morphology.disk(3))

#             # median filter to despeckle
#             hsv_mask = ndimage.median_filter(hsv_mask, size=(3, 3)).astype(np.uint8)

            # binary dilation 
            hsv_mask = morphology.binary_dilation(hsv_mask, np.ones((20, 20))).astype(np.uint8)

            # fill the holes
            hsv_mask = ndimage.binary_fill_holes(hsv_mask).astype(np.uint8)

            # erode the mask
            hsv_mask = morphology.erosion(hsv_mask, morphology.disk(5))
            
            # TODO: remove this it is for testing purposes to show the segmentation
            if (show_img == True):                
                m = MatlabSurrogate()
                m.imshow(cv.bitwise_and(image, image, mask=hsv_mask).astype(np.uint8))
            
            # apply the mask and return the result        
            return cv.bitwise_and(image, image, mask=hsv_mask).astype(np.uint8)


    
    def process_image_to_df(self, input_image, area_th):
        
        seg_img = self.bg_segmentation(input_image, show_img=False)

    #     # make the mask a binary thresholded image
        mask = cv.cvtColor(seg_img, cv.COLOR_BGR2GRAY)
        mask = cv.GaussianBlur(mask,(5,5),0)
        ret3, mask = cv.threshold(mask,0,255,cv.THRESH_BINARY)



        # output image with contours drawn on the original image
        output_image = input_image.copy()

        # find the contours of the detected objects in the image
        contours, hier = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


        # create the df that we'll return for this image
        df = pd.DataFrame(columns=['color'])

        # # reset the object num
        object_num = 0
        
        for cnt in contours:
            # draw contours on the output image for our personal enjoyment
            cv.drawContours(output_image, [cnt], 0, color=(255, 255, 255), thickness=5)
            # CALCULATE ALL THE CONTOUR SHAPE FEATURES

            # get the x, y, w, h of the bounding rect for the contour
            x, y, w, h = cv.boundingRect(cnt)

            # contour features
            area = cv.contourArea(cnt)
            rect_area = w * h
            fullosity = area / rect_area

            aspect_ratio = float(w)/h
            extent = float(area/ rect_area)
            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            solidity = float(area)/hull_area

            eq_diameter = np.sqrt(4*area/np.pi)

            M= cv.moments(cnt)
            cx= int(M['m10']/M['m00'])
            cy= int(M['m01']/M['m00'])

            # take this rectangle as a subset of the input_image, and calculate things within it
            img_subset = input_image[y:y+h, x:x+w, :]

            # convert to hsv for extracting those values
            img_subset_hsv = cv.cvtColor(img_subset, cv.COLOR_BGR2HSV)

            # FILTER OUT THE WEIRD ONES
            # get rid of tiny objects that are probably noisef
            if area > area_th:
                # draw a blank canvas to put the contour onto, JUST THIS ONE not the others
                # this is a mask
                cimg_justthiscontour = np.zeros_like(input_image)

                # draw the contours on the blank canvas which is original sized
                cv.drawContours(cimg_justthiscontour, [cnt], 0, color=(255, 255, 255), thickness=-1)

                # now take the subset of just the area around the contour of interest
                cimg_subset = cimg_justthiscontour[y:y+h, x:x+w, :]

                # make a binary mask
                cimg_mask = cv.cvtColor(cimg_subset, cv.COLOR_BGR2GRAY)


                ret2, mask = cv.threshold(cimg_mask,0,255,cv.THRESH_BINARY)

                # draw contours on the output image for our personal enjoyment
                cv.drawContours(output_image, [cnt], 0, color=(255, 255, 255), thickness=5)

                img_subset = cv.bitwise_and(img_subset, img_subset, mask=mask).astype(np.uint8)

                # calculate where the object is
                pts = np.where(cimg_subset == 255)
                hue = img_subset_hsv[pts[0], pts[1], 0]
                sat = img_subset_hsv[pts[0], pts[1], 1]
                val = img_subset_hsv[pts[0], pts[1], 2]
                r = img_subset[pts[0], pts[1], 0]
                g = img_subset[pts[0], pts[1], 1]
                b = img_subset[pts[0], pts[1], 2]

                # and export the image for later analysis with something else like a neural network
                cv.imwrite(f"images/train/XX_{object_num}_{randint(10000,99999)}.png", img_subset)

                # add the object labels to the cimg for identification
                cv.putText(output_image, text= str(object_num), 
                           org=(cx - 5,cy - 5), 
                           fontFace= cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=3, 
                           color=(255,255,255), 
                           thickness=5, 
                           lineType=cv.LINE_AA)

        #         print(r.mean(), g.mean(), b.mean(), gli.mean())
                df = df.append({'color' : 0,
                                'x': x,
                                'y': y,
                                'object_num': object_num,
                                'r': r.mean(),
                                'g': g.mean(),
                                'b': b.mean(),
                                'hue': hue.mean(),
                                'sat': sat.mean(),
                                'val': val.mean()
                                 }, ignore_index=True)

                # last thing we do on this loop is increment the object_num
                object_num += 1

            #

        # end result should be a pandas dataframe and the contour image with numbers
        return df.sort_values(by='object_num', axis=0, ascending=True), output_image
    
    def hsv_slide_tool(self, image):
        
        def empty(a):
            pass
        
        h, w = int(image.shape[1]/2), int(image.shape[0]/2)
        cv.namedWindow('masked_image', cv.WINDOW_NORMAL)
        cv.resizeWindow('masked_image', h, w)
        
        cv.namedWindow("trackbars")
        cv.resizeWindow("trackbars", 800, 300)
                
        # color mask trackbars
        cv.createTrackbar("hue_min", "trackbars", 0, 179, empty)
        cv.createTrackbar('hue_max', 'trackbars', 179, 179, empty)
        cv.createTrackbar('sat_min', 'trackbars', 0, 255, empty)
        cv.createTrackbar('sat_max', 'trackbars', 255, 255, empty)
        cv.createTrackbar('val_min', 'trackbars', 0, 255, empty)
        cv.createTrackbar('val_max', 'trackbars', 255, 255, empty)
       

        while True:
            # get image
            img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            
            # get trackbar positions
            h_min = cv.getTrackbarPos("hue_min", "trackbars")
            h_max = cv.getTrackbarPos('hue_max', 'trackbars')
            s_min = cv.getTrackbarPos('sat_min', 'trackbars')
            s_max = cv.getTrackbarPos('sat_max', 'trackbars')
            v_min = cv.getTrackbarPos('val_min', 'trackbars')
            v_max = cv.getTrackbarPos('val_max', 'trackbars')
            
            #         self.black_lower = (0, 0, 0)
#         self.black_upper = (179, 255, 30)
#         self.hsv_lower = (0, 0, 100)
#         self.hsv_upper = (179, 255, 255)
            
            # create mask
            hsv_lower = np.array([h_min, s_min, v_min])
            hsv_upper = np.array([h_max, s_max, v_max])
            black_lower = np.array([0, 0, 0])
            black_upper = np.array([179, 255, 30])
            
            color_mask = cv.inRange(img_hsv, hsv_lower, hsv_upper)
            black_mask = cv.inRange(img_hsv, black_lower, black_upper)
            mask = color_mask + black_mask
            masked_image = cv.bitwise_and(img_hsv, img_hsv, mask=mask)
            
            cv.imshow('masked_image', masked_image)
            k = cv.waitKey(1000) & 0xFF # large wait time
            if k == 113 or k == 27:
                break
        
        cv.destroyAllWindows()
        print(f'hsv_lower is {hsv_lower}, hsv_upper = {hsv_upper}')
              
 
        
    def label_dataframe(self, image_df, class_list):
        for i, row in image_df.iterrows():
            image_df.loc[i, 'color'] = class_list[i]
        print(type(image_df))
        return image_df
    
#     def fake_df(self, input_df, reps = 3):
#         # creates a bunch of fake adjustments to the dataframe so my train set is bigger
#         output_df = input_df.copy()
        
#         for rep in range(0, reps):
#             fake_df = input_df.copy()
#             for i, row in fake_df.iterrows():
#                 fake_df.loc[i, 'r'] = fake_df.loc[i, 'r'] + uniform(-.1, .1)
#                 fake_df.loc[i, 'g'] = fake_df.loc[i, 'g'] + uniform(-.1, .1)
#                 fake_df.loc[i, 'b'] = fake_df.loc[i, 'b'] + uniform(-.1, .1)
#             output_df = pd.concat(output_df, fake_df)
                
#         return output_df
        
    def otsu_threshold(self, image):
        blur = cv.GaussianBlur(image,(5,5),0)
        ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return ret3, th3

    def process_image_make_predictions(self, input_image, model):
        
        predictive_model = model
        
        
        area_th = 400
        
        seg_img = self.bg_segmentation(input_image, show_img=False)

    #     # make the mask a binary thresholded image
        mask = cv.cvtColor(seg_img, cv.COLOR_BGR2GRAY)
        mask = cv.GaussianBlur(mask,(5,5),0)
        ret3, mask = cv.threshold(mask,0,255,cv.THRESH_BINARY)

        # output image with contours drawn on the original image
        output_image = input_image.copy()

        # find the contours of the detected objects in the image
        contours, hier = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # create the df that we'll return for this image
        df = pd.DataFrame(columns=['color'])

        # # reset the object num
        object_num = 0
        
        for cnt in contours:
            # draw contours on the output image for our personal enjoyment
            cv.drawContours(output_image, [cnt], 0, color=(255, 255, 255), thickness=5)
            # CALCULATE ALL THE CONTOUR SHAPE FEATURES

            # get the x, y, w, h of the bounding rect for the contour
            x, y, w, h = cv.boundingRect(cnt)

            # contour features
            area = cv.contourArea(cnt)
            rect_area = w * h
            fullosity = area / rect_area

            aspect_ratio = float(w)/h
            extent = float(area/ rect_area)
            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            solidity = float(area)/hull_area

            eq_diameter = np.sqrt(4*area/np.pi)

            M= cv.moments(cnt)
            cx= int(M['m10']/M['m00'])
            cy= int(M['m01']/M['m00'])

            # take this rectangle as a subset of the input_image, and calculate things within it
            img_subset = input_image[y:y+h, x:x+w, :]

            # convert to hsv for extracting those values
            img_subset_hsv = cv.cvtColor(img_subset, cv.COLOR_BGR2HSV)


            # FILTER OUT THE WEIRD ONES
            # get rid of tiny objects that are probably noisef
            if area > area_th:
                # draw a blank canvas to put the contour onto, JUST THIS ONE not the others
                # this is a mask
                cimg_justthiscontour = np.zeros_like(input_image)

                # draw the contours on the blank canvas which is original sized
                cv.drawContours(cimg_justthiscontour, [cnt], 0, color=(255, 255, 255), thickness=-1)

                # now take the subset of just the area around the contour of interest
                cimg_subset = cimg_justthiscontour[y:y+h, x:x+w, :]

                # make a binary mask
                cimg_mask = cv.cvtColor(cimg_subset, cv.COLOR_BGR2GRAY)


                ret2, mask = cv.threshold(cimg_mask,0,255,cv.THRESH_BINARY)

                # draw contours on the output image for our personal enjoyment
                cv.drawContours(output_image, [cnt], 0, color=(255, 255, 255), thickness=5)

                img_subset = cv.bitwise_and(img_subset, img_subset, mask=mask).astype(np.uint8)

                # calculate where the object is
                pts = np.where(cimg_subset == 255)
                hue = img_subset_hsv[pts[0], pts[1], 0]
                sat = img_subset_hsv[pts[0], pts[1], 1]
                val = img_subset_hsv[pts[0], pts[1], 2]
                r = img_subset[pts[0], pts[1], 0]
                g = img_subset[pts[0], pts[1], 1]
                b = img_subset[pts[0], pts[1], 2]
                
                df = [{'r': (r.mean() / 255),
                      'g': (g.mean() / 255),
                      'b': (b.mean() / 255),
                      'hue': (hue.mean() / 255),
                      'sat': (sat.mean() / 255),
                      'val': (val.mean() / 255)}]
                
                
                
                df = pd.DataFrame.from_dict(df)
                
                pred = predictive_model.get_predictions(df)
                
                class_dict = {0:"medium_blue",
                              1:"black",
                              2:"darK_stone_gray",
                              3:"bright_green",
                              4:"light_green",
                              5:"bright_orange",
                              6:"bright_red",
                              7:"bright_blue",
                              8:"white",
                              9:"bright_yellow"}
                color_text = class_dict[pred[0]]
                
                object_label = "obj" + str(object_num) + "_pred" + str(pred[0])
                print(object_label)    
                
                # add the object labels to the cimg for identification
                cv.putText(output_image, text= str(object_label), 
                           org=(cx - 5,cy - 5), 
                           fontFace= cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=1,
                           color=(0,255,0), 
                           thickness=3,
                           lineType=cv.LINE_AA)
                
                # last thing we do on this loop is increment the object_num
                object_num += 1
        
        # AFTER ALL CONTOURS HAVE BEEN DONE submit the df to the model for predictions
        
#         results = predictive_model.blind_predictions()
        # result = loaded_model.get_predictions(X_test, Y_test)
        # print(result)

        # # use the test set to see how we do
        # y_test_predictions = nb.get_predictions(X_test)

        # # scores
        # acc = nb.get_accuracy(y_test, y_test_predictions)
        # prec = precision_score(y_test, y_test_predictions, average="micro")
        # rec = recall_score(y_test, y_test_predictions, average="micro")
        # print(f"precision is {prec}, recall is {rec}, accuracy = {acc}")

        # # confusion matrix
        # labels = [(i, c) for i, c in class_dict.items()]
        # cm = confusion_matrix(y_test, y_test_predictions)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(cm)
        # plt.title('confusion matrix of the classifier')
        # fig.colorbar(cax)
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.show()
        # print(labels)

            # take the row
            

        
        
        # end result should be a pandas dataframe and the contour image with numbers
        return output_image
