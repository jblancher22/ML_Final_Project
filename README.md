# ML_Final_Project
In this project, I used an ensemble of different ML method's to predict crime statistics as a final project for my Machine Learning Class.

The order of operations for this project is the following:
1. Feature_selection_and_LR. This file has the code for the process of feature selection and implements a linear regression model. The features selected and their subsequent test/training data are then written in the for_importing file for simplicity, such that we can import the for_importing file in each classification model's file and have the same data for each without having to re-write the code.
   
2. Binning. This file goes through the process of binning the target crime variable so that it can be a classification instead of a regression problem. Again, this binning is re-written in a concise way in binning_to_import so that it can be easily imported into subsequent classification models.
   
3. DNN, LogReg8, Neural_net_weighted, and SVC are the implementations of 4 different ML algorithms.

4. Ensemble. This file creates an ensemble learner of the aforementioned classification algorithms and the regression model.

5. Normalize. This file defines a function that will allow the raw crime data to be passed into the models by normalizing all features and the target variable from 0 to 1

6. Tkinter_gui. This creates a simple GUI to input crime data as percentages and it will output the predicted violent crime rate per 100k and its classification as either "high" or "low" crime neighborhood.

7. The Final_writeup file is a final writeup of the results and implementation of this study.
