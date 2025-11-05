from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split

#This file cleans and splits the data (as is done in Feature_selection_and_LR.ipynb),
# so that it can be imported in other files, rather than having the same pre-processing steps every time


communities_and_crime = fetch_ucirepo(id=183)
# data (as pandas dataframes)
X = communities_and_crime.data.features
y = communities_and_crime.data.targets.iloc[:,0]

X = X.replace('?', np.nan) #replace question marks with NaN
X=X.dropna(axis='columns') #remove columns with NaN
X=X.drop(['state', 'communityname','fold'], axis=1)
X_cleaned=X
xtr,xts,ytr,yts=train_test_split(X_cleaned,y,test_size=0.2,random_state=22)
x2tr,x2ts=xtr[['racePctWhite','PctKids2Par']],xts[['racePctWhite','PctKids2Par']]
eight_columns=['racepctblack', 'racePctWhite', 'pctWInvInc', 'pctWPubAsst',
       'PctPopUnderPov', 'TotalPctDiv', 'PctKids2Par', 'PctPersOwnOccup']
x8tr,x8ts=xtr[eight_columns],xts[eight_columns]

