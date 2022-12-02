###### MODEL DEPLOYMENT: HOUSE PRICES ######

import pandas as pd
import numpy as np
import streamlit as st

#Load the Dataset
train = pd.read_csv("train.csv")

#Pick Variables
train_a = train[['SalePrice','LotArea', 'LotFrontage','OverallCond','Foundation', 'MSSubClass']].copy() ##create a new dataframe for selected variables

#One Hot Encoding, Fill NA
for i in train_a.columns[train_a.isnull().any(axis=0)]:     
    train_a['LotFrontage'].fillna(train_a[i].mean(),inplace=True) ## fill na with mean 
    
train_a = pd.get_dummies(train_a)
train_a.info()

#Defining Feature Matrix(X) and Response Vector(Y)

X = train_a.drop('SalePrice',axis=1)
Y = train_a['SalePrice']

#Splitting X and Y into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.4,
                                                    random_state=1)

#Title of Web
st.title("Predict Your Dream House's Price Here!")

#Input data 1
OverallCond = st.selectbox(
    'From a scale of 1-10, what condition do you want for your house? (1 being "very poor", and 10 being "very excellent")',
    (1,2,3,4,5,6,7,8,9,10))
'You selected:', OverallCond

#Input data 2
MSSubClass = st.selectbox('What type of dwelling involved in the sale do you prefer?',
    ('1-STORY 1946 & NEWER ALL STYLES','1-STORY 1945 & OLDER','1-STORY W/FINISHED ATTIC ALL AGES',
    '1-1/2 STORY - UNFINISHED ALL AGES','1-1/2 STORY FINISHED ALL AGES','2-STORY 1946 & NEWER',
    '2-STORY 1945 & OLDER','2-1/2 STORY ALL AGES','SPLIT OR MULTI-LEVEL','SPLIT FOYER','DUPLEX - ALL STYLES AND AGES',
    '1-STORY PUD (Planned Unit Development) - 1946 & NEWER','1-1/2 STORY PUD - ALL AGES',
    '2-STORY PUD - 1946 & NEWER','PUD - MULTILEVEL - INCL SPLIT LEV/FOYER',))

'You selected:', MSSubClass

if MSSubClass=='1-STORY 1946 & NEWER ALL STYLES':
    MSSubClass=20
elif MSSubClass=='1-STORY 1945 & OLDER':
    MSSubClass=30
elif MSSubClass=='1-STORY W/FINISHED ATTIC ALL AGES':
    MSSubClass=40
elif MSSubClass=='1-1/2 STORY - UNFINISHED ALL AGES':
    MSSubClass=45
elif MSSubClass=='1-1/2 STORY FINISHED ALL AGES':
    MSSubClass=50
elif MSSubClass=='2-STORY 1946 & NEWER':
    MSSubClass=60
elif MSSubClass=='2-STORY 1945 & OLDER':
    MSSubClass=70
elif MSSubClass=='2-1/2 STORY ALL AGES':
    MSSubClass=75
elif MSSubClass=='SPLIT OR MULTI-LEVEL':
    MSSubClass=80
elif MSSubClass=='SPLIT FOYER':
    MSSubClass=85
elif MSSubClass=='DUPLEX - ALL STYLES AND AGES':
    MSSubClass=90
elif MSSubClass=='1-STORY PUD (Planned Unit Development) - 1946 & NEWER':
    MSSubClass=120
elif MSSubClass=='1-1/2 STORY PUD - ALL AGES':
    MSSubClass=150
elif MSSubClass=='2-STORY PUD - 1946 & NEWER':
    MSSubClass=160
elif MSSubClass=='PUD - MULTILEVEL - INCL SPLIT LEV/FOYER':
    MSSubClass=180
else:
    MSSubClass=190
    
#Input data 3
LotArea = st.slider('What is your preferable lot size?',int(X.LotArea.min()),int(X.LotArea.max()),int(X.LotArea.mean()))

#Input data 4
LotFrontage = st.slider('How many feet of street connected to the property do you want?',int(X.LotFrontage.min()),int(X.LotFrontage.max()),int(X.LotFrontage.mean()))

st.write('If you have picked "Yes" at a type of foundation that represent your house, please leave the other foundations as "No"')

#Input data 5
Foundation_BrkTil = st.selectbox(
    'Do you want brick and tile as a foundation for your house?',
     ('No','Yes'))

'You selected:', Foundation_BrkTil

if Foundation_BrkTil=="Yes":
    Foundation_BrkTil=1
else:
    Foundation_BrkTil=0

#Input data 6
Foundation_CBlock = st.selectbox(
    'Do you want cinder block as a foundation for your house?',
      ('No','Yes'))
'You selected:', Foundation_CBlock

if Foundation_CBlock=="Yes":
    Foundation_CBlock=1
else:
    Foundation_CBlock=0

#Input data 7
Foundation_PConc = st.selectbox(
    'Do you want poured contrete as a foundation for your house?',
      ('No','Yes'))
'You selected:', Foundation_PConc

if Foundation_PConc=="Yes":
    Foundation_PConc=1
else:
    Foundation_PConc=0

#Input data 8
Foundation_Slab = st.selectbox(
    'Do you want slab as a foundation for your house?',
      ('No','Yes'))
'You selected:', Foundation_Slab

if Foundation_Slab=="Yes":
    Foundation_Slab=1
else:
    Foundation_Slab=0

#Input data 9
Foundation_Stone = st.selectbox(
    'Do you want stone as a foundation for your house?',
      ('No','Yes'))
'You selected:', Foundation_Stone

if Foundation_Stone=="Yes":
    Foundation_Stone=1
else:
    Foundation_Stone=0

#Input data 10
Foundation_Wood = st.selectbox(
    'Do you want wood as a foundation for your house?',
      ('No','Yes'))
'You selected:', Foundation_Wood

if Foundation_Wood=="Yes":
    Foundation_Wood=1
else:
    Foundation_Wood=0

#Create linear regression object
from sklearn import linear_model
reg = linear_model.LinearRegression()

#Train the model using the training sets
reg.fit(X_train, Y_train)

#Output data
prediction = reg.predict([[OverallCond, MSSubClass, LotArea, LotFrontage, Foundation_BrkTil, Foundation_CBlock, Foundation_PConc,Foundation_Slab, Foundation_Stone,Foundation_Wood]])[0]

if st.button('Predict'):
    st.header('Your estimated dream house price will be $ {}'.format(int(prediction)))
