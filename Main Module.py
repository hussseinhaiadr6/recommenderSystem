# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:33:52 2021

@author: Hasan Ramadan
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor

#ML MODULE
df = pd.read_csv("m3p.csv")
# df = pd.get_dummies(df, drop_first=True)

df.replace(np.nan, 0, inplace = True)

X = np.array(df.drop(['STRESS/100', 'WorkLoad/100'], 1))
y1 = np.array(df['WorkLoad/100'])
y2 = np.array(df['STRESS/100'])

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.1, random_state = 0)
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.1, random_state = 0)

XGBModel1 = XGBRegressor()
XGBModel1.fit(X_train,y1_train , verbose=False)

XGBModel2 = XGBRegressor()
XGBModel2.fit(X_train,y2_train , verbose=False)

#Recommender  System Part
ds = pd.read_csv('course data.csv') #replace with your own csv file
ds = ds.iloc[:,:3]


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words= "english")
tfidf_matrix = tf.fit_transform(ds["description"])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results = {}
for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['ID'][i]) for i in similar_indices]
    results[row['ID']] = similar_items[1:]

#Final Model (FUNCTION)
def predict(L):
    def index(course):
        return ds.loc[ds['course ']==course].iat[0,0] #fast access using https://stackoverflow.com/questions/16729574/how-to-get-a-value-from-a-cell-of-a-dataframe
    def recommend(course, n):
        w = []
        id = index(course)
        recomms = results[id][:n]   
        for recomm in recomms: 
           w.append(ds['course '][recomm[1]-1]) 
        return w
    Q = []
    for i in range(len(L)):
            Q.append(recommend(L[i],3))
    def predict_(L):
        C = [0]*66
        P = []
        U = df.columns
        for x in U:
            P.append(x)
        for i in range(len(L)):
            C[P.index(L[i])] = 1
        C = np.array(np.array(C))
        C = C.reshape(1,-1)
        return C
    K = predict_(L)
    WL = XGBModel1.predict(K)
    Stress = XGBModel2.predict(K)
    WL = round(WL[0])
    Stress = round(Stress[0])
    if WL > 100: WL = 100
    if Stress > 100: Stress = 100
    print("Easier Alternatives: ")
    wlmin = 100
    smin = 100
    for i in range(len(L)):
        S = L.copy()
        for j in range(len(Q[i])):
            if (Q[i][j] not in S) and len(S[i])==len(Q[i][j]):
                S[i] = Q[i][j]
                s = predict_(S)
                wl = round(XGBModel1.predict(s)[0])
                stress = round(XGBModel2.predict(s)[0])
                if wl<=WL and stress<=Stress:
                    print(S,("Workload %: " + str(wl),"Stress %: " + str(stress)))
                if wl<=wlmin and stress<=smin:
                    l = S.copy()
                    wlmin = wl
                    smin = stress 
    print("Your Schedule: ") 
    print((L,("Workload %: " + str(WL),"Stress %: " + str(Stress))))
    print("Your SCHEDBEST schedule: ")
    return (l,("Workload %: " + str(wlmin),"Stress %: " + str(smin)))
   
     
    
    
        
#Test
print(predict(["EECE 442", "EECE 430","MKTG 210", "DCSN 200","EECE 442L", "EECE 451L"]))































































































































"""SHEEEEEEEEEEEESH"""


