# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:48:58 2022

@author: MOB220005316
"""
#!pip install MissForest
#!pip install pmdarima
#!pip install tk


import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os

# Imputer Models
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
#from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer   # IterativeImputer is experimental and the API might change without any deprecation cycle. To use it, you need to explicitly import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import BayesianRidge
import sklearn.neighbors._base    # Run this to import Miss Forest 
import sys # Run this to import Miss Forest 
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base # Run this to import Miss Forest 
from missingpy import MissForest
# Evaluation 
#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import roc_auc_score
#Time Series data


from pmdarima.arima import auto_arima
from statsmodels.tsa.vector_ar import vecm


def Null_Values_display(dfx):
    print("\n--------Percentage of Missing values---------\n")
    for columns in dfx.columns:        
        print("  " ,columns,":",round(dfx[columns].isnull().sum()*100/len(dfx),2),'%',"  (",dfx[columns].dtype,")")
    #count_missing = df1.isnull().sum()
    print("\n--------------Data Summary--------------\n")
    print(" ")
    print(dfx.describe(include='all').T[['unique','top','freq','mean','min','max']])
    #percent_missing = df1.isnull().sum() * 100 / len(df1)    
    #print(percent_missing)
    #print("\n")
    
def DataPreprocessing(dfx):
    dfx= dfx.replace("[@_!#$%^&*()<>?/|}{~:]",'',regex=True)
    return dfx

def Select_features(dfx):
    global Not_Req_columns
    print(" ")
    print("---------Drop unnecessary features----------")
    print(pd.DataFrame(dfx.columns))
    Not_Req_columns = []   
    print("")
    n=int(input("Enter the number of features to be dropped:"))
    for i in range(0,n):
        x=int(input("Enter the Feature index :"))
        Not_Req_columns.append(x)
    dfx=dfx.drop(dfx.columns[Not_Req_columns], axis=1)

    return dfx  

def LabelEncoding(dfx):
    global d,cat_columns,num_columns
    original = dfx.copy()
    #get all categorical columns
    cat_columns = dfx.select_dtypes(['object']).columns
    num_columns = dfx.select_dtypes(['int','float']).columns
    #convert all categorical columns to numeric
    d={}
    for col in cat_columns:
        d[col]=LabelEncoder().fit(dfx[col])
    mask = dfx.isnull()
    for col in cat_columns:    
        dfx[col] = d[col].transform(dfx[col]) 
        dfx = dfx.where(~mask, original)   
    #dfx[cat_columns] = dfx[cat_columns].apply(le.fit_transform)
    #dfx[cat_columns] = dfx[cat_columns].apply(lambda x: pd.factorize(x)[0])
    return dfx

       
def inverse_labels(dfx,d):
    for col in cat_columns:
        dfx[col] = dfx[col].astype(int)
    for col in cat_columns:
        dfx[col] = d[col].inverse_transform(dfx[col])
    return dfx

def Evaluation_clf(X,y,model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                        random_state=random_state,stratify=y)   
    if y.nunique()[0]==2:
        clf = RandomForestClassifier(random_state=random_state).fit(X_train,y_train)
        print('Roc_auc_score for',model_name,round(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]),3))
    else:
        y = y.astype('int')
        clf = RandomForestClassifier(random_state=random_state).fit(X,y)
        print('Roc_auc_score for',model_name,round(roc_auc_score(y, clf.predict_proba(X),multi_class='ovr'),3))

def Evaluation_regr(X,y,model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                        random_state=random_state,stratify=y)   
    regr = RandomForestRegressor(random_state=random_state).fit(X_train,y_train)
    print('RMSE_score for',model_name, round(np.sqrt(metrics.mean_squared_error(y_test, regr.predict(X_test))),3))       

def Imputers_V2(dfx):
    global random_state,scaler,imp_cat,dfx_imputed_MF,max_iter
    random_state = 11
    max_iter = 10
    n_neighbors = [5,7,9] #3, 5,9,7
    print("\n Select Classification if the dataset has Mixed features (Both Numerical and Categorical) \n Select Regression if dataset has only Numerical features \n")
    print(" 1 Classification \n 2 Regression \n")
    opt = int(input('Enter the option:'))
    
    if opt ==1:
        print(pd.DataFrame(dfx.columns))
        target = int(input("Enter the index of the target variable :"))
        X = dfx.drop(dfx.columns[target],axis=1).columns
        y = pd.DataFrame(dfx.iloc[:,target]).columns #dfx.columns[target]
        print("\n-------- Evaluation scores for different Imputers---------\n")  
        scaler = MinMaxScaler()
        dfx_num = pd.DataFrame(scaler.fit_transform(dfx[num_columns]), columns = dfx[num_columns].columns)
        
        imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        dfx_cat_imputed = pd.DataFrame(imp_cat.fit_transform(dfx[cat_columns]), columns = dfx[cat_columns].columns)
        for k in n_neighbors:
            KNN_model = KNNImputer(n_neighbors=k)
            dfx_copy = dfx.copy()
            dfx_num_imputed = pd.DataFrame(KNN_model.fit_transform(dfx_num),columns=dfx_num.columns)   
            dfx_imputed = pd.concat([dfx_num_imputed,dfx_cat_imputed],axis=1)
            dfx_imputed_X = dfx_imputed[X]
            dfx_imputed_y = dfx_imputed[y]
            model_name = f"KNN_Imputer:(k={k})"
            Evaluation_clf(dfx_imputed_X,dfx_imputed_y,model_name)   
            
        Iterative_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,random_state=random_state,imputation_order='random')
        dfx_imputed_II = pd.DataFrame(Iterative_imputer.fit_transform(dfx),columns=dfx_copy.columns)
        dfx_imputed_X_II = dfx_imputed_II[X]
        dfx_imputed_y_II = dfx_imputed_II[y]
        model_name = "Iterative_Imputer"
        Evaluation_clf(dfx_imputed_X_II,dfx_imputed_y_II,model_name)  

        #Miss Forest
        MissForest_imputer = MissForest(max_iter=max_iter,random_state=random_state)
        dfx_imputed_MF = pd.DataFrame(MissForest_imputer.fit_transform(dfx),columns=dfx_copy.columns)
        dfx_imputed_X_MF = dfx_imputed_MF[X]
        dfx_imputed_y_MF = dfx_imputed_MF[y]
        model_name = "Miss_Forest_Imputer"
        Evaluation_clf(dfx_imputed_X_MF,dfx_imputed_y_MF,model_name)  

    if opt==2:
        print(pd.DataFrame(dfx.columns))
        target = int(input("Enter the index of the target variable :"))
        X = dfx.drop(dfx.columns[target],axis=1).columns
        y = pd.DataFrame(dfx.iloc[:,target]).columns #dfx.columns[target]
        print("\n-------- Evaluation scores for different Imputers---------\n")     
        for k in n_neighbors:
            KNN_model = KNNImputer(n_neighbors=k)
            dfx_copy = dfx.copy()
            dfx_imputed = pd.DataFrame(KNN_model.fit_transform(dfx),columns=dfx_copy.columns)        
            dfx_imputed_X = dfx_imputed[X]
            dfx_imputed_y = dfx_imputed[y]
            model_name = f"KNN_Imputer:(k={k})"
            Evaluation_regr(dfx_imputed_X,dfx_imputed_y,model_name)   
            
        Iterative_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,random_state=random_state,imputation_order='random')
        dfx_imputed_II = pd.DataFrame(Iterative_imputer.fit_transform(dfx),columns=dfx.columns)
        dfx_imputed_X_II = dfx_imputed_II[X]
        dfx_imputed_y_II = dfx_imputed_II[y]
        model_name = "Iterative_Imputer"
        Evaluation_regr(dfx_imputed_X_II,dfx_imputed_y_II,model_name) 
        
        MissForest_imputer = MissForest(max_iter=max_iter,random_state=random_state)
        dfx_imputed_MF = pd.DataFrame(MissForest_imputer.fit_transform(dfx),columns=dfx.columns)
        dfx_imputed_X_MF = dfx_imputed_MF[X]
        dfx_imputed_y_MF = dfx_imputed_MF[y]
        model_name = "Miss_Forest_Imputer"
        Evaluation_regr(dfx_imputed_X_MF,dfx_imputed_y_MF,model_name)

def Evaluation_univariate_ts(dfx,model_name):
    train_df, test_df = dfx[:-10], dfx[-10:]
    steps = len(test_df)
    model = auto_arima(train_df, start_p=0, start_q=0,d=1,max_p=3, max_q=3,start_P=0,start_Q=0,
                       max_P=3,max_Q=3,D=1,
                       stationarity=True,random_state=11)
    forecast = model.predict(n_periods=steps)
    rmse = np.sqrt(metrics.mean_squared_error(test_df,forecast))
    print('RMSE for the',model_name,rmse)
    
def Evaluation_multivariate_ts(dfx,model_name):   
    train_df, test_df = dfx[:-10], dfx[-10:]
    steps = len(test_df)
    vecm_model = vecm.VECM(endog = train_df, coint_rank = 3, deterministic = 'cili')
    vecm_fit = vecm_model.fit()
    vecm_test_forecast = vecm_fit.predict(steps)
    rmse_vecm = []
    
    for i in range(test_df.shape[1]):
        rmse_dummy=np.sqrt(metrics.mean_squared_error(test_df.iloc[:,i],vecm_test_forecast[:,i].round(2)))
        rmse_vecm.append(round(rmse_dummy,3))
    print('RMSE for the',model_name,rmse_vecm)
    
def time_univariate_imputers(dfx): 
    global dfx_imputed_MF,max_iter,random_state,n_neighbors
    max_iter = 10
    random_state=11
    n_neighbors = [7,9]
    for k in n_neighbors:
        KNN_model = KNNImputer(n_neighbors=k)
        dfx_imputed = pd.DataFrame(KNN_model.fit_transform(dfx),columns=dfx.columns) 
        model_name = f"KNN_Imputer:(k={k})"
        Evaluation_univariate_ts(dfx_imputed,model_name)
        
    Iterative_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,random_state=random_state,imputation_order='random')
    dfx_imputed_II = pd.DataFrame(Iterative_imputer.fit_transform(dfx),columns=dfx.columns)
    model_name = "Iterative_Imputer"
    Evaluation_univariate_ts(dfx_imputed_II,model_name) 
    
    #MissForest_imputer = MissForest(max_iter=max_iter,random_state=random_state)
    #dfx_imputed_MF = pd.DataFrame(MissForest_imputer.fit_transform(dfx),columns=dfx.columns)
    #model_name = "Miss_Forest_Imputer"
    #Evaluation_univariate_ts(dfx_imputed_MF,model_name)
        
def time_multivariate_imputers(dfx):
    global dfx_imputed_MF,max_iter,random_state,n_neighbors
    max_iter = 10
    random_state=11
    n_neighbors = [5,7,9]
    print("\n-------- Evaluation scores for different Imputers---------\n ")
    for k in n_neighbors:
        KNN_model = KNNImputer(n_neighbors=k)
        dfx_imputed = pd.DataFrame(KNN_model.fit_transform(dfx),columns=dfx.columns) 
        model_name = f"KNN_Imputer:(k={k})"
        Evaluation_multivariate_ts(dfx_imputed,model_name)
        
    Iterative_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,random_state=random_state,imputation_order='random')
    dfx_imputed_II = pd.DataFrame(Iterative_imputer.fit_transform(dfx),columns=dfx.columns)
    model_name = "Iterative_Imputer"
    Evaluation_multivariate_ts(dfx_imputed_II,model_name) 
    
    MissForest_imputer = MissForest(max_iter=max_iter,random_state=random_state)
    dfx_imputed_MF = pd.DataFrame(MissForest_imputer.fit_transform(dfx),columns=dfx.columns)
    model_name = "Miss_Forest_Imputer"
    Evaluation_multivariate_ts(dfx_imputed_MF,model_name)
    
def Imputers(X,y):
    cv = 5
    scoring = "roc_auc" #"accuracy"
    random_state = 11
    forest = RandomForestClassifier()
    #Store IterativeImputer scores
    strategies = ['ascending', 'random','arabic']# 'roman', 'arabic','random'
    for s in strategies:#ExtraTreesRegressor() 
        pipe = make_pipeline(IterativeImputer(estimator=BayesianRidge(), random_state=random_state,imputation_order=s), forest)
        ii_scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv)
        print(f"Iterative Imputer: Imputation order - {s}:",round(ii_scores.mean(),2))
        print(" ")
    # Store KNNImputer scores
    n_neighbors = [7] #2, 3, 5,9,7
    for k in n_neighbors:
        pipe = make_pipeline(KNNImputer(n_neighbors=k), forest)
        knn_scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv)
        print(f"KNN(k = {k}):",round(knn_scores.mean(),2))
        #print('KNN Imputer:',round(knn_scores.mean(),2))
        
    #final_scores = pd.concat([ii_scores, knn_scores],axis=1,keys=["iterative_imputer", "knn_imputer"])
    #print(final_scores.mean())
   
def Imputation(dfx):
    choice=str(input("\nProceed to Impute the data? YES/NO : "))
    if(choice.upper()=='YES'):        
        print('List of Imputers: \n 1 KNNImputer \n 2 IterativeImputer \n 3 MissForest')
        model = int(input('Enter the index of the Imputer:'))
        if model ==1:    
            model_param = int(input('Enter the value for n_neighbors (5 or 7 or 9):'))
            KNN_imputer = KNNImputer(n_neighbors=model_param)
            scaler = MinMaxScaler()
            imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            df_num = pd.DataFrame(scaler.fit_transform(dfx[num_columns]), columns = dfx[num_columns].columns)
            df_num_imputed = pd.DataFrame(KNN_imputer.fit_transform(df_num),columns=df_num.columns)
            df_num_imputed = pd.DataFrame(scaler.inverse_transform(df_num_imputed),columns=df_num.columns)
            df_cat_imputed = pd.DataFrame(imp_cat.fit_transform(dfx[cat_columns]), columns = dfx[cat_columns].columns)
            dfx_imputed = pd.concat([df_num_imputed,df_cat_imputed],axis=1)
            dfx_imputed[cat_columns] = np.round(dfx_imputed[cat_columns])
            
        if model ==2:
            imp_order = str(input('Enter the imputation order (descending or ascending or random):'))
            Iterative_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,random_state=random_state,imputation_order=imp_order)
            #df_imputed = df.copy()
            dfx_imputed = pd.DataFrame(Iterative_imputer.fit_transform(dfx),columns=dfx.columns)   # check the output!!!!!  
            dfx_imputed[cat_columns] = np.round(dfx_imputed[cat_columns])
        if model ==3:
            dfx_imputed = dfx_imputed_MF        
            dfx_imputed[cat_columns] = np.round(dfx_imputed[cat_columns])
        
        print("\n------------Count of null values after Imputation----------\n",dfx_imputed.isnull().sum())
        
    else:
        n=int(input("\nEnter 1 for Main Menu "))
        if n==1:
            Start_Imputation()
        else: 
            exit()
        
    return dfx_imputed


def Ts_Imputation(dfx):
    choice=str(input("\nProceed to Impute the data? YES/NO : "))
    if(choice.upper()=='YES'):        
        print('List of Imputers: \n 1 KNNImputer \n 2 IterativeImputer \n 3 MissForest')
        model = int(input('Enter the index of the Imputer:'))
        if model ==1:    
            model_param = int(input('Enter the value for n_neighbors (5 or 7 or 9):'))
            KNN_imputer = KNNImputer(n_neighbors=model_param)
            scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(dfx), columns = dfx.columns)
            dfx_imputed = pd.DataFrame(KNN_imputer.fit_transform(df_scaled),columns=df_scaled.columns)
            dfx_imputed = pd.DataFrame(scaler.inverse_transform(dfx_imputed),columns=dfx_imputed.columns)
                       
        if model ==2:
            imp_order = str(input('Enter the imputation order (descending or ascending or random):'))
            Iterative_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=max_iter,random_state=random_state,imputation_order=imp_order)
            #df_imputed = df.copy()
            dfx_imputed = pd.DataFrame(Iterative_imputer.fit_transform(dfx),columns=dfx.columns)   # check the output!!!!!  
        if model ==3:
            dfx_imputed = dfx_imputed_MF        
        
        print("\n------------Count of null values after Imputation----------\n",dfx_imputed.isnull().sum())
        print("\n--------------Imputed Data Summary--------------\n")
        print(dfx_imputed.describe(include='all').T[['count','mean','min','max']])       
        print("\n--------------Imputed Data--------------\n")
        print(dfx_imputed)
        
    else:
        n=int(input("\nEnter 1 for Main Menu "))
        if n==1:
            Start_Imputation()
        else: 
            exit()        
    return dfx_imputed

   
def Feedback(dfx,dfx_imputed):
    dfx_imputed.to_csv("Imputed_data.csv",index=False)
    print("\n**************The missing values in the given dataset is imputed and downloaded! Check the data!! **************\n")
    print(" ")
    print("--------------Do you want to Exit or run the model with different parameters?--------------\n")#Feedback
    opt=int(input("Enter the Option(1. EXIT/n 2.RE-RUN): "))
    if(opt==1):
        print("Exit")
        Start_Imputation()
    else:
        print("\nInpute the dataset using different Imputers / Parameters\n")
        dfx_imputed =Imputation(dfx) 
        dfx_imputed = inverse_labels(dfx_imputed,d)
        dfx_imputed = pd.concat([df_org.iloc[:,Not_Req_columns],dfx_imputed],axis=1)
        print("\n--------------Imputed Data Summary--------------\n")
        print(dfx_imputed.describe(include='all').T[['count','top','freq','mean','max']])
        opt=str(input("\n Do you want to download the data? YES/NO : "))
        if(opt.upper()=='YES'):
            print("Downloaded successfully!")
            dfx_imputed.to_csv("Imputed_data.csv",index=False)
        else:
            dfx_imputed =Imputation(dfx) 
        
def Ts_Feedback(dfx,dfx_imputed):
    dfx_imputed.to_csv("Imputed_data.csv",index=False)
    print(" ")
    print("\n**************The missing values in the given dataset is imputed and downloaded! Check the data!! **************\n")
    print(" ")
    print("--------------Do you want to Exit or run the model with different parameters?--------------\n")#Feedback
    opt=int(input("Enter the Option(1. EXIT \n 2.RE-RUN): "))
    if(opt==1):
        print("Exit")
        Start_Imputation()
    else:
        print("\nInpute the dataset using different Imputers / Parameters\n")
        dfx_imputed =Ts_Imputation(dfx)              
        print("\n--------------Imputed Data Summary--------------\n")
        print(dfx_imputed.describe(include='all').T[['count','mean','min','max']])
        opt=str(input("\n Do you want to download the data? YES/NO : "))
        if(opt.upper()=='YES'):
            print("Downloaded successfully!")
            dfx_imputed.to_csv("Imputed_data.csv",index=False)
        else:
            dfx_imputed =Imputation(dfx) 
            
            
            
def set_date_as_index(dfx):
    print(pd.DataFrame(dfx.columns))
    print("\n Select the index of the Date column to set as index \n")
    date_idx = int(input("Enter the index:"))
    print(dfx.iloc[:,date_idx].name)
    dfx.set_index(dfx.iloc[:,date_idx].name,inplace=True)
    return dfx
    
def import_csv_data(csv_file_path):
    global df_org
    df = pd.read_excel(csv_file_path)
    df_org = df.copy()
    # Dimensions of the Data
    print("\n Total Rows:",df.shape[0],"\n Total Columns:",df.shape[1])
    Null_Values_display(df)
    df = DataPreprocessing(df)
    return df 

    
def Start_Imputation():
    root = tk.Tk()
    root.withdraw()
    print("-----------------------------------------------------------")
    print("....................Missing Values Imputation.....................")
    print(" 1 Non-Time Series Data \n 2 Time Series Data \n 3 Exit")
    print("-----------------------------------------------------------")
    opt=int(input("Enter the Option : "))
    if(opt==1):
        file_path = filedialog.askopenfilename()
        #print(file_path)
        df=import_csv_data(file_path)
        df = Select_features(df)    
        df = LabelEncoding(df)
        #Imputers(X,y) 
        #Imputers_V2(df)
        df_imputed =Imputation(df)
        df_imputed = inverse_labels(df_imputed,d)
        df_imputed = pd.concat([df_org.iloc[:,Not_Req_columns],df_imputed],axis=1)
        print("\n--------------Imputed Data Summary--------------\n")
        print(df_imputed.describe(include='all').T[['count','top','freq','mean','max']])
        print("\n--------------Imputed Data--------------\n")
        print(df_imputed)
        Feedback(df,df_imputed)                  
    elif(opt==2):
        file_path = filedialog.askopenfilename()
        df=import_csv_data(file_path)
        df = Select_features(df)
        df= set_date_as_index(df)       
        print("\n Is the imported Time series data Univariate or Multivariate? \n ")
        print(" 1 Univariate \n 2 Multi-variate \n 3 Exit")
        opt=int(input("Enter the Option : "))
        if(opt==1):
            time_univariate_imputers(df)   
            df_imputed = Ts_Imputation(df) #Miss Forest will not work
            Ts_Feedback(df,df_imputed) 
        elif(opt==2):
            time_multivariate_imputers(df)
            df_imputed= Ts_Imputation(df)
            Ts_Feedback(df,df_imputed) 
        else:
            print("Exit")
            Start_Imputation()
            
    else:
        print("Exit")
        Start_Imputation()
   
Start_Imputation()
