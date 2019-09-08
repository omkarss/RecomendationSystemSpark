
import pandas as pd
import numpy as np

R = pd.DataFrame(np.array([[5,-1,6],[4,6,0],[6,4,-1]]))

P = pd.DataFrame(np.array(np.random.rand(len(R.index),2)))

Q = pd.DataFrame(np.array(np.random.rand(2,len(R.columns))))


alpha = 0.001
lamda = 0.02

for count in range(0,10):
    for i in R.index:
        for j in R.columns:
            if(R.loc[i,j] >0):
            # Calculate error 
                error = R.loc[i][j] - np.dot(P.iloc[i],Q[j])
        
            # Derivative of P[i,k]
            
                for k in range(0,2):
                    dpik = -2*(error*Q.iloc[k,j] - lamda*P.iloc[i,k])
        
            # Derivative of Q[j,k]
        
                    dqjk = -2*(error*P.iloc[i,k] - lamda*Q.iloc[k,j])
        
            # Updating the P matrix
                    P.iloc[i,k] = P.iloc[i,k] - alpha*dpik
        
            # Updating the Q Matrix     
                    Q.iloc[k,j] = Q.iloc[k,j] - alpha*dqjk       
    
        e=0
        total_error = 0
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    #Sum of squares of the errors in the rating
                    #temp = pow((A.loc[i,j] - np.dot(P.iloc[i],Q[j])),2)
                    e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
                    #total_error = total_error + pow(temp,2) +lamda*((np.linalg.norm(P.loc[i])*np.linalg.norm(P.loc[i])) + (np.linalg.norm(Q.loc[j])*np.linalg.norm(Q.loc[j]))) 

        if e<0.001:
            break
        
                