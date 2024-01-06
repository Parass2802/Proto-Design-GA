#Side Profile optimization

import matplotlib.pyplot as plt
from ypstruct import structure
import GA
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from smt.surrogate_models import KRG

#Loading data
df=pd.read_csv(r"E:\ML\Optimization techniques\Full code surrogate plus opti\proto\side_profile_data.txt",delimiter='\t',header=None)

X=df.drop([3],axis=1)
y=df[3]

#Normalizing Data
'''
scaler = MinMaxScaler()
X= scaler.fit_transform(X)
'''
X[0]=X[0]/0.6
X[1]=X[1]/0.6
X[2]=X[2]/0.6

#Splitting Data
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.2)


#Kriging Surrogate model
sm=KRG(theta0=[10])
sm.set_training_values(np.array(X_train),np.array(y_train).reshape(-1,1))
sm.train()

y_predicted=sm.predict_values(np.array(X_test))

# Calculate R-squared score for the predictions
r2 = r2_score(y_test, y_predicted)
print(f"R-squared score: {r2}")


#plotting r2 score
plt.figure(figsize=(11,7),dpi=300)
plt.plot(y_test,y_test,color='black')
plt.scatter(y_predicted,y_test,color='red',marker='o')
plt.xlabel('Predicted $C_d [-]$')
plt.ylabel('Actual $C_d [-]$')
plt.title(f'Kriging surrogate model, $R^2$ score={r2:.2f}')



def proto_GA(x):
    X_GA=np.array(x)
    X_GA=X_GA.reshape(-1,3)
    X_GA[0][0]=X_GA[0][0]/0.6
    X_GA[0][1]=X_GA[0][1]/0.6
    X_GA[0][2]=X_GA[0][2]/0.6

    y_predicted_GA=sm.predict_values(X_GA)
    
    return y_predicted_GA[0][0]

# Problem Definition
problem = structure()
problem.costfunc = proto_GA
problem.nvar = 3
problem.varmin = [0.2, 0.35, 0.2]
problem.varmax = [0.6, 0.6, 0.6]

# GA Parameters
params = structure()
params.maxit = 100
params.npop = 50         #number of chromosomes / population size
params.beta = 1
params.pc = 1
params.gamma = 0.1
params.mu = 0.1
params.sigma = 0.1

# Run GA
out = GA.run(problem, params)

# Results

lw=2
plt.figure(figsize=(11,7),dpi=600)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 2

plt.tick_params(axis='both',size=8,labelsize=17,direction='inout')
plt.xlim(0, params.maxit)
plt.plot(out.bestcost)
plt.xlabel('Iterations',fontsize=20)
plt.ylabel('Best Cost',fontsize=20)
plt.title('Genetic Algorithm (GA)',fontsize=24)
plt.grid(True)
#optimum values
print('Optimum paramters values:', str(out.bestsol))
