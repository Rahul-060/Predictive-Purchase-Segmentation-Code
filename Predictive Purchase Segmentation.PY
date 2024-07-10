'''A retail company wants to predict customer purchasing behavior based on their age, salary and past purchase history. The company aims to use K-Nearest Neighbors (KNN) algorithm to classify customers into potential buying groups to personalize marketing strategies.
This predictive model will help the company understand and target specific customer segments more effectively, thereby increasing sales and customer satisfaction.'''

# We will use KNN because of classification and on those classification company will built model

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# step-2 - form data
data=np.array([[25,50000,2],[30,80000,1],[35,60000,3],[20,30000,2],[40,90000,1],[45,75000,2]])
labels=np.array([1,2,1,0,2,1]) # 0:low, 1:medium, 2:high


# step-3 - train and test
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=42)

# step-4 - overwritting
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

# step-5 - telling model which nearest meighbour it has to pick
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

# In KNN accuracy score is already inbuilt and don't have to export 
accuracy=knn.score(X_test,y_test)
print(f"Accuracy of model is : {accuracy}")

# user input
user_input = np.array([[25, 30000, 3]])
user_input_scaled = scaler.transform(user_input)
prediction=knn.predict(user_input_scaled)
print(prediction)


