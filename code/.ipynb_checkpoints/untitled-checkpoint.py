import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score  
  
# 读取CSV文件  
data = pd.read_csv('your_data.csv')  