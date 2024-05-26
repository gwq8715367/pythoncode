import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score  
  
# 读取CSV文件  
data = pd.read_csv('iris.data')
# 假设CSV文件的列名未知，我们直接通过列索引来获取特征和目标  
# 最后一列作为目标列  
y = data.iloc[:, -1]  # 使用iloc通过整数位置索引来选择最后一列  
  
# 除了最后一列的所有列作为特征列  
X = data.iloc[:, :-1]  # :-1表示从第一列开始到倒数第二列结束  
  
# 如果CSV文件有列名，并且你想要基于列名来选择目标列和特征列  
# 假设最后一列的列名是'target'  
# y = data['target']  
# X = data.drop('target', axis=1)  
  
# 划分数据集为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
# 创建决策树分类器对象  
clf = DecisionTreeClassifier()  
  
# 训练决策树模型  
clf.fit(X_train, y_train)  
  
# 使用训练好的模型对测试集进行预测  
y_pred = clf.predict(X_test)  
  
# 评估模型性能  
accuracy = accuracy_score(y_test, y_pred)  
print(f'Accuracy: {accuracy}')  
  
# 如果需要，你可以进一步保存模型或进行其他操作  
# 例如，使用joblib保存模型：  
# from sklearn.externals import joblib  
# joblib.dump(clf, 'decision_tree_model.joblib')  
  
# 注意：在较新版本的scikit-learn中，'sklearn.externals.joblib'已被弃用  
# 应该直接使用 'joblib' 包：  
# import joblib  
# joblib.dump(clf, 'decision_tree_model.joblib')