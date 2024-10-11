import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据集
train_data = pd.read_csv('Bayesian_Dataset_train.csv', header=None)
test_data = pd.read_csv('Bayesian_Dataset_test.csv', header=None)

# 2. 添加列名
columns = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Marital_Status', 'Occupation',
           'Relationship', 'Race', 'Sex', 'Native_Country', 'Income']
train_data.columns = columns
test_data.columns = columns

# 3. 数据预处理
# 3.1 替换缺失值标记为 NaN（假设缺失值用 '?' 表示）
train_data.replace(' ?', np.nan, inplace=True)
test_data.replace(' ?', np.nan, inplace=True)

# 3.2 删除含有缺失值的行
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# 3.3 编码分类变量
label_encoders = {}
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train_data[column].values) + list(test_data[column].values))
        train_data[column] = le.transform(train_data[column])
        test_data[column] = le.transform(test_data[column])
        label_encoders[column] = le

# 4. 特征和标签
X_train = train_data.drop('Income', axis=1)
y_train = train_data['Income']
X_test = test_data.drop('Income', axis=1)
y_test = test_data['Income']

# 5. 模型训练
model = CategoricalNB()
model.fit(X_train, y_train)

# 6. 模型预测
y_pred = model.predict(X_test)

# 7. 模型评估
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# 8. 结果保存
# 8.1 将数值标签转换回原始标签
income_le = label_encoders['Income']
test_data['Predicted_Income'] = income_le.inverse_transform(y_pred)

# 8.2 保存结果到指定文件
output_filename = f'result_precision={precision:.2f}_recall={recall:.2f}_F1 score={f1:.2f}.csv'
test_data.to_csv(output_filename, index=False)
print(f'\n结果已保存到文件：{output_filename}')

# 9. 可视化结果
# 9.1 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=income_le.classes_, yticklabels=income_le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 10. 参数影响探索
# 10.1 调整平滑参数 alpha 的影响，并分别绘制各类别的指标曲线

alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0,
6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0,
7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0,
8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0,
9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0,
10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0,
11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0,
12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8
]

# 初始化用于存储各类别指标的列表
precisions_class0 = []
precisions_class1 = []
recalls_class0 = []
recalls_class1 = []
f1_scores_class0 = []
f1_scores_class1 = []

for alpha in alpha_values:
    model = CategoricalNB(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred_alpha = model.predict(X_test)
    report = classification_report(y_test, y_pred_alpha, output_dict=True)
    # 假设类别标签为 0（<=50K）和 1（>50K）
    precisions_class0.append(report['0']['precision'])
    precisions_class1.append(report['1']['precision'])
    recalls_class0.append(report['0']['recall'])
    recalls_class1.append(report['1']['recall'])
    f1_scores_class0.append(report['0']['f1-score'])
    f1_scores_class1.append(report['1']['f1-score'])

# 绘制精确率曲线
plt.figure(figsize=(8, 6))
plt.plot(alpha_values, precisions_class0, marker='o', label='<=50K Precision')
plt.plot(alpha_values, precisions_class1, marker='o', label='>50K Precision')
plt.title('Effect of Alpha on Precision')
plt.xlabel('Alpha')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()

# 绘制召回率曲线
plt.figure(figsize=(8, 6))
plt.plot(alpha_values, recalls_class0, marker='o', label='<=50K Recall')
plt.plot(alpha_values, recalls_class1, marker='o', label='>50K Recall')
plt.title('Effect of Alpha on Recall')
plt.xlabel('Alpha')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
plt.show()

# 绘制 F1 分数曲线
plt.figure(figsize=(8, 6))
plt.plot(alpha_values, f1_scores_class0, marker='o', label='<=50K F1-score')
plt.plot(alpha_values, f1_scores_class1, marker='o', label='>50K F1-score')
plt.title('Effect of Alpha on F1 Score')
plt.xlabel('Alpha')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.show()
