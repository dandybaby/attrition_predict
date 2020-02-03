import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# 数据准备
full = train.append(test, ignore_index=True)
BusinessTravelDF = pd.get_dummies(full['BusinessTravel'], prefix='BusinessTravel')
DepartmentDF = pd.get_dummies(full['Department'], prefix='Department')
EducationFieldDF = pd.get_dummies(full['EducationField'], prefix='EducationField')
JobRoleDF = pd.get_dummies(full['JobRole'], prefix='JobRole')
MaritalStatusDF = pd.get_dummies(full['MaritalStatus'], prefix='MaritalStatus')
GenderDF = pd.get_dummies(full['Gender'], prefix='Gender')

Over18_Dict = {'Y': 1}
full['Over18'] = full['Over18'].map(Over18_Dict)
OverTimeDF = pd.get_dummies(full['OverTime'], prefix='OverTime')

Attrition_Dict = {'Yes': 1, 'No': 0}
full['Attrition'] = full['Attrition'].map(Attrition_Dict)
full = pd.concat([full, BusinessTravelDF, DepartmentDF, EducationFieldDF,
                  JobRoleDF, MaritalStatusDF, GenderDF, OverTimeDF], axis=1)
full.drop(['BusinessTravel', 'Department', 'EducationField', 'JobRole',
           'MaritalStatus', 'Gender', 'OverTime'], axis=1, inplace=True)
# 获取特征相关性
# corrDF = full.corr()
# print(corrDF['Attrition'].sort_values(ascending=False))
#
# full2 = pd.concat([BusinessTravelDF, DepartmentDF, EducationFieldDF,
#                    JobRoleDF, MaritalStatusDF, GenderDF, OverTimeDF,
#                    full['DistanceFromHome'], full['NumCompaniesWorked'], full['MonthlyRate']], axis=1)
# # 样本特征数据
full2 = full.drop(['Attrition'], axis=1)
data_x = full2.loc[0:1175, :]
# 样本标签数据
data_y = full.loc[0:1175, 'Attrition']
# 预测特征数据
pre_data = full2.loc[1176:, :]
# 根据样本数据拆分训练集
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size=0.8)


# 训练逻辑回归模型
model_lr = LogisticRegression(max_iter=7500)
model_lr.fit(train_x, train_y)
print(model_lr.score(test_x, test_y))

pre_attrition = model_lr.predict_proba(pre_data)[:, 1]
# 提取数据
user_ID = full.loc[1176:, 'user_id']
pre_result = pd.DataFrame({'user_id': user_ID, 'Attrition': pre_attrition})
pre_result.to_csv('pre_attrition2.csv', index=False)

# # 随机森林模型
# # model_rf = RandomForestClassifier()
# # model_rf.fit(train_x, train_y)
# # print(model_rf.score(test_x, test_y))
#
# # kfold = model_selection.KFold(n_splits=10, random_state=7)
# # modelCV = RandomForestClassifier()
# # scoring = 'accuracy'
# # results = model_selection.cross_val_score(modelCV, train_x, train_y, cv=kfold, scoring=scoring)
# # print("10次交叉验证平均精度：%.3f"% (results.mean()))
#
#
# # #支持向量机
# # model_svc = SVC()
# # model_svc.fit(train_x, train_y)
# # print(model_svc.score(test_x, test_y))
