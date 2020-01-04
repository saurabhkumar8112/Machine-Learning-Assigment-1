import pandas as pd
import numpy as np
import MLE

df = pd.read_csv('railway_booking/railwayBookingList.csv')
df_train = df.sample(frac = 0.8)

df_0 = df_train.loc[df['boarded'] == 0]
df_1 = df_train.loc[df['boarded'] == 1]

df_0 = df_0[['budget', 'memberCount', 'preferredClass', 'sex', 'age']]
df_1 = df_1[['budget', 'memberCount', 'preferredClass', 'sex', 'age']]

array_0 = np.array(df_0)
array_1 = np.array(df_1)

budget_0 = np.reshape(array_0[:, 0], (1, -1))
budget_1 = np.reshape(array_1[:, 0], (1, -1))

parameters_0 = []
parameters_1 = []

parameters_0.append(MLE.multivariate_gaussian_independent(budget_0))
parameters_1.append(MLE.multivariate_gaussian_independent(budget_1))

memberCount_0 = np.zeros(11)
memberCount_1 = np.zeros(11)

for i in range(11):
    df_0_i = df_0.loc[df['memberCount'] == i]
    df_1_i = df_1.loc[df['memberCount'] == i]

    memberCount_0[i] = float(df_0_i.shape[0]/np.shape(array_0)[0])
    memberCount_1[i] = float(df_1_i.shape[0]/np.shape(array_1)[0])

    parameters_0.append(memberCount_0)
    parameters_1.append(memberCount_1)

preferred_0 = np.zeros(3)
preferred_1 = np.zeros(3)

for i in range(3):
    df_0_i = df_0.loc[df['preferredClass'] == i+1]
    df_1_i = df_1.loc[df['preferredClass'] == i+1]

    preferred_0[i] = float(df_0_i.shape[0]/np.shape(array_0)[0])
    preferred_1[i] = float(df_1_i.shape[0]/np.shape(array_1)[0])

    parameters_0.append(preferred_0)
    parameters_1.append(preferred_1)

sex_0 = np.zeros(2)
sex_1 = np.zeros(2)

for i in range(2):
    df_0_i = df_0.loc[df['sex'] == i]
    df_1_i = df_1.loc[df['sex'] == i]

    sex_0[i] = float(df_0_i.shape[0]/np.shape(array_0)[0])
    sex_0[i] = float(df_1_i.shape[0]/np.shape(array_1)[0])

    parameters_0.append(sex_0)
    parameters_1.append(sex_1)

age_0 = np.zeros(9)
age_1 = np.zeros(9)

for i in range(9):
    df_0_i = df_0.loc[df['age'] == i]
    df_1_i = df_1.loc[df['age'] == i]

    age_0[i] = float(df_0_i.shape[0]/np.shape(array_0)[0])
    age_1[i] = float(df_1_i.shape[0]/np.shape(array_1)[0])

    parameters_0.append(age_0)
    parameters_1.append(age_1)

prior = np.zeros(2)
prior[0] = np.shape(array_0)[0]
prior[1] = np.shape(array_1)[0]

df_test = df.sample(frac = 0.2)
df_test = df_test[['boarded', 'budget', 'memberCount', 'preferredClass', 'sex', 'age']]
test = np.array(df_test)
correct = 0
true_pos = 0
false_neg = 0
false_pos = 0

for i in range(np.shape(test)[0]):
    pdf_0 = MLE.multivariate_gaussian_independent_pdf(test[i][1], parameters_0[0]) * parameters_0[1][test[i][2]] * parameters_0[2][test[i][3]] * parameters_0[3][test[i][4]] * parameters_0[4][test[i][5]]
    pdf_1 = MLE.multivariate_gaussian_independent_pdf(test[i][1], parameters_1[0]) * parameters_1[1][test[i][2]] * parameters_1[2][test[i][3]] * parameters_1[3][test[i][4]] * parameters_1[4][test[i][5]]

    post_0 = prior[0] * pdf_0
    post_1 = prior[1] * pdf_1

    if post_0 > post_1:
        output = 0
    else:
        output = 1

    if output == 1 and test[i][0] == 1:
        true_pos += 1

    if output == 0 and test[i][0] == 1:
        false_neg += 1

    if output == 1 and test[i][0] == 0:
        false_pos += 1

    if output == test[i][0]:
        correct += 1

accuracy = float(correct/np.shape(test)[0]*100)
precision = float(true_pos/(true_pos + false_pos)*100)
recall = float(true_pos/(true_pos + false_neg)*100)

print("Accuracy = " + str(accuracy))
print("Precision = " + str(precision))
print("Recall = " + str(recall))









