#%%
'''1st of all we have HR Employees dataset  
in this Project i will brakdown the dataset through 
analysing the data seeking to predict the reasone or 
the causeing factor of the employees attrition 
Education
1 'Below College'
2 'College'
3 'Bachelor'
4 'Master'
5 'Doctor'

EnvironmentSatisfaction
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

JobInvolvement
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

JobSatisfaction
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

PerformanceRating
1 'Low'
2 'Good'
3 'Excellent'
4 'Outstanding'

RelationshipSatisfaction
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

WorkLifeBalance
1 'Bad'
2 'Good'
3 'Better'
4 'Best'
'''
#%%

# import libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# read the dataset 
df= pd.read_csv('C:/Users/LENOVO/Desktop/internship/HR-Employee.csv')
df.head()

#EDA Exploration Data Analysis
df.shape # 1470 Raws with 35 columns (factors)
nullValues = df.isnull().sum().sum()#EDA : is to identify the pattterns through different data visualization
nullValues #No null values in this dataset
duplicatedValues= df.duplicated().sum()
duplicatedValues# No duplcated values in this dataset

#compare btw count of employees who have attririon and who'r not
categorical_count = df['Attrition'].value_counts().to_frame()
categorical_count # by value_counts function i breakdown the Attrition column into two values Yes and No to countvalues for each

sns.set_style('dark')
sns.countplot(x='Attrition',data=df) #Initial plot to compare btw yes and no who are left the company and who'r not. So we realized tht who were stayed more than who were left

#with pie chart
plt.title('Attrition Distribution')
df['Attrition'].value_counts().plot.pie(autopct='%1.2f%%')
#Initial Piechart we can see that 16.12% out of 83.88% of the employees who were left 

# we can seperte each of yes or no in seperated dataframe
Attrition_yes = df[df['Attrition']=='Yes']
Attrition_no = df[df['Attrition']=='No']
Attrition_yes.shape #we can see that 237 employee had left the company while 1233 who are not

# No we should moved into the visualization and see the correlation btw the factors, and that must give us a clear insights about the independents variables(factors)
#be4 that i will start with label the categorical columns  with numbers
# let us see the datatype 1st
df.dtypes #Attrition, BusinessTravel, Department, EducationField, Gender, JobeRole, MaritalStatus, Over18, OverTime

df = df.replace(to_replace = ['Yes','No'],value = ['1','0'])
df = df.replace(to_replace = ['Travel_Rarely',
'Travel_Frequently','Non-Travel'],value = ['2','1','0'])
df = df.replace(to_replace = ['Married','Single','Divorced'],value = ['2','1','0'])
df = df.replace(to_replace = ['Male','Female'],value = ['1','0'])
#---
df = df.replace(to_replace = ['Human Resources','Research & Development','Sales'],value = ['0','1','2'])
df = df.replace(to_replace = ['Human Resources','Life Sciences','Marketing','Medical','Technical Degree','Other'],value = ['0','1','2','3','4','5'])
df = df.replace(to_replace = ['Healthcare Representative','Human Resources','Laboratory Technician','Manager','Manufacturing Director','Research Director','Research Scientist','Sales Executive','Sales Representative'],value = [0,1,2,3,4,5,6,7,8])

df.head(5) #check the dataset again

#DistanceFromHome and Attrition by histogram with boxplot
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='DistanceFromHome', hue='Attrition',kde='True')
plt.title('Age Distribution by Attrition')
plt.show() #Here we can see the left employees have small distance numbers from home and most of them their home very close from the company with numbers histated with around 0-10 Klg  

#
sns.distplot(df.loc[df['Attrition']=='1']['Education'])
sns.distplot(df.loc[df['Attrition']=='0']['Education']); #We can realised from this distribution plot that most of employees had Bachelor degree wether they were lefot or not, in the 2nd level they have Master degree

##


plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='JobRole', hue='MaritalStatus', palette='viridis')
plt.title('Attrition by Job Role')
plt.xticks(rotation=50)
plt.show()
''' we can see most married employees were worked 
in Sales Executive postition with more then 140 
person, then in the same position around 100 person 
were single and around 60 were divorced. Its clear 
that the top 3 positions played by most of employees
beside the Sales Executive were Research Scientist
and Lavoratory Technicuian while for the most
marital status for the employees was married then 
single and divorsed employees were shaped the lowest 
proprtion. Finally we can mentioned that the Humen
Resources position was the lowest role was played by
the employees '''

##

plt.figure(figsize=(6,12))
sns.boxplot(data=df, x='Attrition',y='JobSatisfaction',palette='coolwarm')
plt.title('Job Stasfaction Vs. Attrition')
plt.xticks([0,1],['No Attrition','Attrition'])
plt.show() 
''' we can see that ppl who stay the company 50% of 
them have medium to high level of satisfaction and
the rest have high to very high level of
sastisfaction. Then the left employees were all of
them had low to medium lecvel of satisfaction '''


#Creating bar plot
sns.barplot(x = 'JobSatisfaction',y = 'JobRole',data = df ,palette = "Blues")
#Adding the aesthetics
plt.title('Chart title')
plt.xlabel('Job Satisfaction')
plt.ylabel('Job Role') 
# Show the plot
plt.show()
''' we can see here part of workers in Humen Resources& workers in
Sales Representative and Sales Scientist employees were vary satisfied. '''

#Attrition by overtime work
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='OverTime', hue='Attrition', palette= 'Greens')
plt.title('Attrition by Overtime Work')
plt.show()
''' We can noticed that between 200-400 left ppl
were having overtime work, while arund 100 dont have.
'''
# Attrition by Marital status and Age
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='MaritalStatus',y='Age',hue='Attrition', palette='coolwarm')
plt.title('Attrition by Age and Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Age')
plt.xticks(rotation=90)
plt.show()

#
features= ['YearsSinceLastPromotion','YearsInCurrentRole', 'WorkLifeBalance','RelationshipSatisfaction','NumCompaniesWorked','JobLevel']
fig= plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j,data=df)
    plt.xticks(rotation=90)
    plt.title('No, of employee')


fig= plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j,data=df, hue='Attrition')
    plt.xticks(rotation=90)
    plt.title('No, of employee')
    
    
# drop unnecessery columns
DF = df.drop(['EmployeeCount','Over18','StandardHours'])
# Let's see the information of our updated dataset DF
DF.info()
''' This dataset had 1470 samples and 32 attributes,
(24 integer + 8 objects ) No variables have non null/
missing values'''

DF.describe()
# train the model

X= DF[['BusinessTravel','Department','DistanceFromHome','HourlyRate' ,'JobLevel', 'JobRole', 'JobSatisfaction', 'MonthlyIncome', 'OverTime','TotalWorkingYears','WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion']]
y= DF['Attrition']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y ,test_size=0.3, random_state=42, )
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

#GB
Model1= GradientBoostingClassifier()
Model1.fit(x_train,y_train)
gb_y_pred= Model1.predict(x_test)

#RF
Model2= RandomForestClassifier()
Model2.fit(x_train,y_train)
rf_y_pred = Model2.predict(x_test)

#SVM

Model3=SVC()
Model3.fit(x_train,y_train)
svc_y_pred= Model3.predict(x_test)


#Evaluating
from sklearn.metrics import accuracy_score, precision_score, recall_score
Model1= GradientBoostingClassifier()
Model1.fit(x_train,y_train)
gb_y_pred= Model1.predict(x_test)

Models ={
    'GradientBoostingClassifier' : gb_y_pred,
    'RandomForestClassifier' : rf_y_pred,
    'SVM' : svc_y_pred
}

models= pd.DataFrame(Models)

for i in models:
    acc= accuracy_score(y_test, models[i])
    prec= precision_score(y_test, models[i],pos_label='1')
    recall= recall_score(y_test, models[i], pos_label='1')
    results= pd.DataFrame([[i,acc,prec,recall]],
                          columns = ['model','accuracy','precision','recall' ])
    print(results)
    
    
'''
x_train = scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)




groupset=groupset.drop(['EducationField','JobRole','Over18'],axis=1)
groupset.groupby('Department').mean()['Attrition'].plot(kind='bar',color=['Green','Blue','Pink'])
plt.title('Attrition Rate by Department')'''
