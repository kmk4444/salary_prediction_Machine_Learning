# Baseball salary estimate

# Baseball with salary information and career statistics for 1986 Develop a machine learning model to estimate the salary of players.

#Importing Dataset and Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn import preprocessing

from warnings import filterwarnings
filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def load():
    data= pd.read_csv("/kaggle/input/hitters/hitters.csv")
    return data

df = load()

df.head()

df_=df.copy() #boş değerlere atadığım değerleri kontrol etmek için.

# Examine the data

def check_df(dataframe, head=5):
    print("############### shape #############")
    print(dataframe.shape)
    print("############### types #############")
    print(dataframe.dtypes)
    print("############### head #############")
    print(dataframe.head())
    print("############### tail #############")
    print(dataframe.tail())
    print("############### NA #############")
    print(dataframe.isnull().sum())
    print("############### Quantiles #############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

#Finding numeric and categorical variables.
def grab_col_names(df, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if
                str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int",
                                                                                              "float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ["category",
                                                                                                   "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

print(f"cat_cols: {cat_cols}")
print(f"num_cols: {num_cols}")
print(f"cat_but_car: {cat_but_car}")
print(f"num_but_cat: {num_but_cat}")

#Analyzing of the numeric and categorical variables.
def cat_summary(dataframe, col_name, plot=False):  # create plot graph
    if df[col_name].dtypes == "bool":
        df[col_name] = df[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:  # meaning that plot is true
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:  # meaning that plot is true
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Analyzing target variable. (meaning of the target variable by categorical variables, meaning of numerical variables by target variables)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN":dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df,"Salary",col)
print("#####################################")

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}),end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Salary", col)
print("#####################################")


# we check that league and new league are the same or not.
def league(League,NewLeague):
    i=0
    if League == NewLeague:
        i=i+1
    return i

league_check_df = pd.DataFrame(df.apply(lambda x: league(x["League"],x["NewLeague"]), axis=1))
column= ["check_league"]
league_check_df.columns = column
print(league_check_df.head(10))
league_check_df.value_counts()

#Outlier Analyzing
#How can we find this values ?

#industry knowledge
#standard deviation approach
#z-score approach
#boxplot(interquantile range -IQR) Method
# lof method => multiple variables method

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))


# Missing observation analysis.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    print("###############")

    if na_name:
        return na_columns


na_columns = missing_values_table(df, True)
print(f'na_columns: {na_columns}')


# missing values correlation
msno.heatmap(df)
plt.show()

#This code is not necessary, because target variables are missing.
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Salary", na_columns)

msno.matrix(df)
plt.show()

#Correlation Analysis.
corr = df[num_cols].corr()

f, ax = plt.subplots(figsize=[10, 10])
sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#Missing values solution
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

scaler = MinMaxScaler() # değerleri 1 ile 0 'a dönüştür yapmayı sağlıyor.
dff = pd.DataFrame(scaler.fit_transform(dff), columns = dff.columns) # formatı dataframe'e çeviriyoruz.
dff.head()

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5) # en yakın 5 komşu diyoruz.
dff = pd.DataFrame(imputer.fit_transform(dff), columns = dff.columns) # doldurduk otomatik olarak ama doldurduğum yerleri göremiyorum. Görmek istiyorum!
dff.head()

#kıyaslamak için eski haline dönüştürüyorum. (standartlaştırma öncesi)
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

dff.isnull().sum()

#we check missing values for salary.
df_[["Salary_knn"]] = dff[["Salary"]]

df_[df_.isnull().any(axis=1)]

#Local Outlier Factor

clf = LocalOutlierFactor(n_neighbors=20) # komşuluk sayısına 20 dedik ancak bu değişebilir
# ama burada komşuluk sayılarında hangisi iyidir yorumlayamıyoruz. bu nedenle 20 alın.
clf.fit_predict(dff)
df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-') #x gözlemler, y onların outlier scoreları.
plt.show()

th = np.sort(df_scores)[4]

dff[df_scores<th].shape

#drop lof values

index = df[df_scores<th].index

dff = dff.drop(index)

df[df_scores<th]

#we check delete operation
print(df.shape)
print(dff.shape)

#Correlation

def high_correlated_cols(dataframe, plot=False, corr_th=0.95):
    corr = dataframe.corr()
    cor_matrix = corr.abs() #positive
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(dff)
drop_list = high_correlated_cols(dff, plot=True)# all variables
print(drop_list)

dff = dff.drop(drop_list, axis = 1)

#Creating new variables

dff.head()
#1.
dff['NEW_AtBat'] = pd.qcut(dff['AtBat'], 3 ,labels=["low","normal","high"])
#2.
dff['NEW_CRUNS_CHMRUN_RATE'] = dff["CHmRun"] / dff["CAtBat"]
#3.
dff['NEW_YEARS'] = pd.qcut(dff['Years'], 3 ,labels=["young","normal","old"])
#4.
dff["NEW_CRuns_Assits"] = dff["Runs"] * dff["Assists"]

dff.head()

#Encoding

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dff)

print(f"cat_cols: {cat_cols}")
print(f"num_cols: {num_cols}")
print(f"cat_but_car: {cat_but_car}")
print(f"num_but_cat: {num_but_cat}")

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in dff.columns if dff[col].dtype not in [int, float]
               and dff[col].nunique() == 2] # boş değerleride saydığı için len(unique) almadık.

for col in binary_cols:
    label_encoder(dff, col)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(dff, "Salary", cat_cols) # dff yazdığımızda hata veriyor, bu yüzden yapmadım.

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy() # kopya alınmış.

    rare_columns = [col for col in temp_df.columns if str(temp_df[col].dtypes) in ["category","object"]and (temp_df[col].value_counts() / len(temp_df)<rare_perc).any(axis=None)] # kategorik değişken ve oranı 0.01 ise bu değerleri getir.

    for var in rare_columns: #rare column'larda gezilmiş.
        tmp = temp_df[var].value_counts() / len(temp_df)  #rare_column'ın oranı belirlenmiş.
        rare_labels = tmp[tmp < rare_perc].index # index buluyoruz.
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
        # eğer rare_columns'larda gezdiğin değerler rare_labels'da var ise rare yaz, yoksa aynı şekilde bırak.
    return temp_df

dff = rare_encoder(dff, 0.01)

def one_hot_encoder(dataframe, categorical_cols, drop_first= True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first = drop_first)
    return dataframe

ohe_cols = [col for col in dff.columns if 10 >= dff[col].nunique()> 2]

dff = one_hot_encoder(dff, ohe_cols)
dff.head()

#Standartization
dff.index = np.arange(0, len(dff))

df_salary=dff[["Salary"]]
print(df_salary)

temp_df=dff.drop(["Salary"],axis=1)

scaler = MinMaxScaler() # değerleri 1 ile 0 'a dönüştür yapmayı sağlıyor.


dff = pd.DataFrame(scaler.fit_transform(temp_df), columns = temp_df.columns) # formatı dataframe'e çeviriyoruz.
dff_2= pd.concat((dff,df_salary),axis=1)

dff_2.head()

#Creating model
models = []

models.append(('KNN', KNeighborsRegressor()))
models.append(('SVR', SVR()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('RandomForests', RandomForestRegressor()))
models.append(('GradientBoosting', GradientBoostingRegressor()))
models.append(('XGBoost', XGBRegressor()))
models.append(('Light GBM', LGBMRegressor()))

X = dff_2.drop("Salary",axis=1)
y = dff_2["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

for name,model in models:
    mod = model.fit(X_train,y_train)
    y_pred = mod.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(name,rmse)
    print("-------------")

