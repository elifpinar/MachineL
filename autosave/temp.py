# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

#veri yükleme
stores = pd.read_csv('Stores.csv')
#print(stores)


storeSales = stores[['Store_Sales']]
#print(storeSales)

#eksik verilerin çözümü için farklı yöntemler var
storesMissing  = pd.read_csv('StoresMissing.csv')
print(storesMissing)


#eksik verilerin içine ortalama yazarak

#sklearn kullanmadan bulup ortalama yazma
storesMissing.mean()
storesMissing_mean = storesMissing.fillna(storesMissing.mean())
print(storesMissing_mean.Daily_Customer_Count)

#sklearn kullanarak
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan ,strategy='mean')
Daily_Customer_Count=storesMissing.iloc[:,1:4].values
print(Daily_Customer_Count)
imputer = imputer.fit(Daily_Customer_Count[:,1:4]) #ortalamayı öğrenme
Daily_Customer_Count[:,1:4]=imputer.transform(Daily_Customer_Count[:,1:4]) #nan değerlerin yerine yazdırma
print(Daily_Customer_Count)



#--
#kategorik verilerin sayısal verilere çevrilmesi (encode edilmesi)
olympics= pd.read_csv('Olympics2022.csv')
continent =olympics.iloc[:,1:2].values #1 ile 2 arasındaki değerleri alır 
print(continent)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()  
continent[:,0]=le.fit_transform(olympics.iloc[:,1:2]) 
print(continent) #europe=1, asia=0, north america=2 olarak nümerik yaptik

ohe= preprocessing.OneHotEncoder()
continent =ohe.fit_transform(continent).toarray()
print (continent) #çalışma mantığı 

sonuc= pd.DataFrame(data=continent,index=range(4),columns=['Asia','Europe','North America'] )
print(sonuc)

"""
birleştirme
birlestirme=pd.concat([sonuc,sonuc2],axis=1)
birlestirme2=pd.concat([birlestirme,sonuc3],axis=1) """

################
#1kutuphaneler


#2veri onisleme
#2.1veri yukleme
veriler = pd.read_csv('Veriler.csv')
print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi boy yas gibi farklı şeylerin aynı dünyaya çekilmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)









