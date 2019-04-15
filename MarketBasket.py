import os
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# !pip install mlxtend
os.path
os.chdir("D:\data science\Materials\probability\Materials\Market Basket")
os.getcwd()
df = pd.read_csv("Online Retail.csv",encoding = "ISO-8859-1")
df.head()

pd.isnull(df).sum()
df['CustomerID'] = df['CustomerID'].fillna(df['CustomerID'].mean)
df = df[~pd.isnull(df.CustomerID)]
df.shape
df.head()
'############################################################################'

import datetime
 
asd = np.array([datetime.datetime.strptime(i,"%d-%m-%y").date() for i in df['InvoiceDate']])

df['InvoiceDate'] = np.array([datetime.datetime.strptime(i,"%d-%m-%y").date() for i in df['InvoiceDate']])

df['InvoiceDate'].min(),df['InvoiceDate'].max()

# Analysis period = 1 yr
 
df = df[df['InvoiceDate'] >= pd.to_datetime('2010-12-09').date()]

df.shape

df['InvoiceDate'].min(),df['InvoiceDate'].max()

df.shape

' ########################################################################'

df['Description'] = df['Description'].str.strip()

df['InvoiceNo'] = df['InvoiceNo'].astype('str')

df = df[~df['InvoiceNo'].str.contains('C')]

df.shape

' ########################################################################'

pd.value_counts(df.Country)

pd.value_counts(df.Country).plot(kind="bar")
# taking only UK Records
basket = (df[df['Country'] =="United Kingdom"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket.head()
def buy_or_not(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


basket.apply(sum)

basket['10 COLOUR SPACEBOY PEN'].map(lambda x:x-2)

basket_sets = basket.applymap(buy_or_not)

basket_sets.drop('POSTAGE', inplace=True, axis=1)

' #############################  Apriori Algo ###########################'

frequent_itemsets = apriori(basket_sets, min_support=0.025, use_colnames=True)

frequent_itemsets.head()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

rules = rules.sort_values('lift',ascending=False)

rules.shape
rules.columns
rules.head()
