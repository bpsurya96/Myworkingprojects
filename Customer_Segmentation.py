import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir("D:/Data Science/cluster")

os.getcwd()

data = pd.read_csv("Online Retail.csv",
                   encoding = "ISO-8859-1")
data.info()

data['InvoiceDate']

data.head()
'############################################################################'

'# NA s '
 
data.shape

pd.isnull(data).sum()

data = data[~pd.isnull(data.CustomerID)]

data.shape

'############################################################################'

import datetime

data.dtypes

asd = np.array([datetime.datetime.strptime(i,"%d-%m-%Y").date() for i in data['InvoiceDate']])

data['InvoiceDate'] = np.array([datetime.datetime.strptime(i,"%d-%m-%Y").date() for i in data['InvoiceDate']])

data['InvoiceDate'].min(),data['InvoiceDate'].max()

# Analysis period = 1 yr
 
data = data[data['InvoiceDate'] >= pd.to_datetime('2010-12-09').date()]

data.shape

data['InvoiceDate'].min(),data['InvoiceDate'].max()

'############################################################################'

pd.value_counts(data.Country)

pd.value_counts(data.Country).plot(kind="bar")

data = data[data.Country == "United Kingdom"]

data.shape

'############################################################################'

# Identify returns

import re

data['item.return'] = np.array([(bool(re.search('C',i))) for i in data['InvoiceNo']])

data['purchase.invoice'] = np.array([0 if i==True else 1 for i in data['item.return']])

plt.hist(data['purchase.invoice'])

'####################################################################'
            #'# Customer-level dataset and RFM Analysis'
'####################################################################'

len(np.unique(data['CustomerID']))

customers =  pd.DataFrame(np.unique(data['CustomerID']))

customers.head()

customers = customers.rename(columns = {0:'CustomerID'})

customers.head()

'####################################################################'
              #  '# Recency & Tenure'
'####################################################################'

data['InvoiceDate'].min(),data['InvoiceDate'].max()

data['recency'] =  pd.to_datetime('2011-12-10').date() - data['InvoiceDate']

data.info()

data.head()

data['recency'] = np.array([i.days for i in data['recency']])

# remove returns so only consider the data of most recent *purchase*
temp = data[data['purchase.invoice'] == 1]
temp.shape
temp.head()

# Obtain # of days since most recent purchase


list(temp[['recency','CustomerID']].groupby('CustomerID'))

recency = temp[['recency','CustomerID']].groupby('CustomerID').agg(min).reset_index()

recency.head()

tenure = temp[['recency','CustomerID']].groupby('CustomerID').agg(max).reset_index()

tenure.head()

tenure.rename(columns={'recency':'tenure'},inplace=True)

tenure.head()

concat_data = recency.merge(tenure,how="inner")

concat_data.head()

# Add recency to customer data
customers = customers.merge(concat_data,how="left")

customers.head()

'####################################################################'
             #   '# Frequency '
'####################################################################'

# No of purchases

customer_invoices = data.reindex(columns = ["CustomerID","InvoiceNo", "purchase.invoice"])
customer_invoices.head()

customer_invoices = customer_invoices[~customer_invoices.duplicated()].reset_index()
del customer_invoices['index']
customer_invoices.head()


freq_table = customer_invoices[['purchase.invoice','CustomerID']].groupby('CustomerID').agg(sum).reset_index()
freq_table.rename(columns={'purchase.invoice':'frequency_p'},inplace=True)
freq_table.head()

# No of Cancellation

customer_invoices = data.reindex(columns = ["CustomerID","InvoiceNo", "item.return"])
customer_invoices.head()

customer_invoices = customer_invoices[~customer_invoices.duplicated()].reset_index()
del customer_invoices['index']
customer_invoices.head()


freq_table2 = customer_invoices[['item.return','CustomerID']].groupby('CustomerID').agg(sum).reset_index()
freq_table2.rename(columns={'item.return':'frequency_c'},inplace=True)
freq_table2.head()


# Add # of invoices to customers data
customers = customers.merge(freq_table, how='left')
customers = customers.merge(freq_table2, how='left')
customers.head()
customers['frequency'] = customers['frequency_p'] - customers['frequency_c']
del customers['frequency_p'] 
del customers['frequency_c']

customers.head()

# Remove customers who have not made any purchases in the past year
customers = customers[customers['frequency'] > 0]

customers.head()

'####################################################################'
          #  '# Monetary Value of Customers #'
'####################################################################'

# Total spent on each item on an invoice
data['Amount'] = data['Quantity'] * data['UnitPrice']
data.head()

# Aggregated total sales to customer
sales_table = data[['Amount','CustomerID']].groupby('CustomerID').agg(sum).reset_index()
sales_table.head()

# Add monetary value to customers dataset
customers = customers.merge(sales_table, how="left")
customers.head()
customers.rename(columns = {'Amount':'monetary'},inplace=True)

# Identify customers with negative monetary value numbers, as they were presumably returning purchases from the preceding year
plt.hist(customers['monetary'])

customers['monetary'] = np.where(customers['monetary'] < 0 ,0, customers['monetary']) # reset negative numbers to zero
plt.hist(customers['monetary'])

###############################
# Breadth of Customers - Unique SKU's purchased.
###############################

invoice = data.reindex(columns = ["InvoiceNo","CustomerID","StockCode","purchase.invoice"])

inv1 = invoice[invoice['purchase.invoice'] == 1]

sku_table =  inv1[['StockCode','CustomerID']].groupby('CustomerID').agg(pd.Series.nunique).reset_index()

sku_table.rename(columns={'StockCode':"pur.breadth"},inplace=True)

sku_table.head()


inv2 = invoice[invoice['purchase.invoice'] == 0]

sku_table2 =  inv2[['StockCode','CustomerID']].groupby('CustomerID').agg(pd.Series.nunique).reset_index()

sku_table2.rename(columns={'StockCode':"ret.breadth"},inplace=True)

sku_table2.head()


customers  = customers.merge(sku_table,how="left")

customers  = customers.merge(sku_table2,how="left")

customers.head()


customers.loc[pd.isnull(customers['pur.breadth']),"pur.breadth"] = 0

customers.loc[pd.isnull(customers['ret.breadth']),"ret.breadth"] = 0

customers['breadth'] = customers['pur.breadth'] - customers['ret.breadth']

del customers['pur.breadth']
del customers['ret.breadth']


#backup = customers

backup = customers.copy()

customers = backup.copy()

customers.columns

###################################################################
#Preprocess data
##################################################################

import seaborn as sns
customers.head()
z_customers   = customers.apply(lambda x: (x-x.mean())/x.std())
z_customers['CustomerID'] = customers['CustomerID']

fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(531)
sns.kdeplot(customers['recency'],ax=ax1)

ax2 = fig.add_subplot(534)
sns.kdeplot(customers['frequency'],ax=ax2)

ax3 = fig.add_subplot(537)
sns.kdeplot(customers['monetary'],ax=ax3)

ax4 = fig.add_subplot(5,3,10)
sns.kdeplot(customers['breadth'],ax=ax4)

ax5 = fig.add_subplot(5,3,13)
sns.kdeplot(customers['tenure'],ax=ax5)

################################ LOG Transformed Vars #################################

ax6 = fig.add_subplot(532)
sns.kdeplot(np.log(customers['recency']),ax=ax6)

ax7 = fig.add_subplot(535)
sns.kdeplot(np.log(customers['frequency']),ax=ax7)

ax8 = fig.add_subplot(538)
sns.kdeplot(np.log(customers['monetary']),ax=ax8)

ax9 = fig.add_subplot(5,3,11)
sns.kdeplot(np.log(customers['breadth']),ax=ax9)

ax10 = fig.add_subplot(5,3,14)
sns.kdeplot(np.log(customers['tenure']),ax=ax10)

###################################################################################

ax11 = fig.add_subplot(533)
sns.kdeplot(z_customers['recency'],ax=ax11)

ax12 = fig.add_subplot(536)
sns.kdeplot(z_customers['frequency'],ax=ax12)

ax13 = fig.add_subplot(539)
sns.kdeplot(z_customers['monetary'],ax=ax13)

ax14 = fig.add_subplot(5,3,12)
sns.kdeplot(z_customers['breadth'],ax=ax14)

ax15 = fig.add_subplot(5,3,15)
sns.kdeplot(z_customers['tenure'],ax=ax15)

###################################################################
customers['recency'] = np.log(customers['recency']+0.1)
customers['frequency'] = np.log(customers['frequency']+0.1)
customers['monetary'] = np.log(customers['monetary']+0.1)
customers['breadth'] = np.log(customers['breadth']+0.1)
customers['tenure'] = np.log(customers['tenure']+0.1)

z_customers   = customers.apply(lambda x: (x-x.mean())/x.std())
z_customers['CustomerID'] = customers['CustomerID']

final_dataset = z_customers
final_dataset = final_dataset.fillna(0)
pd.isna(final_dataset).sum()

from sklearn.cluster import KMeans

# Elbow Chart
z_customers.columns

wss = []


for i in range(2,20):
    print(i)
    model = KMeans(n_clusters=(i), random_state=0).fit(final_dataset.iloc[:,1:])
    wss.append(model.inertia_)
 
wss

plt.plot(range(2,20), wss,'-*',color="red",linewidth=2.0)

##################################################################

final_model = KMeans(n_clusters=4, random_state=0).fit(final_dataset.iloc[:,1:])

final_model.labels_

z_customers['clusterno'] = final_model.labels_

z_customers.head()

plt.scatter(customers['monetary'],customers['tenure'],c=final_model.labels_)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(z_customers['recency'],z_customers['tenure'],z_customers['frequency'],c=z_customers['clusterno'])
