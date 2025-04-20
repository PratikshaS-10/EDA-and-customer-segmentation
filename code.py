import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

data=pd.read_csv('Sales_August_2019.csv')

print(data.head())
print(data.shape) #------(12011,6)  
print(data.info()) # return each column data type and non null values in it

data['Order ID']=pd.to_numeric(data['Order ID'],errors='coerce')
data['Quantity Ordered']=pd.to_numeric(data['Quantity Ordered'],errors='coerce')
data['Price Each']=pd.to_numeric(data['Price Each'],errors='coerce')
data['Order Date']=pd.to_datetime(data['Order Date'],errors='coerce')

print(data.dtypes)
print(pd.isnull(data).sum())#finding number of null values i.e missing values
print(data.duplicated().sum()) #return no of all duplicates

data.drop_duplicates(inplace=True)#droping duplicates
print(data.duplicated().sum())#checking after droping duplicates

print(pd.isnull(data).sum())#only 1 missing value after removing duplicates so we remove it 
data.dropna(inplace=True)# deleted the missing value
print(data.isnull().sum())#checking after deleting missing value 
print(data.shape)#checking no of rows and columns after cleaning (11939,6)

data['Total Price']=data['Quantity Ordered']*data['Price Each']#made a column of total quantity
print(data.head())#viewing
print(data.shape)#---(11939,7)

print(data[['Quantity Ordered','Price Each','Total Price']].describe())#gives basic descriptive statistics for necessary numeric values

#DATA VISUALISATION
#DAILY SALES
daily_sales=data.groupby(data['Order Date'].dt.date)['Total Price'].sum()
sns.lineplot(x=daily_sales.index,y=daily_sales.values)
plt.title("Daily Sales over time")
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.show()

#TOP 10 PRODUCTS
best_selling=data.groupby('Product').agg({'Quantity Ordered':'sum','Total Price':'sum'}).sort_values(by='Total Price',ascending =False)
sns.barplot(x=best_selling.head(10).index,y=best_selling.head(10)['Total Price'],palette='Set1')#palette gives different color to every bar acc to set1 pallete
plt.title('Top 10 Selling Products by Total Price')
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.subplots_adjust(bottom=0.3)#adjust the margin to fit data-->>not cut off the labels
plt.xticks(rotation=45)
plt.show()

# Plot the pie chart -->>weekly quantity purchased and weekly sales
data['day of week']=data['Order Date'].dt.dayofweek
weekly_sales=data.groupby('day of week')['Total Price'].sum()
print(weekly_sales)
weekly_purchase=data.groupby('day of week')['Quantity Ordered'].sum()
print(weekly_purchase)
day_names={0:'Mon',1:'Tues',2:'Wed',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}
fig,axes=plt.subplots(1,2)
axes[0].pie(weekly_sales,labels=weekly_sales.index.map(day_names),autopct='%1.2f%%',startangle=90)
axes[0].set_title("Weekly Sales")
#plt.pie(weekly_sales, labels=weekly_sales.index.map(day_names), autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
axes[1].pie(weekly_purchase,labels=weekly_purchase.index.map(day_names),autopct='%1.2f%%',startangle=90)
axes[1].set_title('weekly quantity purchase')
plt.show() 

#city v/s sales
data['City'] = data['Purchase Address'].apply(lambda x: x.split(',')[1].strip())
city_sales = data.groupby('City')['Total Price'].sum().sort_values(ascending=True)
plt.figure(figsize=(12,8))
sns.barplot(y=city_sales.index, x=city_sales.values, palette='viridis')
plt.title('Total Sales by City')
plt.xlabel('City')
plt.ylabel('Total Sales')
for index,value in enumerate(city_sales.values):
    plt.text(value,index,f'{value}',fontweight='bold')
plt.show() 

#HEatmap between quantity ordered and price each-------look at scatter plot again
corr_matrix=data[['Quantity Ordered','Price Each']].corr()
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
plt.show()# we get negative correlation=-0.15
data['Price Group'] = np.where(data['Price Each'] > 50, 'High Price', 'Low Price')


#CLUSTERING
customer_data = data.groupby('Order ID').agg({
    'Total Price': 'sum',
    'Quantity Ordered': 'sum'
}).reset_index()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['Total Price', 'Quantity Ordered']])

# Elbow Method 
inertia = []
k_range = range(1, 11)  

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method - Choosing k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=6, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_data)

customer_data.rename(columns={
    'Total Price': 'Total money spend',
    'Quantity Ordered': 'Total quantity ordered',
    'Order ID': 'Number of Orders'
}, inplace=True)

cluster_labels_k6 = {
    0: 'Moderate Spenders',
    1: 'Frequent Low-Value Buyers',
    2: 'Bulk Bargain Shoppers',
    3: 'Luxury/Premium Buyers',
    4: 'Occasional High-Spend Buyers',
    5: 'Infrequent Low-Spenders'
}

customer_data['Customer Segment'] = customer_data['Cluster'].map(cluster_labels_k6)

print("Customer Segments (based on KMeans clustering):")
segment_summary = customer_data.groupby(['Cluster', 'Customer Segment']).agg({
    'Total money spend': 'sum',           
    'Total quantity ordered': 'sum',
    'Number of Orders': 'count'                   
})

segment_summary['Average selling price'] = segment_summary['Total money spend'] / segment_summary['Total quantity ordered']
segment_summary['Average revenue per order'] = segment_summary['Total money spend'] / segment_summary['Number of Orders']
print(segment_summary)


silhouette = silhouette_score(scaled_data, customer_data['Cluster'])
dbi = davies_bouldin_score(scaled_data, customer_data['Cluster'])
ch_score = calinski_harabasz_score(scaled_data, customer_data['Cluster'])

print(f"\nSilhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {dbi:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.2f}")
