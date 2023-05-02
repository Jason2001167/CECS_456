!pip install Cython
!pip install hdbscan
!pip install kneed # To install only knee-detection algorithm
import hdbscan
import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans # for KMeans algorithm
from kneed import KneeLocator # To find elbow point
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

data = pd.read_csv("SouthGermanCredit.csv")

import plotly.express as px
import warnings

# These featrues are chosen due to being qualitive features of the CSV file
# Outliers sorted based on amount 
# Can compare other features to these qualtiive features such as credit amount vs risk etc...... 

warnings.filterwarnings('ignore')
# generating statistics of data
data.describe()[['age','amount','duration']]
# Plot distribution plot for features

data.sample(5)

plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
sns.distplot(data['amount'])
plt.subplot(1,3,2)
sns.distplot(data['duration'])
plt.subplot(1,3,3)
sns.distplot(data['age'])
plt.show()

# Due to all features having a skewed disbruition, z-score treatment is not ideal
# IQR is a better approach

print("Old Shape: ", data.shape)

q_low = data["amount"].quantile(0.05)
q_hi  = data["amount"].quantile(0.95)

data_filtered = data[(data['amount'] < q_hi) & (data['amount'] > q_low)]


print("New Shape: ", data_filtered.shape)

plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
sns.distplot(data_filtered['amount'])
plt.subplot(1,3,2)
sns.distplot(data_filtered['duration'])
plt.subplot(1,3,3)
sns.distplot(data_filtered['age'])
plt.show()

# Drops around 100 rows of data
# Maybe only drop top 5% of data due to bottom 5% being prominent to data and patterns

data_filtered.corr()

sns.heatmap(data_filtered.corr());

plt.figure(figsize=(20, 10))
# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(data_filtered.corr(), dtype=np.bool))
heatmap = sns.heatmap(data_filtered.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);



plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(data_filtered.corr()[['amount']].sort_values(by='amount', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with amount', fontdict={'fontsize':18}, pad=16);

# Create matrix

sns.set_theme(style="white")

data_filtered.corr()

corr = data_filtered.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Possbily plot via seaborn?

# basically plotting different repersentions of the data 
# Can use to repersent patterns in data for presentation

# Histogram of the credit amount
plt.hist(data["amount"],bins=10)
plt.xlabel("Amount")
plt.ylabel("People")
plt.title("Number of people vs Credit Amount")
plt.legend("1 GDM = ")
plt.show()

# Histogram of Duration
plt.hist(data["duration"],bins=10)
plt.xlabel("Duration")
plt.ylabel("People")
plt.title("Number of people vs Duration")
plt.show()

y = np.array([700,300])
mylabels = ["Good Credit 30%", "Bad Credit 70%"]
myexplode = [0.2, 0]

plt.pie(y, labels = mylabels, explode = myexplode, shadow = True)
plt.show()

# basically plotting different repersentions of the data 
# Can use to repersent patterns in data for presentation

# Histogram of the credit amount
plt.hist(data["amount"],bins=10)
plt.xlabel("Amount")
plt.ylabel("People")
plt.title("Number of people vs Credit Amount")
plt.legend("1 GDM = ")
plt.show()

# Histogram of Duration
plt.hist(data["duration"],bins=10)
plt.xlabel("Duration")
plt.ylabel("People")
plt.title("Number of people vs Duration")
plt.show()

y = np.array([700,300])
mylabels = ["Good Credit 30%", "Bad Credit 70%"]
myexplode = [0.2, 0]

plt.pie(y, labels = mylabels, explode = myexplode, shadow = True)
plt.show()

plt.plot(range(1,11),sse)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances')
plt.title('Elbow curve')
plt.show()

k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
print(k1.elbow)


k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
kmeans = KMeans(n_clusters=k1.elbow)
kmeans.fit((data_filtered[['age','amount']]))

plt.xlabel('amount')
plt.ylabel('age')
plt.title('Amount Vs Age')

plt.scatter(data_filtered['amount'], data_filtered['age'], c=kmeans.labels_)
plt.show()

# filters data to only those with good credit risk
data_filtered2 = data_filtered[(data['credit_risk'] == 0)]
sse = []
for k in range(1,11):
  kmeans = KMeans(n_clusters=k, random_state=0,n_init=10)
  kmeans.fit(data_filtered2[['amount','age']])
  sse.append(kmeans.inertia_)
k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
kmeans = KMeans(n_clusters=k1.elbow)
kmeans.fit((data_filtered2[['amount','age']]))

plt.xlabel('amount')
plt.ylabel('age')
plt.title('Amount Vs Age with good credit risk')

plt.scatter(data_filtered2['amount'], data_filtered2['age'], c=kmeans.labels_)
plt.show()

# filters data to only those with bad credit risk
data_filtered3 = data_filtered[(data['credit_risk'] == 1)]
sse = []
for k in range(1,11):
  kmeans = KMeans(n_clusters=k, random_state=0,n_init=10)
  kmeans.fit(data_filtered3[['amount','age']])
  sse.append(kmeans.inertia_)
k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
kmeans = KMeans(n_clusters=k1.elbow)
kmeans.fit((data_filtered3[['amount','age']]))

plt.xlabel('amount')
plt.ylabel('age')
plt.title('Amount Vs Age with bad credit risk')

plt.scatter(data_filtered3['amount'], data_filtered3['age'], c=kmeans.labels_)
plt.show()


sse = []
for k in range(1,11):
  kmeans = KMeans(n_clusters=k, random_state=0,n_init=10)
  kmeans.fit(data_filtered[['amount','duration']])
  sse.append(kmeans.inertia_)

plt.plot(range(1,11),sse)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances')
plt.title('Elbow curve')
plt.show()

k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
print(k1.elbow)

k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
kmeans = KMeans(n_clusters=k1.elbow)
kmeans.fit((data_filtered[['amount','duration']]))

plt.scatter(data_filtered['amount'], data_filtered['duration'], c=kmeans.labels_)

plt.xlabel('amount')
plt.ylabel('duration')
plt.title('Amount Vs Duration')

plt.show()

# filters data to only those with good credit risk
data_filtered2 = data_filtered[(data['credit_risk'] == 0)]
sse = []
for k in range(1,11):
  kmeans = KMeans(n_clusters=k, random_state=0,n_init=10)
  kmeans.fit(data_filtered2[['amount','duration']])
  sse.append(kmeans.inertia_)
k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
kmeans = KMeans(n_clusters=k1.elbow)
kmeans.fit((data_filtered2[['amount','duration']]))

plt.xlabel('amount')
plt.ylabel('duration')
plt.title('Amount Vs Duration with good credit risk')

plt.scatter(data_filtered2['amount'], data_filtered2['duration'], c=kmeans.labels_)
plt.show()

# filters data to only those with bad credit risk
data_filtered3 = data_filtered[(data['credit_risk'] == 1)]
sse = []
for k in range(1,11):
  kmeans = KMeans(n_clusters=k, random_state=0,n_init=10)
  kmeans.fit(data_filtered3[['amount','duration']])
  sse.append(kmeans.inertia_)
k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
kmeans = KMeans(n_clusters=k1.elbow)
kmeans.fit((data_filtered3[['amount','duration']]))

plt.xlabel('amount')
plt.ylabel('age')
plt.title('Amount Vs Age with bad credit risk')

plt.scatter(data_filtered3['amount'], data_filtered3['duration'], c=kmeans.labels_)
plt.show()

sse = []
for k in range(1,11):
  kmeans = KMeans(n_clusters=k, random_state=0,n_init=10)
  kmeans.fit(data_filtered[['duration','age']])
  sse.append(kmeans.inertia_)

plt.plot(range(1,11),sse)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of squared distances')
plt.title('Elbow curve')
plt.show()

k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
print(k1.elbow)

k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
kmeans = KMeans(n_clusters=3)
kmeans.fit((data_filtered[['duration','age']]))

plt.scatter(data_filtered['duration'], data_filtered['age'], c=kmeans.labels_)

plt.xlabel('Duration')
plt.ylabel('Age')
plt.title('Age Vs. Duration')


plt.show()

silhouette = silhouette_score(data_filtered,kmeans.labels,metric='euclidean')
print(silhouette)
