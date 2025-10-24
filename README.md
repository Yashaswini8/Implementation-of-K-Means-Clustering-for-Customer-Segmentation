# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Data Preparation:Collect customer data and select relevant numeric features such as Annual Income and Spending Score.
2. Determine Clusters:Use the Elbow Method to find the optimal number of clusters by plotting WCSS (Within-Cluster Sum of Squares) against cluster counts.
3. Apply K-Means:Fit the K-Means algorithm on the selected features, assign each customer to the nearest cluster, and compute cluster centroids.
4. Analyze & Visualize:Visualize clusters on a scatter plot with centroids, label each customer according to their cluster, and use this segmentation for targeted marketing or decision-making.


 ## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Yashaswini S
RegisterNumber:  212224220123
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Create a sample dataset
data = {
    'CustomerID': range(1, 11),
    'AnnualIncome': [15, 16, 17, 20, 25, 50, 55, 60, 80, 90],
    'SpendingScore': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}
df = pd.DataFrame(data)

# Step 2: Select features for clustering
X = df[['AnnualIncome', 'SpendingScore']]

# Step 3: Determine optimal clusters using Elbow method
wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 6), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Step 4: Apply K-Means with 3 clusters (example)
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Step 5: Visualize clusters
plt.scatter(X.iloc[y_kmeans == 0, 0], X.iloc[y_kmeans == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X.iloc[y_kmeans == 1, 0], X.iloc[y_kmeans == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X.iloc[y_kmeans == 2, 0], X.iloc[y_kmeans == 2, 1], s=50, c='green', label='Cluster 3')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids', marker='X')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
```

## Output:
<img width="1250" height="755" alt="Screenshot 2025-10-24 105554" src="https://github.com/user-attachments/assets/c00ffd2f-8666-4523-a4b4-ca1db1572467" />
<img width="1127" height="761" alt="Screenshot 2025-10-24 105606" src="https://github.com/user-attachments/assets/2c7145f0-6fe6-4375-b2be-021d25c44c23" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
