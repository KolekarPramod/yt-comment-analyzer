import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

df = pd.read_csv('/home/pramod/work/yt-comment-analyzer/bank.csv')

df2 = df.select_dtypes(include=['int64'])
# Separating features and target variable
X = df2
y = df['y']

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Creating a DataFrame with PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['y'] = y.values

df_pca.to_csv(os.path.join('data','processed','bank_pca.csv'), index=False)