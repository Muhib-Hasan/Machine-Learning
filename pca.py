import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing 



dataframe=pd.read_csv('dataset_v2.csv')

features = ['Tempareture Avg', 'Humidity Avg', 'Rainfall Avg(mm)', 'Wind Speed Avg(Meter/Sec)',
			'Bright Sunshine (Hours)', 'Area(Acres)']
target = 'Production(M.tons)'

x = dataframe[features]
y = dataframe[target]

print(x)
print(y)

print(x.shape, y.shape)

scaled_data=preprocessing.scale(data.T)

pca=PCA()
pca.fit(scaled_data)
pca_data=pca.transform(scaled_data)

per_var=np.round(pca.explained_variance_ratio_*100,decimal=1)
labels=['PC'+str(x) for x in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1),height=per_var, tick_label=labels)
plt.ylabel('percentage of Explained Variance')
plt.title('scree Plot')
plt.show()