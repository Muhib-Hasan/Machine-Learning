import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

FILEPATH = "dataset_v4.csv"

def load_dataframe(filepath):
	dataframe=pd.read_csv(filepath)

	return dataframe

def retrieve_features(dataframe):
	features = ["Code","Year","Tempareture","Humidity","Rainfall","Wind Speed",
				"Bright Sunshine","Cloud Coverage","Area"]
	target = "Production"

	x = dataframe[features]
	y = dataframe[target]

	return x, y

def data_scaling(X):
	scaler = StandardScaler()
	scaler.fit(X)
	scaled_x = scaler.transform(X)

	return scaled_x

def apply_pca(scaled_x):
	pca = PCA(n_components=1, random_state=0).fit(scaled_x)
	print("PCA Components: ", pca.n_components_)
	x_pca = pca.transform(scaled_x)

	return x_pca

def plot(x, y):
	colors = np.random.rand(136)
	plt.scatter(x, y, c=colors, alpha=0.7)
	plt.show()

if __name__ == '__main__':
	df = load_dataframe(FILEPATH)
	
	x, y = retrieve_features(df)
	print(x.shape, y.shape)

	# with PCA
	print("\n###########Model with PCA###############\n")
	x_scaled = data_scaling(x)
	print(x_scaled.shape)

	x_pca = apply_pca(x_scaled)
	print(x_pca.shape)

	plot(x_pca, y)