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

def split_x_y(x, y):
	x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.30, random_state=0)

	return x_train,x_test,y_train,y_test

def model_sklearn(x_train, y_train):
	model = LinearRegression()
	model.fit(x_train,y_train)

	return model

def model_evaluation(model, x_test, y_test):
	# checking R2 score
	y_predict = model.predict(x_test)
	score = r2_score(y_test,y_predict)
	print(score)

	# RMSE

def data_scaling(X_train, X_test):
	scaler = StandardScaler()
	scaler.fit(X_train)
	scaled_x_train = scaler.transform(X_train)
	scaled_x_test = scaler.transform(X_test)

	return scaled_x_train, scaled_x_test

def apply_pca(scaled_x_train, scaled_x_test):
	pca = PCA(n_components=2, random_state=0).fit(scaled_x_train)
	print("PCA Components: ", pca.n_components_)
	x_train_pca = pca.transform(scaled_x_train)
	x_test_pca = pca.transform(scaled_x_test)

	return x_train_pca, x_test_pca

if __name__ == '__main__':
	df = load_dataframe(FILEPATH)
	
	x, y = retrieve_features(df)
	print(x.shape, y.shape)
	
	x_train,x_test,y_train,y_test = split_x_y(x,y)
	print(x_train.shape, x_test.shape)
	print(y_train.shape, y_test.shape)

	# Without PCA
	print("\n###########Model without PCA###############\n")
	model_1 = model_sklearn(x_train, y_train)
	model_evaluation(model_1, x_test, y_test)

	# with PCA
	print("\n###########Model with PCA###############\n")
	x_train_scaled, x_test_scaled = data_scaling(x_train, x_test)
	print(x_train_scaled.shape, x_test_scaled.shape)

	x_train_pca, x_test_pca = apply_pca(x_train_scaled, x_test_scaled)
	print(x_train_pca.shape, x_test_pca.shape)

	model_2 = model_sklearn(x_train_pca, y_train)

	model_evaluation(model_2, x_test_pca, y_test)