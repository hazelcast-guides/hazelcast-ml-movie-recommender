import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

df1 = pd.read_csv("archive/moviedb/movies.csv")
df2 = pd.read_csv("archive/moviedb/cast.csv")
df3 = pd.read_csv("archive/moviedb/ratings.csv")
df4 = pd.read_csv("archive/moviedb/users.csv")

