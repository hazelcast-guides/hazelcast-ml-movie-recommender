import datetime
import random
import time

import numpy as np
import pandas as pd

from surprise import BaselineOnly
from surprise import CoClustering
from surprise import Dataset, accuracy
from surprise import KNNBaseline
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import NormalPredictor
from surprise import Reader
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from tabulate import tabulate

ratings = pd.read_csv('archive/moviedb/ratings.csv')

ratings_dict = {'itemID': ratings.movie_id_ml,
                'userID': ratings.user_id,
                'rating': ratings.rating
                }

df = pd.DataFrame(ratings_dict)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)

genres = ['unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
          'fantasy', 'noir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller', 'war', 'western']
column_item = ["movie_id_ml", "title", "release", "vrelease", "url"] + genres

mat = np.zeros((max(ratings.user_id), max(ratings.movie_id_ml)))
ind = np.array(list(zip(list(ratings.user_id - 1), list(ratings.movie_id_ml - 1))))
mat[ind[:, 0], ind[:, 1]] = 1
movies_ = mat.sum(axis=0).argsort() + 1
np.random.shuffle(movies_)
top15 = movies_[:15]
pd.read_csv('archive/moviedb/u.item.txt', delimiter='|', encoding="ISO-8859-1", header=None)

df_ML_movies = pd.read_csv('archive/moviedb/u.item.txt', delimiter='|', names=column_item, encoding="ISO-8859-1")

mat.sum(axis=0).argsort()

start = time.time()
algo = NMF()
algo.fit(trainset)
predictions = algo.test(testset)
print("Test Set Error\n--------------")
accuracy.mae(predictions)
print("--------------\nFinished in {:.3f} sec.".format(time.time() - start))

# algo.pu -> User Matrix
# algo.qi -> Item Matrix

# The algorithms to cross-validate
classes = (SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline,
           CoClustering, BaselineOnly, NormalPredictor)

# seeds
np.random.seed(0)
random.seed(0)

kf = KFold(random_state=0)  # folds will be the same for all algorithms.
table = []
for klass in classes:
    start = time.time()
    out = cross_validate(klass(), data, ['rmse', 'mae'], kf)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))

    mean_rmse = '{:.3f}'.format(np.mean(out['test_rmse']))
    mean_mae = '{:.3f}'.format(np.mean(out['test_mae']))

    new_line = [klass.__name__, mean_rmse, mean_mae, cv_time]
    #     print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
    table.append(new_line)

header = ['RMSE', 'MAE', 'Time']
print(tabulate(table, header, tablefmt="pipe"))

svd = SVD()

start = time.time()

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
svd.fit(trainset)

predictions = svd.test(testset)

print("\nTest Set Error\n--------------")
accuracy.mae(predictions)
print("--------------\nFinished in {:.3f} sec.".format(time.time() - start))

# predict(user_id, movie_id) -> returns an estimated prediction of 2.21
svd.predict(22, 377, 3)

algo = NMF()

start = time.time()

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
algo.fit(trainset)

predictions = algo.test(testset)

print("\nTest Set Error\n--------------")
accuracy.mae(predictions)
print("--------------\nFinished in {:.3f} sec.".format(time.time() - start))
