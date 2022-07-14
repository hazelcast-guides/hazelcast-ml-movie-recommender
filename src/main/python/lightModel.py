# ratings.txt
import datetime
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Embedding, Flatten, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

df_movies = pd.read_csv("archive/moviedb/movies_cast_company.csv", encoding='utf8')
df_movies["cast"] = df_movies["cast"].apply(lambda x: json.loads(x))
df_movies["company"] = df_movies["company"].apply(lambda x: json.loads(x))

df_movies = df_movies.drop(["url"] + list(df_movies.columns[-4:]), axis=1)

print(df_movies.shape)
df_movies.head()


def string2ts(string, fmt="%Y-%m-%d %H:%M:%S"):
    dt = datetime.datetime.strptime(string, fmt)
    t_tuple = dt.timetuple()
    return int(time.mktime(t_tuple))


df_ratings = pd.read_csv("archive/moviedb/ratings.csv")

df_ratings.rating_timestamp = df_ratings.rating_timestamp.apply(lambda x: string2ts(x))

print(df_ratings.shape)
df_ratings.head()

df_users = pd.read_csv("archive/moviedb/users.csv")

additional_rows = ["user_zipcode"]
df_users = df_users.drop(additional_rows, axis=1)

num2occupation = dict(enumerate(df_users.user_occupation.unique()))
occupation2num = {y: x for x, y in num2occupation.items()}
num2gender = dict(enumerate(df_users.user_gender.unique()))
gender2num = {y: x for x, y in num2gender.items()}
df_users.user_occupation = df_users.user_occupation.apply(lambda x: occupation2num[x])
df_users.user_gender = df_users.user_gender.apply(lambda x: gender2num[x])

print(df_users.shape)
df_users.head()

df = pd.merge(df_movies, df_ratings, on="movie_id_ml")
df = pd.merge(df, df_users, on="user_id")

print(df.shape)
df.head()

id2movie = dict(enumerate(df.movie_id_ml.unique()))
movie2id = {y: x for x, y in id2movie.items()}

id2user = dict(enumerate(df.user_id.unique()))
user2id = {y: x for x, y in id2user.items()}
df["iid"] = df.apply(lambda x: movie2id[x.movie_id_ml], axis=1)
df["uid"] = df.apply(lambda x: user2id[x.user_id], axis=1)

# TODO; remove this line when real user
new_user_id = 942  # user_movie_ratings_training[0][0]
df = df[df.uid != new_user_id]
df = df.drop_duplicates()
# shape [n_users, n_user_features]
df_users = df[["uid", "user_age", "user_gender", "user_occupation"]].drop_duplicates()
print(f"Number of users features: {df_users.shape[0]}")

df_movies = df[["iid"] + list(df.columns[3:22])].drop_duplicates()
print(f"Number of movies features: {df_movies.shape[0]}")

columns = ["user_age", "user_gender", "user_occupation", "uid", "iid", "rating", "rating_timestamp"]

# user info
user_age = 19
user_gender = 0
user_occupation = 5

# ratings
user_movie_ratings_training = [[942, 891, 4, 893279438],
                               [942, 130, 2, 893279061],
                               [942, 133, 2, 893279138],
                               [942, 488, 4, 893279438],
                               [942, 854, 3, 893279004],
                               [942, 137, 3, 893279438],
                               [942, 141, 4, 893279437],
                               [942, 609, 3, 893279438],
                               [942, 612, 4, 893279438],
                               [942, 857, 1, 893279173]]

user_movie_ratings_test = [[942, 343, 1, 893278968],
                           [942, 862, 5, 893279437],
                           [942, 1124, 4, 893279437],
                           [942, 918, 4, 893279438],
                           [942, 438, 2, 893278949],
                           [942, 348, 4, 893279438],
                           [942, 447, 3, 893279138],
                           [942, 659, 3, 893279099],
                           [942, 875, 1, 893279311],
                           [942, 878, 1, 893279291]]

new_user_id = 942  # user_movie_ratings_training[0][0]

df_users = df_users.append(
    {"uid": new_user_id, "user_age": user_age, "user_gender": user_gender, "user_occupation": user_occupation},
    ignore_index=True)
print(f"Number of users features: {df_users.shape[0]}")
print(f"Number of movies features: {df_movies.shape[0]}")

# create new user dataframe with training data

data_new_user_training = []
for x in user_movie_ratings_training:
    data_new_user_training.append([user_age, user_gender, user_occupation] + x)

data_new_user_test = []
for x in user_movie_ratings_test:
    data_new_user_test.append([user_age, user_gender, user_occupation] + x)

# user initial input that will be given to him to rate it before recommendation
df_new_user_train = pd.DataFrame(data_new_user_training, columns=columns)
# the input that will be checked if recommendation works fine
df_new_user_test = pd.DataFrame(data_new_user_test, columns=columns)

# training and test data
train_idx, test_idx = train_test_split(range(df.shape[0]), test_size=0.2, random_state=42)

df_train = df.iloc[train_idx]
df_train = pd.concat([df_train, df_new_user_train], sort=False)

df_test = df.iloc[test_idx]
df_test = pd.concat([df_test, df_new_user_test], sort=False)
df_train = df_train[["uid", "iid", "rating", "rating_timestamp"]]
df_test = df_test[["uid", "iid", "rating", "rating_timestamp"]]
df_new_user_train = df_new_user_train[["uid", "iid", "rating", "rating_timestamp"]]
df_new_user_test = df_new_user_test[["uid", "iid", "rating", "rating_timestamp"]]

uids = set(df_train.uid.unique()).union(set(df_test.uid.unique()))
iids = set(df_train.iid.unique()).union(set(df_test.iid.unique()))

rows = max(uids) + 1
cols = max(iids) + 1

print("Users number: ", len(uids), rows)
print("Movies number: ", len(iids), cols)


def _build_interaction_matrix(rows, cols, data):
    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, timestamp in data:
        # Let's assume only really good things are positives
        if rating >= 4.0:
            mat[uid, iid] = 1.0

    return mat.tocoo()


def get_triplets(mat):
    return mat.row, mat.col, np.random.randint(mat.shape[1], size=len(mat.row))


def create_sparse_matrix(df):
    """
    Return (train_interactions, test_interactions).
    """
    return _build_interaction_matrix(rows, cols, df.values.tolist())


def predict(model, uid, pids):
    user_vector = model.get_layer('user_embedding').get_weights()[0][uid]
    item_matrix = model.get_layer('item_embedding').get_weights()[0][pids]

    scores = (np.dot(user_vector,
                     item_matrix.T))

    return scores


def full_auc(model, ground_truth):
    """
    Measure AUC for model and ground truth on all items.

    Returns
    -------
    - float AUC
    """
    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    scores = []

    for user_id, row in enumerate(ground_truth):

        predictions = predict(model, user_id, pid_array)

        true_pids = row.indices[row.data == 1]

        grnd = np.zeros(no_items, dtype=np.int32)
        grnd[true_pids] = 1

        if len(true_pids):
            scores.append(roc_auc_score(grnd, predictions))

    return np.mean(scores)


class TripletLossLayer(Layer):
    def call(self, inputs):
        positive_item_latent, negative_item_latent, user_latent = inputs

        # Bayesian Personalised Ranking (BPR) loss
        loss = 1.0 - K.sigmoid(
            K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
            K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

        return loss


def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


def build_model(num_users, num_items, latent_dim):
    positive_item_input = Input((1,), name='positive_item_input')
    negative_item_input = Input((1,), name='negative_item_input')

    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding(num_items, latent_dim, name='item_embedding', input_length=1)
    user_input = Input((1,), name='user_input')

    positive_item_embedding = Flatten()(item_embedding_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_embedding_layer(negative_item_input))
    user_embedding = Flatten()(Embedding(num_users, latent_dim, name='user_embedding', input_length=1)(user_input))

    loss = TripletLossLayer()((positive_item_embedding, negative_item_embedding, user_embedding))

    model = Model([positive_item_input, negative_item_input, user_input], loss)

    model.compile(loss=identity_loss, optimizer=Adam())

    return model


# train.todense()


def my2csr(df):
    return sp.csr_matrix(df.values)


train = create_sparse_matrix(df_train)  # , mat_type="ratings")
test = create_sparse_matrix(df_test)  # , mat_type="ratings")

# shape [n_users, n_user_features]
user_features = my2csr(df_users)
item_features = my2csr(df_movies)
use_features = False

loss_type = "warp"  # "bpr"

model = LightFM(learning_rate=0.05, loss=loss_type, max_sampled=100)

if use_features:
    model.fit_partial(train, epochs=20, user_features=user_features, item_features=item_features)
    # model.fit(train, epochs=50, user_features=user_features, item_features=item_features)
    train_precision = precision_at_k(model, train, k=10, user_features=user_features,
                                     item_features=item_features).mean()
    test_precision = precision_at_k(model, test, k=10, train_interactions=train, user_features=user_features,
                                    item_features=item_features).mean()

    train_auc = auc_score(model, train, user_features=user_features, item_features=item_features).mean()
    test_auc = auc_score(model, test, train_interactions=train, user_features=user_features,
                         item_features=item_features).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
else:
    model.fit_partial(train, epochs=20)

    train_precision = precision_at_k(model, train, k=10).mean()
    test_precision = precision_at_k(model, test, k=10).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test, train_interactions=train).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

user_ratings = user_movie_ratings_training + user_movie_ratings_test
user_ratings_positive = []
user_ratings_negative = []
for ur in user_ratings:
    if ur[2] >= 4:
        user_ratings_positive.append(ur[1])
    else:
        user_ratings_negative.append(ur[1])


def predict_movies(model, user_id):
    n_movies = train.shape[1]
    if use_features:
        prediction = model.predict(user_id, np.arange(n_movies), user_features=user_features,
                                   item_features=item_features)  # predict(model, user_id, movie_ids)
    else:
        prediction = model.predict(user_id, np.arange(n_movies))  # predict(model, user_id, movie_ids)

    rated_high = [1 if np.round(i) > 0.5 else 0 for i in prediction]
    good_movie_ids = np.where(np.array(rated_high) == 1)[0]
    return list(good_movie_ids)


predicted_good_movies = predict_movies(model, new_user_id)
for m in user_ratings_positive:
    print(m if m in predicted_good_movies else f"{m} PREDICTED WRONG: SHOULD BE GOOD MOVIE FOR THIS USER")

for m in user_ratings_negative:
    print(m if m not in predicted_good_movies else f"{m} PREDICTED WRONG: SHOULD BE BAD MOVIE FOR THIS USER")

user_sparse_matrix = create_sparse_matrix(df_new_user_test)
if use_features:
    acc = auc_score(model, user_sparse_matrix, user_features=user_features, item_features=item_features).mean()
else:
    acc = auc_score(model, user_sparse_matrix).mean()

print(f"Accuracy for new user: {acc * 100:.2f}%")


def predict_top_k_movies(model, user_id, k):
    n_users, n_movies = train.shape
    if use_features:
        prediction = model.predict(user_id, np.arange(n_movies), user_features=user_features,
                                   item_features=item_features)  # predict(model, user_id, np.arange(n_movies))
    else:
        prediction = model.predict(user_id, np.arange(n_movies))

    movie_ids = np.arange(train.shape[1])
    return movie_ids[np.argsort(-prediction)][:k]


k = 10
user_id = new_user_id
movie_ids = np.arange(train.shape[1])

n_users, n_items = train.shape

known_positives = movie_ids[train.tocsr()[user_id].indices]

scores = model.predict(user_id, np.arange(n_items), user_features=user_features, item_features=item_features)
top_items = movie_ids[np.argsort(-scores)]

print("User %s" % user_id)
print("     Known positives:")

for x in known_positives[:k]:
    print(f"        {df[df.iid == x]['iid'].iloc[0]} | {df[df.iid == x]['title'].iloc[0]}")

print("     Recommended:")
for x in top_items[:k]:
    print(f"        {df[df.iid == x]['iid'].iloc[0]} | {df[df.iid == x]['title'].iloc[0]}")
