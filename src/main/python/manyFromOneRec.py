import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

genres = ['unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 'crime', 'documentary', 'drama',
          'fantasy', 'noir', 'horror', 'musical', 'mystery', 'romance', 'scifi', 'thriller', 'war', 'western']


def onehotencoding2genre(x):
    ret_val = []
    for c in genres:
        g = getattr(x, c)
        if g == 1:
            ret_val.append(c)
    return ret_val


df_movies = pd.read_csv("archive/moviedb/movies_cast_company.csv", encoding='utf8')
df_movies["cast"] = df_movies["cast"].apply(lambda x: json.loads(x))
df_movies["company"] = df_movies["company"].apply(lambda x: json.loads(x))
df_movies["genres"] = df_movies.apply(lambda x: onehotencoding2genre(x), axis=1)

df_ratings = pd.read_csv("archive/moviedb/ratings.csv")

df_users = pd.read_csv("archive/moviedb/users.csv")

df = pd.merge(df_movies, df_ratings, on="movie_id_ml")
df = pd.merge(df, df_users, on="user_id")

df_movie_count_mean = df.groupby(["movie_id_ml", "title"], as_index=False)["rating"].agg(
    ["count", "mean"]).reset_index()

C = df_movie_count_mean["mean"].mean()

m = df_movie_count_mean["count"].quantile(0.9)

df_movies_1 = df_movie_count_mean.copy()

df = pd.merge(df_movies, df_movies_1, on=["movie_id_ml", "title"])


def weighted_rating(x, m=m, C=C):
    """Calculation based on the IMDB formula"""
    v = x['count']
    R = x['mean']
    return (v / (v + m) * R) + (m / (m + v) * C)


# Define a new feature 'score' and calculate its value with `weighted_rating()`
df['score'] = df.apply(weighted_rating, axis=1)
# Sort movies based on score calculated above
df = df.sort_values('score', ascending=False).reset_index()

df_cbr = pd.DataFrame()

# handle cast
limit_cast_num = 10
df_cbr['cast'] = df['cast'].apply(
    lambda x: [''.join(i['cast_name'].split(",")[::-1]) for i in x] if isinstance(x, list) else [])
df_cbr['cast'] = df_cbr['cast'].apply(lambda x: x[:limit_cast_num] if len(x) >= limit_cast_num else x)
df_cbr['cast'] = df_cbr['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# handle genres
df_cbr['genre'] = df['genres']

# handle title
df_cbr['title'] = df['title']

# merge all
df_cbr['mixed'] = df_cbr['cast'] + df_cbr['genre']
df_cbr['mixed'] = df_cbr['mixed'].apply(lambda x: ' '.join(x))

count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(df_cbr['mixed'])
count_matrix.todense()

cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(df_cbr.index, index=df_cbr['title'])
titles = df_cbr['title']


def do_recommender(input_list):
    if len(input_list) != 1:
        raise ValueError("Expected input list of length 1.")
    rec_result = get_recommendations(input_list[0])
    return [str(rec_result)]


def get_recs_for_idx(idx):
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:21]

    movie_indices = [i[0] for i in similarity_scores]
    similar_scores = [i[1] for i in similarity_scores]
    similar_scores = pd.Series(similar_scores, index=movie_indices)
    titlesID = titles.iloc[movie_indices]

    df_titles = titlesID.to_frame()
    df_titles = df_titles.rename_axis('id')
    df_scores = similar_scores.to_frame()
    df_scores = df_scores.rename_axis('id')
    final = df_titles.merge(df_scores, left_on='id', right_on='id')
    final = final.drop_duplicates('title')
    return final.dropna()


def get_recommendations(title):
    idxs = indices[title]
    if "int64" in str(type(idxs)):
        return get_recs_for_idx(idxs)
    elif "Series" in str(type(idxs)):
        accum = pd.Series(dtype=object)
        for idx in idxs:
            currentSeries = get_recs_for_idx(idx)
            accum = pd.concat([currentSeries, accum])
            accum = accum.drop_duplicates('title')
        accum = accum[accum != title]
        return accum.dropna()
    else:
        raise TypeError("Unrecognized index type (expected int64 or Series)")


# example, replace with other movie title
# print(get_recommendations("spawn"))
# print(do_recommender(["spawn"]))
