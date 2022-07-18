import pandas as pd
import json
from nltk.stem.snowball import SnowballStemmer
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

# handle keyword
stemmer = SnowballStemmer('english')
df_cbr['keyword'] = df['keyword'].apply(lambda x: eval(x))
df_cbr['keyword'] = df_cbr['keyword'].apply(lambda x: [stemmer.stem(i) for i in x])
df_cbr['keyword'] = df_cbr['keyword'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
# handle genres
df_cbr['genre'] = df['genres']

# handle title
df_cbr['title'] = df['title']

# merge all
df_cbr['mixed'] = df_cbr['keyword'] + df_cbr['cast'] + df_cbr['genre']
df_cbr['mixed'] = df_cbr['mixed'].apply(lambda x: ' '.join(x))

s = df_cbr.apply(lambda x: pd.Series(x['keyword']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]

count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(df_cbr['mixed'])
count_matrix.todense()

cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(df_cbr.index, index=df_cbr['title'])
titles = df_cbr['title']


def get_recommendations(title):
    idx = indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:21]
    movie_indices = [i[0] for i in similarity_scores]
    return titles.iloc[movie_indices]

# example, replace with other movie title
# "Spawn" (1997) doesn't work however
print(get_recommendations('the princess bride'))