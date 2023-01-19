import flask

app = flask.Flask(__name__)


@app.route("/recommendations")
def main_page():
    recommendations= [{"title": "GoldenEye",
                       "image_url": "https://images-na.ssl-images-amazon.com/images/M/MV5BMzk2OTg4MTk1NF5BMl5BanBnXkFtZTcwNjExNTgzNA@@..jpg"},
                      {"title": "Desperado",
                       "image_url": "https://images-na.ssl-images-amazon.com/images/M/MV5BYjA0NDMyYTgtMDgxOC00NGE0LWJkOTQtNDRjMjEzZmU0ZTQ3XkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg"},
                      {"title": "Four Rooms",
                       "image_url": "https://images-na.ssl-images-amazon.com/images/M/MV5BNDc3Y2YwMjUtYzlkMi00MTljLTg1ZGMtYzUwODljZTI1OTZjXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg"},
                      {"title": "Mad Love",
                       "image_url": "https://images-na.ssl-images-amazon.com/images/M/MV5BNDE0NTQ1NjQzM15BMl5BanBnXkFtZTYwNDI4MDU5..jpg"},
                      {"title": "The Aristocats",
                       "image_url": "https://images-na.ssl-images-amazon.com/images/M/MV5BMTU1MzM0MjcxMF5BMl5BanBnXkFtZTgwODQ0MzcxMTE@..jpg"},
                      {"title": "Life of Brian",
                       "image_url": "https://images-na.ssl-images-amazon.com/images/M/MV5BMzAwNjU1OTktYjY3Mi00NDY5LWFlZWUtZjhjNGE0OTkwZDkwXkEyXkFqcGdeQXVyMTQxNzMzNDI@..jpg"}]

    return flask.render_template("recommendations.html", recommendations=recommendations)