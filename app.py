import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("ratings.csv")

# Encode user and movie IDs
user_ids = df["userId"].unique().tolist()
movie_ids = df["movieId"].unique().tolist()

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}

df["user"] = df["userId"].map(user_to_user_encoded)
df["movie"] = df["movieId"].map(movie_to_movie_encoded)

num_users = len(user_ids)
num_movies = len(movie_ids)

# Prepare training data
X = df[["user", "movie"]].values
y = df["rating"].values

# Normalize ratings
y = y / 5.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Model
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size=50):
        super(RecommenderNet, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_size
        )
        self.movie_embedding = tf.keras.layers.Embedding(
            num_movies, embedding_size
        )
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        dot_product = tf.reduce_sum(user_vector * movie_vector, axis=1)
        x = self.dense1(tf.expand_dims(dot_product, 1))
        return self.dense2(x)

model = RecommenderNet(num_users, num_movies)

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["mae"]
)

# Train model
model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=5,
    validation_data=(X_test, y_test)
)

# Predict rating for a user-movie pair
sample_user = 1
sample_movie = 10

encoded_user = user_to_user_encoded.get(sample_user)
encoded_movie = movie_to_movie_encoded.get(sample_movie)

prediction = model.predict(np.array([[encoded_user, encoded_movie]]))
print(f"Predicted Rating: {prediction[0][0] * 5:.2f}")