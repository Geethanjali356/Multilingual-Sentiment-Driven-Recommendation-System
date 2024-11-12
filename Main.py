import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Step 1: Load Data
# Placeholder for multilingual text and ratings
# sentiment_data: user_id, item_id, review_text, sentiment_label (0=negative, 1=positive)
# interaction_data: user_id, item_id, rating
sentiment_data = pd.read_csv('sentiment_data.csv')
interaction_data = pd.read_csv('interaction_data.csv')

# Step 2: Preprocess Text Data
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(sentiment_data['review_text'])

X = tokenizer.texts_to_sequences(sentiment_data['review_text'])
X = pad_sequences(X, maxlen=200)  # Padding sequences

y = sentiment_data['sentiment_label'].values

# Split for sentiment analysis
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Sentiment Analysis Model
input_text = Input(shape=(200,))
embedding = Embedding(input_dim=50000, output_dim=128)(input_text)
lstm = LSTM(128, return_sequences=False)(embedding)
dropout = Dropout(0.2)(lstm)
sentiment_output = Dense(1, activation='sigmoid')(dropout)

sentiment_model = Model(inputs=input_text, outputs=sentiment_output)
sentiment_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Sentiment Model
sentiment_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)

# Step 4: Integrate Sentiment for Recommendations
# Merge sentiment scores into interaction data
sentiment_scores = sentiment_model.predict(X)
sentiment_data['sentiment_score'] = sentiment_scores

# Merge sentiment scores with interaction data
merged_data = pd.merge(interaction_data, sentiment_data[['user_id', 'item_id', 'sentiment_score']], on=['user_id', 'item_id'])

# Create user-item-sentiment matrix
user_item_matrix = merged_data.pivot(index='user_id', columns='item_id', values='sentiment_score').fillna(0).values

# Step 5: Recommendation Model (Matrix Factorization)
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

user_embedding = Embedding(input_dim=user_item_matrix.shape[0], output_dim=50)(user_input)
item_embedding = Embedding(input_dim=user_item_matrix.shape[1], output_dim=50)(item_input)

user_vec = Flatten()(user_embedding)
item_vec = Flatten()(item_embedding)

concat = Concatenate()([user_vec, item_vec])
dense = Dense(128, activation='relu')(concat)
dense = Dropout(0.2)(dense)
recommendation_output = Dense(1)(dense)

recommendation_model = Model(inputs=[user_input, item_input], outputs=recommendation_output)
recommendation_model.compile(optimizer='adam', loss='mse')

# Prepare data for recommendation training
users = merged_data['user_id'].astype('category').cat.codes.values
items = merged_data['item_id'].astype('category').cat.codes.values
ratings = merged_data['rating'].values

X_train, X_val, y_train, y_val = train_test_split(np.array(list(zip(users, items))), ratings, test_size=0.2, random_state=42)

recommendation_model.fit([X_train[:, 0], X_train[:, 1]], y_train, validation_data=([X_val[:, 0], X_val[:, 1]], y_val), epochs=5, batch_size=64)

# Final Step: Make Recommendations
def recommend(user_id, top_k=5):
    user_idx = merged_data['user_id'].astype('category').cat.categories.tolist().index(user_id)
    item_ids = merged_data['item_id'].astype('category').cat.codes.unique()
    
    predictions = recommendation_model.predict([np.array([user_idx]*len(item_ids)), item_ids])
    top_items = item_ids[np.argsort(predictions.flatten())[-top_k:]]
    
    return merged_data['item_id'].astype('category').cat.categories[top_items]

# Example Usage
recommend(user_id=1)
