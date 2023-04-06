import re
from scipy.sparse.linalg import svds
from os import error
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def recommend_movie(movie_names, ratings, genres):
    
    new_df = pd.read_csv('Movies dataset_v1.1.csv')

    # Creating a pivot table with userId as rows, movieId as columns, and rating as values
    svd_input_matrix = new_df.pivot_table(
        index='userId', columns='movieId', values='rating', fill_value=0)

    movie_dict = dict(zip(new_df['movie_name'], new_df['movieId']))

    # Map the movie names from the parameters list to their respective movieIds using the movie_dict
    mapped_movieIds = [movie_dict.get(movie_name) for movie_name in movie_names]

    # create a dictionary to store movieIds and their corresponding ratings
    user_ratings = dict(zip([movie_dict[movie] for movie in movie_names], ratings))
    user_ratings_df = pd.DataFrame.from_dict(user_ratings, orient='index', columns=['rating']).reset_index().rename(columns={'index': 'movieId'})
    user_ratings_df = user_ratings_df.transpose()
    new_row = pd.Series(user_ratings_df.loc['rating'].values, index=user_ratings_df.loc['movieId'].values, name=611)

    # Add the new row to the svd_input_matrix
    svd_input_matrix = svd_input_matrix.append(new_row)
    svd_input_matrix.columns = svd_input_matrix.columns.astype('int64')
    svd_input_matrix.columns.name = 'movieId'
    svd_input_matrix = svd_input_matrix.fillna(0)
    
    # create the matrix
    ratings = np.array(svd_input_matrix)

    # calculate the SVD
    U, sigma, Vt = svds(ratings, k=len(user_ratings))

    # transform sigma into a diagonal matrix
    sigma = np.diag(sigma)

    # calculate the predicted ratings
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    movie_names = svd_input_matrix.columns
    user_ids = svd_input_matrix.index
    predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_ids, columns=movie_names)

    # calculate cosine similarity between the last user and all other users
    similarities = cosine_similarity(predicted_ratings_df.iloc[-1].values.reshape(1,-1), predicted_ratings_df.drop(611).values)

    # find the row of the most similar user
    most_similar_user = predicted_ratings_df.index[np.argmax(similarities)]

    # sorting the list of the movies rated by our most similar user
    top_movies = predicted_ratings_df.loc[most_similar_user].sort_values(ascending=False)

    #removing the movies already rated by the input user from the most_similar_user row
    top_movies = top_movies.drop(list(user_ratings.keys()), errors='ignore')

    top_movies_df = pd.DataFrame(top_movies.reset_index())
    top_movies_df.columns = ['movieId', 'predicted_evaluation']

    movie_names = list(new_df['movie_name'])
    for key in user_ratings:
        if key not in new_df['movieId'].unique():
            user_ratings[key] = "Not present"
        else:
            user_ratings[key] = movie_names[new_df.loc[new_df['movieId'] == key].index[0]]

    user_input_movies_df = pd.DataFrame.from_dict(user_ratings, orient='index', columns=['movie_name'])
    user_input_movies_df = user_input_movies_df[user_input_movies_df['movie_name'] != 'Not present']

    usermoviewithplot = user_input_movies_df.merge(new_df, on='movie_name', how='left')
    usermoviewithplot = usermoviewithplot[['movie_name', 'Plot']].drop_duplicates()

    # Create a dictionary to map movieId to movie_name
    movie_dict = dict(zip(new_df['movieId'], new_df['movie_name']))

    # Replace the movieId with its corresponding movie_name value
    top_movies_df['movieId'] = top_movies.index.map(movie_dict)

    # create a dictionary mapping movie names to genres
    movie_genres = dict(new_df[['movie_name', 'genres']].drop_duplicates().values)

    # add a new 'genres' column to the top_movies dataframe
    top_movies_df['genres'] = top_movies_df['movieId'].map(movie_genres)

    top_movies_df.columns = ['movie_name', 'predicted_evaluation', 'genres']

    disliked_genres = genres[3:]
    preferred_genres = genres[:3]

    # Filter out movies that the user dislikes
    disliked_movies = top_movies_df[top_movies_df['genres'].apply(lambda x: any(genre in disliked_genres for genre in x.split("|")))]

    # Remove movies present in disliked movies from top_movies_df
    top_movies_df = top_movies_df[~top_movies_df['movie_name'].isin(disliked_movies['movie_name'])]

    # Filter out movies that don't match the user's preferred genres
    preferred_movies = top_movies_df[top_movies_df['genres'].apply(lambda x: any(genre in preferred_genres for genre in x.split("|")))]

    # Include movies in preferred_movies in top_movies_df
    top_movies_df = pd.concat([top_movies_df, preferred_movies])

    # Multiply the predicted_evaluation of movies in preferred_movies by 3
    preferred_movies['predicted_evaluation'] = preferred_movies['predicted_evaluation'] * 3

    top_movies_df = top_movies_df.sort_values('predicted_evaluation', ascending=False)
    top_movies_df = top_movies_df.drop_duplicates(subset='movie_name', keep='first')

    #Storing only the top 25 movies in our CF output dataframe
    top_movies_df = top_movies_df[:25]

    # create a dictionary mapping movie names to genres
    movie_plot = dict(new_df[['movie_name', 'Plot']].drop_duplicates().values)

    # add a new 'genres' column to the top_movies dataframe
    top_movies_df['Plot'] = top_movies_df['movie_name'].map(movie_plot)

    def preprocess(text):

        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        text = ' '.join(tokens)

        return text

    # Apply the preprocessing function to the 'Plot' column
    usermoviewithplot['Plot'] = usermoviewithplot['Plot'].apply(preprocess)
    top_movies_df['Plot'] = top_movies_df['Plot'].apply(preprocess)
    
    # Creating the BOW vectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the plot column in user_interesting_movies dataframe using BOW vectorizer
    bow_matrix1 = vectorizer.fit_transform(usermoviewithplot["Plot"])

    # Transform the plot column in top 25 movies dataframe using the same vectorizer
    bow_matrix2 = vectorizer.transform(top_movies_df['Plot'])

    # Compute the cosine similarity between each row in bow_matrix1 and bow_matrix2
    similarity_matrix = cosine_similarity(bow_matrix2, bow_matrix1)

    # Choosing the similarity value of a single movie from the top 25 list by calculating the average of all similarity values obtained. 
    m = np.array(similarity_matrix)
    row_sums = m.sum(axis=1)
    row_average = (row_sums/len(user_input_movies_df))

    top_movies_df["Similarity"] = row_average

    top_movies_df = top_movies_df.sort_values('Similarity', ascending=False)
    top_movies_df = top_movies_df[:10]
    top_movies_df = top_movies_df.loc[:, ['movie_name', 'Similarity', 'predicted_evaluation', 'genres']]

    print(top_movies_df)
    
    recommendation = top_movies_df.iloc[:, 0].tolist()
    return recommendation
