# Movie-Roulette

A hybrid movie recommender chatbot based on Python and Dialogflow which recommends 10 movies after getting the below items as input in the chatbot:

  1. Get 5 movie names that seem interesting to our user.
  2. Ask them to rate the 5 movies on a scale of 0.5 - 5.0 (eg: 0.5, 1.0, 1.5, etc)
  3. Ask the user to to mention their 3 favorite and disliked genres.

## Requirements to execute the python modules and integrate with your dialogflow chatbot:

1. Flask (Latest version) Refer: https://flask.palletsprojects.com/en/2.2.x/installation/#install-flask
2. Ngrok (Free version and you should sign up with ngrok to get a tunnel service) Refer: https://ngrok.com/download
3. Python (Dependencies used in both app.py and recommender.py should be resolved using pip)
4. Movies dataset (Plot summaries Web scraped from IMdb and other values from MovieLens Small dataset)
5. Dialogflow ES

Chatbot Interaction:

![image](https://user-images.githubusercontent.com/129698277/230423141-441bcb48-b5b6-48c5-8fba-d3171e539666.png)

Try our chatbot here: https://bot.dialogflow.com/14359829-8586-4701-9d1e-4a6db5662003

## Processes involved in movie recommendation:-

## Collaborative Filtering:
Using the SVD algorithm based on multiple user ratings to find the most similar user who exhibits the same rating behavior as the input user.

## Genre-based Filtering:
Creating a movie list with values from the SVD algorithm and removing all movies belonging to the user's disliked genres. Multiplying the values of movies belonging to the user's favorite genres in the list by 3 to give them a higher preference in the list.

## Content-based Filtering:
Creating a list containing the top 25 movies sorted by their SVD output values in descending order. Then, we compare the similarity between the plot summaries of the top 25 movies and the plot summaries of the 5 interesting movies given by the input user using the cosine similarity algorithm.

## Preparation of Recommendation List:
Finally, we sort the top 25 movie list with the similarity values obtained in descending order and condense them into a final list of 10 movies. This list will be recommended to the user as a chatbot response.
    
Special thanks to Mr. Brain Fritz for maintaining the OMDB API which helped our dataset prepration.
Check out: http://www.omdbapi.com/
