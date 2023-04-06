from flask import Flask, request, jsonify
from recommender import recommend_movie

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(force=True)
    intent_name = req['queryResult']['intent']['displayName']
        
    if intent_name == 'Get_Interesting_Movies_Genres':

        movie_names = req['queryResult']['outputContexts'][2]['parameters']['movies']
        ratings = req['queryResult']['outputContexts'][2]['parameters']['no']
        genres = req['queryResult']['outputContexts'][2]['parameters']['genres']
        
        response_text = recommend_movie(movie_names, ratings, genres)
        movie_list = ", ".join(response_text)
        response = {
            'fulfillmentMessages': [
                {
                    'text': {
                        'text': [
                            'Here are the movies recommended for you! I hope you like them:'
                        ]
                    }
                },
                {
                    'text': {
                        'text': [
                            movie_list
                        ]
                    }
                }
            ]
        }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
