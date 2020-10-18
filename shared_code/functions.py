import json
import logging
from gremlin_python.driver import client, serializer
import azure.functions as func
from imdb import IMDb
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

class SpeedLayer():


    def __init__():
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('stopwords')
        random.seed(0)

    def main(event: func.EventGridEvent):
        counter = 0
        movie = ''
        try:
            result = json.dumps({
                'id': event.id,
                'data': event.get_json(),
                'topic': event.topic,
                'subject': event.subject,
                'event_type': event.event_type,
            })

            logging.info('Python EventGrid trigger processed an event: %s', event.get_json()['movie'])
            movie = event.get_json()['movie']
            counter = 1
        except Exception as e:
            logging.info('exception occured ' + str(e))
        return movie

    def check_duplicates(movie):
        query = "g.V().has('name', '" + movie + "')"
        callback = client.submitAsync(query)
        empty = False
        if callback.result() is not None:
            # print('row inserted')
            try:
                if len(callback.result().one()) > 0:
                    empty = False
                else:
                    empty = True
                    
            except Exception as e:
                logging.info("exception " + str(e))
        else:
            logging.info("Something went wrong with this query: {0}".format(query))
        return empty

    def get_plot(movie):
        df = pd.Dataframe(columns = {'movie','plot'})
        df = df.append({'movie':movie}, ignore_index=True)
        df['plot'] = df['movie'].apply(lambda x: plot(x))
        return df

    def plot(x):
        ia = IMDb()
        try:
            return ia.get_movie(ia.search_movie(x)[0].movieID)['plot'][0].split('::')[0] + \
                (ia.get_movie(ia.search_movie(x)[0].movieID)['title'])
        except Exception as e:
            return x

    def get_existing_data():
        df = pd.DataFrame()
        share = 'moviedata1'
        file_service = FileService(account_name='storageaccountsidre9829',
                                account_key='LVoJvUD3t9h0GJyWWCZZbwi+AQVX4qF/KaYRuzH+IS56/07dE2qiAqgBMK0W7vLH8BPE3S0t+ZALy0LUt4Fqgw==')
        generator = file_service.list_directories_and_files(share)
        for file_or_dir in generator:
            df_sub = pd.read_csv(file_or_dir.name)
            df = df.append(df_sub)
        return df

    def get_similarities(current_df, all_df):
        df = pd.concat([current_df, all])
        df['movie'] = df['movie'].apply(lambda x: x.replace("'","") + "_" + str(random.randint(1,1000000)))
        df['name'] = df['movie']
        df = df.set_index('movie')
        tfidfvec = TfidfVectorizer()
        tfidf_movieid = tfidfvec.fit_transform((df["plot_processed"]))
        cos_sim = cosine_similarity(tfidf_movieid, tfidf_movieid)
        return df, cos_sim

    def recommendations(df, cosine_sim, movie):
        df_all = pd.DataFrame(columns={'name','plot_processed','rec_mov'})
        indices = pd.Series(df.index)
        recommended_movies = []
        index = indices[indices == movie].index[0]
        similarity_scores = pd.Series(cosine_sim[index]).sort_values(ascending=False)
        top_10_movies = list(similarity_scores.iloc[1:11].index)
        for i,row in similarity_scores.iloc[1:11].iteritems():
            recommended_movies.append((list(df.index)[i], row))

        return mov, recommended_movies

    def insert_graph(client, df):
        movie = movie.replace("'","") + "_" + str(random.randint(1,1000000))
        query = "g.addV('movie').property('id', '" + (row['id']) + \
                "').property('name', '" + str(row['name']) + \
                "').property('plot', '" + str(row['plot_processed']) + \
                "').property('pk', 'pk')"

        callback = client.submitAsync(query)
        if callback.result() is not None:
            # print('row inserted')
            try:
                print("\tInserted this vertex:\n\t{0}\n".format(
                callback.result().one()))
            except Exception as e:
                print("exception " + str(e))
        else:
            print("Something went wrong with this query: {0}".format(query))

    if __name__ == '__main__':
        movie = main()
        empty = check_duplicates(movie)
        if empty is True:
            df = get_plot(movie)
            existing_df = get_existing_data()
            print(df)
            print(existing_df)