import json
import logging
import pandas as pd
import azure.functions as func
from gremlin_python.driver import client, serializer
from imdb import IMDb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

def check_duplicates(cosmos_client, movie):
        query = "g.V().has('name', '" + movie + "')"
        callback = cosmos_client.submitAsync(query)
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

def pre_process_plot_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    VERB_CODES = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

    text = text.lower()
    temp_sent = []
    words = nltk.word_tokenize(text)
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in VERB_CODES:
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        if lemmatized not in stop_words and lemmatized.isalpha():
            temp_sent.append(lemmatized)
    finalsent = ' '.join(temp_sent)
    finalsent = finalsent.replace("n't", " not")
    finalsent = finalsent.replace("'m", " am")
    finalsent = finalsent.replace("'s", " is")
    finalsent = finalsent.replace("'re", " are")
    finalsent = finalsent.replace("'ll", " will")
    finalsent = finalsent.replace("'ve", " have")
    finalsent = finalsent.replace("'d", " would")
    return finalsent

def get_data(cosmos_client):
    counter = 0
    df = pd.DataFrame(columns={'movie','plot', 'name'})
    while(counter == 0):
        try:
            query = "g.V().valueMap(true)"
            callback = cosmos_client.submitAsync(query)
            for result in callback.result():
                for i in result:
                    print(i)
                    df = df.append({'name':i['name'][0], 'plot':i['plot'][0], 'movie':i['id']}, ignore_index=True)
            counter = 1
        except Exception as e:
            print('exception ' + str(e))
    return df
    
def plot(x):
    ia = IMDb()
    try:
        return ia.get_movie(ia.search_movie(x)[0].movieID)['plot'][0].split('::')[0] + \
            (ia.get_movie(ia.search_movie(x)[0].movieID)['title'])
    except Exception as e:
        return x

def get_plot(movie):
    df = pd.DataFrame(columns = {'movie','plot'})
    df = df.append({'movie':movie}, ignore_index=True)
    df['plot'] = df['movie'].apply(lambda x: plot(x))
    df['plot'] = df['plot'].apply(lambda x: pre_process_plot_text(x))
    df['movie'] = df['movie'].apply(lambda x: x.replace("'","") + "_" + str(random.randint(1,1000000)))
    df['name'] = df['movie']
    return df

def send_data(df):
    df = df[~df.movie.str.startswith('Q')]
    df = df.set_index('movie')
    tfidfvec = TfidfVectorizer()
    tfidf_movieid = tfidfvec.fit_transform((df["plot"]))
    cos_sim = cosine_similarity(tfidf_movieid, tfidf_movieid)
    return df, cos_sim

def recommendations(df, cosine_sim, movie):
    df_all = pd.DataFrame(columns={'name','plot_processed','rec_mov'})
    curr_movie = df.loc[df['name'].str.startswith(movie)]
    print("current movie")
    print(curr_movie)
    indices = pd.Series(df.index)
    recommended_movies = []
    df['rec_mov'] = np.empty((len(df), 0)).tolist()
    index = indices[indices == curr_movie['name'].iloc[0]].index[0]
    similarity_scores = pd.Series(cosine_sim[index]).sort_values(ascending=False)
    for i,row in similarity_scores.iloc[1:11].iteritems():
        recommended_movies.append((list(df.index)[i], row))
    print(movie, recommended_movies)
    df_all = df_all.append({'id': curr_movie['name'].iloc[0], 'plot_processed': curr_movie['plot'].iloc[0],
                            'rec_mov': recommended_movies, 'name': curr_movie['name'].iloc[0].split('_')[0]}, ignore_index=True)
    print(df_all)
    return df_all

def insert_graph(client, df):
    print(df)
    for i, row in df.iterrows():
        print(row['name'].replace("'",""), row['plot_processed'], row['rec_mov'])

        query = "g.addV('movie').property('id', '" + (row['id']) + \
                "').property('name', '" + str(row['name']) + \
                "').property('plot', '" + str(row['plot_processed']) + \
                "').property('pk', 'pk')"

        callback = client.submitAsync(query)
        if callback.result() is not None:
            try:
                print("\tInserted this vertex:\n\t{0}\n".format(
                callback.result().one()))
            except Exception as e:
                print("exception " + str(e))
        else:
            print("Something went wrong with this query: {0}".format(query))

def create_edges(client, df):
    for i, row in df.iterrows():
        for j in (row['rec_mov']):
            print(j[0],j[1])
            bound = "g.V('" + row['id'] + "').bothE().where(otherV().hasId('" + str(j[0]) + "'))"
            print('bound')
            print(bound)
            callback_bound = client.submitAsync(bound)
            edge = []

            for result in callback_bound.result():
                edge.append(result[0])

            print("Title " + row['id'] + " Recommendation " + j[0])

            if (len(edge) == 0) :
                try:
                    print("Empty in edge")
                    print("Will make a connection")
                    query = "g.V('" + row['id'] + "').addE('recommends').to(g.V('" + str(j[0]) + "')).property('weight'," + str(j[1]) + ")"
                    print('query is')
                    print(query)
                    callback = client.submitAsync(query)
                    if callback.result() is not None:
                        print("\tInserted this edge:\n\t{0}\n".format(
                            callback.result().one()))
                    else:
                        print("There was a problem with thr query\n\t{0}\n".format(query))
                except Exception as e:
                    print("exception " + str(e))
            else:
                print("Edge already exists")

def main(event: func.EventGridEvent):
    result = json.dumps({
        'id': event.id,
        'data': event.get_json(),
        'topic': event.topic,
        'subject': event.subject,
        'event_type': event.event_type,
    })
    
    logging.info('Python EventGrid trigger processed an event: %s', result)
    movie = event.get_json()['movie']

    account = 'wss://account-cosmos-db.gremlin.cosmosdb.azure.com:443/'
    username = '/dbs/GraphDB1/colls/MovieGraph3'
    password = 'np4uDpHJIdpY1JCWHJpLX1QRxFYxVVT4mnE55qIG3MmrFDoqKHmG7Spptp7dxx8LFLD5D6xeOsRjO1YMpyjZtA=='
    msg_serializer = serializer.GraphSONSerializersV2d0()

    cosmos_client = client.Client(account, 'g', username=username,
                                  password=password, message_serializer= msg_serializer)
    
    empty = check_duplicates(cosmos_client, movie)
    
    if empty is True:
        logging.info('empty')
    else:
        logging.info('not empty')
    if empty is True:
        df = get_plot(movie)
        print(df)
    
        current_df = get_plot(movie)
        existing_df = get_data(cosmos_client)
        df = pd.concat([current_df, existing_df],ignore_index=True)
        print("after concatenation")
        print(df)
        sdf, cos_sim = send_data(df)
        rec_df = recommendations(sdf, cos_sim, movie)
        insert_graph(cosmos_client, rec_df)
        create_edges(cosmos_client, rec_df)
