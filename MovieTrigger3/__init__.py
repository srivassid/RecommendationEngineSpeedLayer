import json
import logging
import pandas as pd
import azure.functions as func
from shared_code.functions import SpeedLayer

def main(event: func.EventGridEvent):
    result = json.dumps({
        'id': event.id,
        'data': event.get_json(),
        'topic': event.topic,
        'subject': event.subject,
        'event_type': event.event_type,
    })
    
    logging.info('Python EventGrid trigger processed an event: %s', result)
    # d = {'a':[1,2], 'b':[3,4]}
    # df = pd.DataFrame(data=d)
    # print(df)
    movie = event.get_json()['movie']
    speed_obj = SpeedLayer()
    empty = speed_obj.check_duplicates(movie)
    if empty is True:
        logging.info('empty')
    else:
        logging.info('not empty')



