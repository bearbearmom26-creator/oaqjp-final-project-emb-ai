{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab560
\pard\pardeftab560\slleading20\partightenfactor0

\f0\fs26 \cf0 import requests\
import json\
\
def emotion_detector(text_to_analyze):\
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"\
    headers = \{"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"\}\
    data = \{ "raw_document": \{ "text": text_to_analyze \} \}\
\
    response = requests.post(url, json=data, headers=headers)\
\
    if response.status_code == 400:\
        return \{\
            'anger': None,\
            'disgust': None,\
            'fear': None,\
            'joy': None,\
            'sadness': None,\
            'dominant_emotion': None\
        \}\
\
    result = json.loads(response.text)\
\
    emotions = result['emotionPredictions'][0]['emotion']\
    anger = emotions['anger']\
    disgust = emotions['disgust']\
    fear = emotions['fear']\
    joy = emotions['joy']\
    sadness = emotions['sadness']\
\
    # Get the dominant emotion\
    emotion_scores = \{\
        'anger': anger,\
        'disgust': disgust,\
        'fear': fear,\
        'joy': joy,\
        'sadness': sadness\
    \}\
\
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)\
\
    return \{\
        'anger': anger,\
        'disgust': disgust,\
        'fear': fear,\
        'joy': joy,\
        'sadness': sadness,\
        'dominant_emotion': dominant_emotion\
    \}}
