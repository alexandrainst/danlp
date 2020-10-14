

import time

def print_speed_performance(start, num_sentences, num_tokens=None):

    span = time.time() - start
    sent_per_sec = int(num_sentences/span)
    span = round(span, 5)
    if not num_tokens == None :
        speed  = "Made predictions on {} sentences and {} tokens in {}s (~{} sentences per second)"
        speed = speed.format(num_sentences, num_tokens, span, sent_per_sec)
    else: 
        speed  = "Made predictions on {} sentences in {}s (~{} sentences per second)"
        speed = speed.format(num_sentences, span, sent_per_sec)
    print(speed)


def sentiment_score_to_label(score):
    if score == 0:
        return 'neutral'
    if score < 0:
        return 'negativ'
    else:
        return 'positiv'

def sentiment_score_to_label_sentida(score):
    # the threshold of 0.4 is fitted on a manually annotated twitter corpus for sentiment on 1327 examples
    if score > 0.4:
        return 'positiv'
    if score < -0.4:
        return 'negativ'
    else:
        return 'neutral'
