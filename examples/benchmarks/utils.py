

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