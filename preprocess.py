from preprocessing.data_cleanup import load_raw_data, apply_clean_func, apply_word_count, calc_sentiment, cluster_text
from preprocessing.freq_count import load_data, get_unique_words, get_top_normal_words, turn_into_rows, FORBIDDEN_WORDS

import pandas as pd

def data_cleanup(src, dst_wc, dst_ct):

    df = load_raw_data(src)

    df = apply_clean_func(df)
    df = apply_word_count(df)
    df = calc_sentiment(df)
    df = cluster_text(df)

    columns_to_save = [c for c in df.columns if c not in ['statement', 'cleaned_text']] 

    df[columns_to_save].to_csv(dst_wc)

    df.to_csv(dst_ct)

def get_freq_count(src, dst):

    df = load_data(src)

    unique_words = get_unique_words(df)
    top_normal_words = get_top_normal_words(unique_words)
    rows = turn_into_rows(unique_words, FORBIDDEN_WORDS, top_normal_words)

    pd.DataFrame(rows, columns=['status', 'word', 'count']).to_csv(dst)

if __name__ == '__main__':
    data_cleanup('data/data.csv', 'data/word_count.csv', 'data/data_cleaned.csv')
    get_freq_count('data/data_cleaned.csv', 'data/freq_count.csv')