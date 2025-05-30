import pandas as pd

FORBIDDEN_WORDS = ['ive', 'also', 'could','also','make','feel','feeling','even','much','take','years','since','life','need',
             'anything','anyone','way','something','someone','cant','cannot','last','went','didnt']

def load_data(src, index_col=0):
    return pd.read_csv(src, index_col=index_col)

def get_unique_words(data, group_column='status', text_column='cleaned_text'):
    unique_words = {}

    for status, group in data.groupby(group_column):
        words = {}
        
        for _, row in group.iterrows():
            cleaned_text = row[text_column]
            
            if isinstance(cleaned_text, str):
                
                statement_words = cleaned_text.split(' ')
                
                for word in statement_words:
                    if word != '':
                        if word in words:
                            words[word] += 1
                        else:
                            words[word] = 1
        
        unique_words[status] = words

    return unique_words

def get_top_normal_words(unique_words, top=20, normal_category_name='Normal'):
    top_normal_words = sorted(unique_words[normal_category_name].items(), key=lambda x: x[1], reverse=True)[:top]
    top_normal_words = [w[0] for w in top_normal_words]

    return top_normal_words

def turn_into_rows(unique_words, forbidden_words=[], top_normal_words=[]):
    rows = []

    for status in unique_words.keys():
        for word, count in unique_words[status].items():
            if word not in top_normal_words + forbidden_words:
                rows.append([status, word, count])

    return rows