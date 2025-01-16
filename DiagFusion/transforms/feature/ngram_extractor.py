import public_function as pf
from sklearn.feature_extraction.text import CountVectorizer


def extract_ngram(all_texts, store_path, n1=2, n2=2):
    NGram_X = CountVectorizer(analyzer='word', ngram_range=(2, 2)).fit_transform(all_texts).toarray()
    nor_NGram_X = pf.min_max_normalized(NGram_X)
    NGram_X_path = store_path + f'/{ n1 }_{ n2 }_NGram.pkl'
    nor_BOW_X_path = store_path + f'/{ n1 }_{ n2 }_nor_NGram_X.pkl'
    pf.save(NGram_X_path, NGram_X)
    pf.save(nor_BOW_X_path, nor_NGram_X)


if __name__ == '__main__':
    config = pf.get_config()
    # 读取texts
    text_path = config['text_path'] + '/all_texts.pkl'
    all_texts = pf.load(text_path)
    extract_ngram(all_texts, config['feature_path'])