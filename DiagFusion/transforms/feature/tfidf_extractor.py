import public_function as pf
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_tfidf(all_texts, store_path):
    TFIDF_X = TfidfVectorizer().fit_transform(all_texts).toarray()
    nor_TFIDF_X = pf.min_max_normalized(TFIDF_X)
    TFIDF_X_path = store_path + '/TFIDF.pkl'
    nor_TFIDF_X_path = store_path + '/nor_TFIDF_X.pkl'
    pf.save(TFIDF_X_path, TFIDF_X)
    pf.save(nor_TFIDF_X_path, nor_TFIDF_X)


if __name__ == '__main__':
    config = pf.get_config()
    # 读取texts
    text_path = config['text_path'] + '/all_texts.pkl'
    all_texts = pf.load(text_path)
    extract_tfidf(all_texts, config['feature_path'])
