import public_function as pf
from sklearn.feature_extraction.text import CountVectorizer


def extract_bow(all_texts, store_path):
    BOW_X = CountVectorizer().fit_transform(all_texts).toarray()
    nor_BOW_X = pf.min_max_normalized(BOW_X)
    BOW_X_path = store_path + '/BOW.pkl'
    nor_BOW_X_path = store_path + '/nor_BOW_X.pkl'
    pf.save(BOW_X_path, BOW_X)
    pf.save(nor_BOW_X_path, nor_BOW_X)


if __name__ == '__main__':
    config = pf.get_config()
    # 读取texts
    text_path = config['text_path'] + '/all_texts.pkl'
    all_texts = pf.load(text_path)
    extract_bow(all_texts, config['feature_path'])