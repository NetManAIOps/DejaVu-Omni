import numpy as np
import public_function as pf


def concatenate_feature(store_path, n1=2, n2=2):
    nor_TFIDF_X = pf.load(store_path + '/nor_TFIDF_X.pkl')
    nor_BOW_X = pf.load(store_path + '/nor_BOW_X.pkl')
    nor_NGram_X = pf.load(store_path + f'/{ n1 }_{ n2 }_NGram_X.pkl')
    TFIDF_BOW_NGram_X = np.concatenate((nor_TFIDF_X, nor_BOW_X, nor_NGram_X), axis=1)
    TFIDF_BOW_X = np.concatenate((nor_TFIDF_X, nor_BOW_X), axis=1)
    TFIDF_NGram_X = np.concatenate((nor_TFIDF_X, nor_NGram_X), axis=1)
    BOW_NGram_X = np.concatenate((nor_BOW_X, nor_NGram_X), axis=1)
    pf.save(store_path + '/TFIDF_BOW_NGram_X.pkl', TFIDF_BOW_NGram_X)
    pf.save(store_path + '/TFIDF_BOW_X.pkl', TFIDF_BOW_X)
    pf.save(store_path + '/TFIDF_NGram_X.pkl', TFIDF_NGram_X)
    pf.save(store_path + '/BOW_NGram_X.pkl', BOW_NGram_X)


if __name__ == '__main__':
    config = pf.get_config()
    concatenate_feature(config['feature_path'])
