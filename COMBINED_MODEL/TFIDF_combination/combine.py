from main_tfidf import *
from CNN_mod_tst import *

tfidf_cls = TFIDF(0.5, 10, 30)

print('Retrieving data')
def retrieve_data():
    train_ds = pd.read_csv('COMBINED_MODEL/TFIDF_combination/data/train_synthetic.csv', sep=',', encoding = "ISO-8859-1")

    X_train = list(train_ds['question'].values)
    y_train = list(train_ds['operation'].values)

    test_ds = pd.read_csv('COMBINED_MODEL/TFIDF_combination/data/reformatted_iit.csv', sep=',', encoding = "ISO-8859-1")

    X_test = list(test_ds['question'].values)
    y_test = list(test_ds['operation'].values)

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = retrieve_data()

print(len(X_test))

tfidf_cls.fit(X_train, y_train)

y_pred, X_left = tfidf_cls.predict(X_test)

print('Multiplication' in y_pred)

TOTALCORRECT = 0

for tfidf_pred in range(len(y_pred)):
    print('{}/{}'.format(tfidf_pred, len(y_test)))
    currTruthValue = y_test[tfidf_pred]
    if y_pred[tfidf_pred] != -1:
        if y_pred[tfidf_pred] == currTruthValue:
            TOTALCORRECT += 1
    else:
        CNNPredictedOp = CNN(X_test[tfidf_pred], True)
        if CNNPredictedOp == currTruthValue:
            TOTALCORRECT += 1


print('Accuracy', TOTALCORRECT/len(y_test))
