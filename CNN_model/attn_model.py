from attn_helper_models import KerasTextClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, accuracy_score

train_ds = pd.read_csv('CNN_model/data/train_synthetic.csv', sep=',', encoding = "ISO-8859-1")

X_train = list(train_ds['question'].values)
y_train = list(train_ds['operation'].values)


test_ds = pd.read_csv('CNN_model/data/iit_test.csv', sep=',', encoding = "ISO-8859-1")

X_test = list(test_ds['question'].values)
y_test = list(test_ds['operation'].values)


kclf = KerasTextClassifier(input_length=100, n_classes=4, max_words=15000)

kclf.fit(X=X_train, y=y_train,  batch_size=128, lr=0.01, epochs=15)

y_pred = kclf.predict(X_test)


label_idx_to_use = [i for i, c in enumerate(list(kclf.encoder.classes_))]
label_to_use = list(kclf.encoder.classes_)
print(label_to_use)

print(classification_report(kclf.encoder.transform(y_test), y_pred,
                            target_names=label_to_use,
                            labels=label_idx_to_use))
