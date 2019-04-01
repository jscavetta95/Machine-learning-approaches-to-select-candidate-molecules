#%% Data Pre-processing

import pandas as pd
from sklearn import metrics

from chembl_data_analysis import update_columns, plot_missing_by_feature, \
    plot_missing_by_sample, remove_samples_missing_features, preprocess_dataset

from machine_learning_methods import logistic_regression, support_vector_machine, \
    random_forest, multilayer_perceptron

approved_drugs = update_columns(pd.read_csv("./data/approved_drugs.csv", encoding="UTF-16LE"))
non_approved_drugs = update_columns(pd.read_csv("./data/nonapproved_drugs.csv", encoding="UTF-16LE"))
withdrawn_drugs = update_columns(pd.read_csv("./data/withdrawn_drugs.csv", encoding="UTF-16LE"))

plot_missing_by_feature(approved_drugs, non_approved_drugs, withdrawn_drugs)
plot_missing_by_sample(approved_drugs, non_approved_drugs, withdrawn_drugs)

remove_samples_missing_features(approved_drugs)
remove_samples_missing_features(non_approved_drugs)
remove_samples_missing_features(withdrawn_drugs)

column_means = pd.concat([approved_drugs, non_approved_drugs, withdrawn_drugs]).mean()
approved_drugs.fillna(column_means, inplace=True)
non_approved_drugs.fillna(column_means, inplace=True)
withdrawn_drugs.fillna(column_means, inplace=True)

approved_drugs['Class'] = 'approved'
non_approved_drugs['Class'] = 'non approved'
withdrawn_drugs['Class'] = 'withdrawn'

approved_non_approved = pd.concat([approved_drugs, non_approved_drugs])
approved_non_approved.reset_index(inplace=True, drop=True)

approved_withdrawn = pd.concat([approved_drugs, withdrawn_drugs])
approved_withdrawn.reset_index(inplace=True, drop=True)

x_train_an, x_test_an, y_train_an, y_test_an = preprocess_dataset(approved_non_approved)
x_train_aw, x_test_aw, y_train_aw, y_test_aw = preprocess_dataset(approved_withdrawn)

#%% Logistic Regression

best_estimator_an = logistic_regression(x_train_an, y_train_an)
best_estimator_aw = logistic_regression(x_train_aw, y_train_aw)

predictions = best_estimator_an.predict(x_test_an)
print(metrics.classification_report(y_test_an, predictions))
print(metrics.cohen_kappa_score(y_test_an, predictions))
print(metrics.confusion_matrix(y_test_an, predictions))

predictions = best_estimator_aw.predict(x_test_aw)
print(metrics.classification_report(y_test_aw, predictions))
print(metrics.cohen_kappa_score(y_test_aw, predictions))
print(metrics.confusion_matrix(y_test_aw, predictions))

#%% Support Vector Machine

best_estimator_an = support_vector_machine(x_train_an, y_train_an)
best_estimator_aw = support_vector_machine(x_train_aw, y_train_aw)

predictions = best_estimator_an.predict(x_test_an)
print(metrics.classification_report(y_test_an, predictions))
print(metrics.cohen_kappa_score(y_test_an, predictions))
print(metrics.confusion_matrix(y_test_an, predictions))

predictions = best_estimator_aw.predict(x_test_aw)
print(metrics.classification_report(y_test_aw, predictions))
print(metrics.cohen_kappa_score(y_test_aw, predictions))
print(metrics.confusion_matrix(y_test_aw, predictions))

#%% Random Forest

best_estimator_an = random_forest(x_train_an, y_train_an)
best_estimator_aw = random_forest(x_train_aw, y_train_aw)

predictions = best_estimator_an.predict(x_test_an)
print(metrics.classification_report(y_test_an, predictions))
print(metrics.cohen_kappa_score(y_test_an, predictions))
print(metrics.confusion_matrix(y_test_an, predictions))

predictions = best_estimator_aw.predict(x_test_aw)
print(metrics.classification_report(y_test_aw, predictions))
print(metrics.cohen_kappa_score(y_test_aw, predictions))
print(metrics.confusion_matrix(y_test_aw, predictions))

#%% Multilayer Perceptron (Neural Network)

best_estimator_an = multilayer_perceptron(x_train_an, y_train_an)
best_estimator_aw = multilayer_perceptron(x_train_aw, y_train_aw)

predictions = best_estimator_an.predict(x_test_an)
print(metrics.classification_report(y_test_an, predictions))
print(metrics.cohen_kappa_score(y_test_an, predictions))
print(metrics.confusion_matrix(y_test_an, predictions))

predictions = best_estimator_aw.predict(x_test_aw)
print(metrics.classification_report(y_test_aw, predictions))
print(metrics.cohen_kappa_score(y_test_aw, predictions))
print(metrics.confusion_matrix(y_test_aw, predictions))
