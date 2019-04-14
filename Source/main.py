#%% Data Pre-processing

import pandas as pd
import numpy as np
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

column_means = pd.concat([approved_drugs, non_approved_drugs, withdrawn_drugs])._get_numeric_data().mean()
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

x_train_an, x_test_an, y_train_an, y_test_an, ro5_test_an, ids_test_an = preprocess_dataset(approved_non_approved)
x_train_aw, x_test_aw, y_train_aw, y_test_aw, ro5_test_aw, ids_test_aw = preprocess_dataset(approved_withdrawn)

# Note: Approved vs. Non approved... 1 = approved, 0 = non approved
#       Approved vs. Withdrawn... 1 = approved, 0 = withdrawn

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

#%% Predictions
predictions_prob = best_estimator_an.predict_proba(x_test_an)
top_approved_order = np.argsort(predictions_prob[:, 1])[::-1][:5]
top_non_approved_order = np.argsort(predictions_prob[:, 1])[::1][:5]
top_approved_ids = np.array(ids_test_an)[top_approved_order]
top_non_approved_ids = np.array(ids_test_an)[top_non_approved_order]

predictions_prob = best_estimator_aw.predict_proba(x_test_aw)
top_approved_order = np.argsort(predictions_prob[:, 1])[::-1][:5]
top_withdrawn_order = np.argsort(predictions_prob[:, 1])[::1][:5]
print(np.array(ids_test_aw)[top_approved_order])
print(np.array(ids_test_aw)[top_withdrawn_order])

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


#%% Rule of 5 (Approved = less than 2 violations)... Approved = 1, Non approved = 0

ro5_approved = y_test_an[ro5_test_an < 2]

precision = sum(ro5_approved == 1) / len(ro5_approved)
recall = sum(ro5_approved == 1) / len(y_test_an == 1)
f1 = 2*(precision*recall/(precision+recall))

print(precision)
print(recall)
print(f1)


ro5_approved = y_test_aw[ro5_test_aw < 2]

precision = sum(ro5_approved == 1) / len(ro5_approved)
recall = sum(ro5_approved == 1) / len(y_test_aw == 1)
f1 = 2*(precision*recall/(precision+recall))

print(precision)
print(recall)
print(f1)

#%% Plot Best Models against each other
import matplotlib.pyplot as plt

plt.bar((1, 2, 3, 4),
        (0.784, 0.823, 0.832, 0.769),
        yerr=(0.02, 0.021, 0.021, 0.02), width=0.25,
        color='seagreen', label='Approved vs. Non-approved')

plt.bar((1.25, 2.25, 3.25, 4.25),
        (0.516, 0.609, 0.687, 0.667),
        yerr=(0.096, 0.079, 0.044, 0), width=0.25,
        color='lightsalmon', label='Approved vs. Withdrawn')

plt.xticks((1, 2, 3, 4),
           ('Linear Regression', 'Support Vector Machine', 'Random Forest', 'Multilayer Perceptron'),
           rotation=8)
plt.ylim(0, 1)
plt.ylabel('F Score')
plt.legend(loc='best')
plt.show()