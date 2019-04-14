import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def update_columns(df):
    new_df = df.loc[:, ['Molecular Weight','Targets','Bioactivities','AlogP',
                        'PSA','HBA','HBD','#Rotatable Bonds','ACD ApKa',
                        'ACD BpKa','ACD LogP','ACD LogD','Aromatic Rings','Heavy Atoms',
                        'Molecular Weight (Monoisotopic)', '#RO5 Violations', 'ChEMBL ID']]
    new_df = new_df.rename(index=str, columns={'#Rotatable Bonds':'Rotatable Bonds',
                                               'Molecular Weight (Monoisotopic)':'Monoisotopic Mass'})
    return new_df


def plot_missing_by_feature(approved_drugs, non_approved_drugs, withdrawn_drugs):
    approved_drugs = approved_drugs.drop(['ChEMBL ID', '#RO5 Violations'], axis=1)
    non_approved_drugs = non_approved_drugs.drop(['ChEMBL ID', '#RO5 Violations'], axis=1)
    withdrawn_drugs = withdrawn_drugs.drop(['ChEMBL ID', '#RO5 Violations'], axis=1)

    approved_y = approved_drugs.isnull().sum()/len(approved_drugs)
    non_approved_y = non_approved_drugs.isnull().sum()/len(non_approved_drugs)
    withdrawn_y = withdrawn_drugs.isnull().sum()/len(withdrawn_drugs)

    x = np.arange(len(approved_y))
    width = 1/3 - 0.05
    figure, axis = plt.subplots()

    rects_approved = axis.bar(x, list(approved_y), width, color='seagreen')
    rects_non_approved = axis.bar(x + width, list(non_approved_y), width, color='lightsalmon')
    rects_withdrawn = axis.bar(x + width*2, list(withdrawn_y), width, color='steelblue')

    axis.set_ylabel('Proportion of Samples Missing Data')
    axis.set_xticks(x + width)
    axis.set_xticklabels(list(approved_drugs), rotation=90, fontsize=6)

    axis.legend((rects_approved[0], rects_non_approved[0], rects_withdrawn[0]),
                ('Approved', 'Non-approved', 'Withdrawn'))

    figure.show()


def plot_missing_by_sample(approved_drugs, non_approved_drugs, withdrawn_drugs):
    approved_drugs = approved_drugs.drop(['ChEMBL ID', '#RO5 Violations'], axis=1)
    non_approved_drugs = non_approved_drugs.drop(['ChEMBL ID', '#RO5 Violations'], axis=1)
    withdrawn_drugs = withdrawn_drugs.drop(['ChEMBL ID', '#RO5 Violations'], axis=1)

    approved_missing = approved_drugs.isnull().sum(axis=1)
    non_approved_missing = non_approved_drugs.isnull().sum(axis=1)
    withdrawn_missing = withdrawn_drugs.isnull().sum(axis=1)

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(5,8))

    axes[0].hist(approved_missing, color='seagreen')
    axes[1].hist(non_approved_missing, color='lightsalmon')
    axes[2].hist(withdrawn_missing, color='steelblue')

    axes[2].set_xlabel("Number of Features Missing Data")
    axes[1].set_ylabel("Number of Samples")

    axes[0].set_title("Approved")
    axes[1].set_title("Non-approved")
    axes[2].set_title("Withdrawn")

    figure.show()


# Removes samples with greater than x amount of features missing.
def remove_samples_missing_features(df):
    features_missing_data = 5
    df.dropna(thresh=len(df.columns) - features_missing_data, inplace=True)


def biplot(score, coeff, labels=None):
    x = score[:,0]
    y = score[:,1]
    plt.scatter(x * 1.0/(x.max() - x.min()), y * 1.0/(y.max() - y.min()), c=y, alpha=0.2)
    for i in range(coeff.shape[0]):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show()


def preprocess_dataset(df):
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    features = df.drop('Class', axis=1)
    labels = df.loc[:, ['Class']]

    features_resampled, labels_resampled = RandomUnderSampler(random_state=23)\
        .fit_resample(features, labels.values.flatten())

    x_train, x_test, y_train, y_test = train_test_split(features_resampled, labels_resampled, random_state=23,
                                                        test_size=0.2, stratify=labels_resampled)
    y_train = pd.get_dummies(y_train).iloc[:, 0]
    y_test = pd.get_dummies(y_test).iloc[:, 0]

    ro5_test = x_test[:, 15]
    ids_test = x_test[:, 16]
    x_train = np.delete(x_train, [15, 16], 1)
    x_test = np.delete(x_test, [15, 16], 1)

    mean_variance_scaler = StandardScaler()
    x_train = mean_variance_scaler.fit_transform(x_train.astype(float))
    x_test = mean_variance_scaler.transform(x_test.astype(float))

    pca = PCA(n_components=0.90)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')

    plt.show()
    print("Variance Explained: ", sum(pca.explained_variance_ratio_))

    biplot(x_train_pca[:, 0:2], np.transpose(pca.components_[0:2, :]), list(features.columns))

    return x_train_pca, x_test_pca, y_train, y_test, ro5_test, ids_test
