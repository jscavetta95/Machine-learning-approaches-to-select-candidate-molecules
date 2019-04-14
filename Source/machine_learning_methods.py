import numpy as np
import scipy.stats


def __report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean F score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def logistic_regression(x, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV

    clf = LogisticRegression(solver="lbfgs", random_state=23)

    param_dist = {"C": scipy.stats.expon(scale=100)}
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, scoring="f1",
                                       return_train_score=True, n_iter=100,
                                       cv=10, n_jobs=-1)
    random_search.fit(x, y)
    __report(random_search.cv_results_)

    __plot_logistic_regression(random_search.cv_results_)
    return random_search.best_estimator_


def __plot_logistic_regression(results):
    import matplotlib.pyplot as plt

    order = np.argsort(results['param_C'].data)
    x = np.array(results['param_C'].data)[order]
    mean_train_score = np.array(results['mean_train_score'].data)[order]
    mean_test_score = np.array(results['mean_test_score'].data)[order]
    std_train_score = np.array(results['std_train_score'].data)[order]
    std_test_score = np.array(results['std_test_score'].data)[order]

    plt.plot(x, mean_train_score, color='seagreen', label='Training score')
    plt.fill_between(x.astype(float), mean_train_score, mean_train_score + std_train_score,
                     color='seagreen', alpha=0.3)
    plt.fill_between(x.astype(float), mean_train_score, mean_train_score - std_train_score,
                     color='seagreen', alpha=0.3)

    plt.plot(x, mean_test_score, color='steelblue', label='Testing score')
    plt.fill_between(x.astype(float), mean_test_score, mean_test_score + std_test_score,
                     color='steelblue', alpha=0.3)
    plt.fill_between(x.astype(float), mean_test_score, mean_test_score - std_test_score,
                     color='steelblue', alpha=0.3)

    plt.ylim(0, 1)
    plt.xlabel('Inverse of Regularization Strength')
    plt.ylabel('F Score')
    plt.legend(loc='best')
    plt.show()


def support_vector_machine(x, y):
    from sklearn import svm
    from sklearn.model_selection import RandomizedSearchCV

    clf = svm.SVC(cache_size=2000, random_state=23, gamma='auto')
    param_dist = {"C": scipy.stats.expon(scale=100),
                  "kernel": ["linear", "rbf"]}
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, scoring="f1",
                                       return_train_score=True, n_iter=50, cv=10, n_jobs=-1)
    random_search.fit(x, y)
    __report(random_search.cv_results_)
    __plot_support_vector_machine(random_search.cv_results_)
    return random_search.best_estimator_


def __plot_support_vector_machine(results):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 6))
    ###
    indices = [i for i, item in enumerate(results['param_kernel'].data)
               if (item == "linear")]
    order = np.argsort(results['param_C'].data[indices])
    x = np.array(results['param_C'].data[indices])[order]
    mean_train_score = np.array(results['mean_train_score'][indices])[order]
    mean_test_score = np.array(results['mean_test_score'][indices])[order]
    std_train_score = np.array(results['std_train_score'][indices])[order]
    std_test_score = np.array(results['std_test_score'][indices])[order]

    axes[0].plot(x, mean_train_score, color='seagreen', label='Training score')
    axes[0].fill_between(x.astype(float), mean_train_score,
                         mean_train_score + std_train_score,
                         color='seagreen', alpha=0.3)
    axes[0].fill_between(x.astype(float), mean_train_score,
                         mean_train_score - std_train_score,
                         color='seagreen', alpha=0.3)

    axes[0].plot(x, mean_test_score, color='steelblue', label='Testing score')
    axes[0].fill_between(x.astype(float), mean_test_score,
                         mean_test_score + std_test_score,
                         color='steelblue', alpha=0.3)
    axes[0].fill_between(x.astype(float), mean_test_score,
                         mean_test_score - std_test_score,
                         color='steelblue', alpha=0.3)

    ###
    indices = [i for i, item in enumerate(results['param_kernel'].data)
               if (item == "rbf")]
    order = np.argsort(results['param_C'].data[indices])
    x = np.array(results['param_C'].data[indices])[order]
    mean_train_score = np.array(results['mean_train_score'][indices])[order]
    mean_test_score = np.array(results['mean_test_score'][indices])[order]
    std_train_score = np.array(results['std_train_score'][indices])[order]
    std_test_score = np.array(results['std_test_score'][indices])[order]

    axes[1].plot(x, mean_train_score, color='seagreen', label='Training score')
    axes[1].fill_between(x.astype(float), mean_train_score,
                         mean_train_score + std_train_score,
                         color='seagreen', alpha=0.3)
    axes[1].fill_between(x.astype(float), mean_train_score,
                         mean_train_score - std_train_score,
                         color='seagreen', alpha=0.3)

    axes[1].plot(x, mean_test_score, color='steelblue', label='Testing score')
    axes[1].fill_between(x.astype(float), mean_test_score,
                         mean_test_score + std_test_score,
                         color='steelblue', alpha=0.3)
    axes[1].fill_between(x.astype(float), mean_test_score,
                         mean_test_score - std_test_score,
                         color='steelblue', alpha=0.3)

    plt.ylim(0, 1)
    axes[1].set_xlabel('Inverse of Regularization Strength')
    axes[1].set_ylabel("F Score")
    axes[0].set_title("Kernel = linear")
    axes[1].set_title("Kernel = RBF")
    axes[0].legend(loc='best')
    figure.show()


def random_forest(x, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    clf = RandomForestClassifier(random_state=23)
    param_dist = {'n_estimators': scipy.stats.randint(1, 1000)}
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, scoring="f1",
                                       return_train_score=True, n_iter=50, cv=10, n_jobs=-1)
    random_search.fit(x, y)
    __report(random_search.cv_results_)
    __plot_random_forest(random_search.cv_results_)
    return random_search.best_estimator_


def __plot_random_forest(results):
    import matplotlib.pyplot as plt

    order = np.argsort(results['param_n_estimators'].data)
    x = np.array(results['param_n_estimators'].data)[order]
    mean_train_score = np.array(results['mean_train_score'].data)[order]
    mean_test_score = np.array(results['mean_test_score'].data)[order]
    std_train_score = np.array(results['std_train_score'].data)[order]
    std_test_score = np.array(results['std_test_score'].data)[order]

    plt.plot(x, mean_train_score, color='seagreen', label='Training score')
    plt.fill_between(x.astype(float), mean_train_score, mean_train_score + std_train_score,
                     color='seagreen', alpha=0.3)
    plt.fill_between(x.astype(float), mean_train_score, mean_train_score - std_train_score,
                     color='seagreen', alpha=0.3)

    plt.plot(x, mean_test_score, color='steelblue', label='Testing score')
    plt.fill_between(x.astype(float), mean_test_score, mean_test_score + std_test_score,
                     color='steelblue', alpha=0.3)
    plt.fill_between(x.astype(float), mean_test_score, mean_test_score - std_test_score,
                     color='steelblue', alpha=0.3)

    plt.ylim(0, 1)
    plt.xlabel('Number of Estimators')
    plt.ylabel('F Score')
    plt.legend(loc='best')
    plt.show()


def multilayer_perceptron(x, y):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV

    clf = MLPClassifier(solver='adam', activation="logistic", random_state=23)
    param_dist = {'alpha': scipy.stats.expon(scale=100),
                  'hidden_layer_sizes': [(50, 50, 50), (100, 100, 100), 100]}
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, scoring="f1",
                                       return_train_score=True, n_iter=50, cv=10, n_jobs=-1)
    random_search.fit(x, y)
    __report(random_search.cv_results_)
    __plot_multilayer_perceptron(random_search.cv_results_)
    return random_search.best_estimator_


def __plot_multilayer_perceptron(results):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6, 7))

    ###
    indices = [i for i, item in enumerate(results['param_hidden_layer_sizes'].data)
               if (type(item) == tuple) and (item[0] == 50)]
    order = np.argsort(results['param_alpha'].data[indices])
    x = np.array(results['param_alpha'].data[indices])[order]
    mean_train_score = np.array(results['mean_train_score'][indices])[order]
    mean_test_score = np.array(results['mean_test_score'][indices])[order]
    std_train_score = np.array(results['std_train_score'][indices])[order]
    std_test_score = np.array(results['std_test_score'][indices])[order]

    axes[0].plot(x, mean_train_score, color='seagreen', label='Training score')
    axes[0].fill_between(x.astype(float), mean_train_score,
                         mean_train_score + std_train_score,
                         color='seagreen', alpha=0.3)
    axes[0].fill_between(x.astype(float), mean_train_score,
                         mean_train_score - std_train_score,
                         color='seagreen', alpha=0.3)

    axes[0].plot(x, mean_test_score, color='steelblue', label='Testing score')
    axes[0].fill_between(x.astype(float), mean_test_score,
                         mean_test_score + std_test_score,
                         color='steelblue', alpha=0.3)
    axes[0].fill_between(x.astype(float), mean_test_score,
                         mean_test_score - std_test_score,
                         color='steelblue', alpha=0.3)

    ###
    indices = [i for i, item in enumerate(results['param_hidden_layer_sizes'].data)
               if (type(item) == tuple) and (item[0] == 100)]
    order = np.argsort(results['param_alpha'].data[indices])
    x = np.array(results['param_alpha'].data[indices])[order]
    mean_train_score = np.array(results['mean_train_score'][indices])[order]
    mean_test_score = np.array(results['mean_test_score'][indices])[order]
    std_train_score = np.array(results['std_train_score'][indices])[order]
    std_test_score = np.array(results['std_test_score'][indices])[order]

    axes[1].plot(x, mean_train_score, color='seagreen', label='Training score')
    axes[1].fill_between(x.astype(float), mean_train_score,
                         mean_train_score + std_train_score,
                         color='seagreen', alpha=0.3)
    axes[1].fill_between(x.astype(float), mean_train_score,
                         mean_train_score - std_train_score,
                         color='seagreen', alpha=0.3)

    axes[1].plot(x, mean_test_score, color='steelblue', label='Testing score')
    axes[1].fill_between(x.astype(float), mean_test_score,
                         mean_test_score + std_test_score,
                         color='steelblue', alpha=0.3)
    axes[1].fill_between(x.astype(float), mean_test_score,
                         mean_test_score - std_test_score,
                         color='steelblue', alpha=0.3)

    ###
    indices = [i for i, item in enumerate(results['param_hidden_layer_sizes'].data)
               if (type(item) != tuple)]
    order = np.argsort(results['param_alpha'].data[indices])
    x = np.array(results['param_alpha'].data[indices])[order]
    mean_train_score = np.array(results['mean_train_score'][indices])[order]
    mean_test_score = np.array(results['mean_test_score'][indices])[order]
    std_train_score = np.array(results['std_train_score'][indices])[order]
    std_test_score = np.array(results['std_test_score'][indices])[order]

    axes[2].plot(x, mean_train_score, color='seagreen', label='Training score')
    axes[2].fill_between(x.astype(float), mean_train_score,
                         mean_train_score + std_train_score,
                         color='seagreen', alpha=0.3)
    axes[2].fill_between(x.astype(float), mean_train_score,
                         mean_train_score - std_train_score,
                         color='seagreen', alpha=0.3)

    axes[2].plot(x, mean_test_score, color='steelblue', label='Testing score')
    axes[2].fill_between(x.astype(float), mean_test_score,
                         mean_test_score + std_test_score,
                         color='steelblue', alpha=0.3)
    axes[2].fill_between(x.astype(float), mean_test_score,
                         mean_test_score - std_test_score,
                         color='steelblue', alpha=0.3)

    plt.ylim(0, 1)
    axes[2].set_xlabel('Alpha')
    axes[1].set_ylabel("F Score")
    axes[0].set_title("Hidden Layers = (50,50,50)")
    axes[1].set_title("Hidden Layers = (100,100,100)")
    axes[2].set_title("Hidden Layers = (100)")
    axes[0].legend(loc='best')
    figure.show()
