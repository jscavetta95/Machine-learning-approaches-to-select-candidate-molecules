import numpy as np


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
    from sklearn.model_selection import GridSearchCV

    clf = LogisticRegression(solver="lbfgs", random_state=23)
    param_grid = {"C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring="f1", return_train_score=True,
                               cv=10, n_jobs=-1)
    grid_search.fit(x, y)
    __report(grid_search.cv_results_)
    __plot_logistic_regression(grid_search.cv_results_)
    return grid_search.best_estimator_


def __plot_logistic_regression(results):
    import matplotlib.pyplot as plt

    plt.plot(results['param_C'].data,
             results['mean_train_score'].data,
             color='seagreen', label='Training score')
    plt.errorbar(results['param_C'].data,
                 results['mean_train_score'].data,
                 yerr=results['std_train_score'].data,
                 color='seagreen')
    plt.plot(results['param_C'].data,
             results['mean_test_score'].data,
             color='steelblue', label='Testing score')
    plt.errorbar(results['param_C'].data,
                 results['mean_test_score'].data,
                 yerr=results['std_test_score'].data,
                 color='steelblue')
    plt.ylim(0, 1)
    plt.xlabel('Regularization Multiplier')
    plt.ylabel('F Score')
    plt.legend(loc='best')
    plt.show()


def support_vector_machine(x, y):
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV

    clf = svm.SVC(cache_size=2000, random_state=23)
    param_grid = {"C": [1, 10, 20],
                  "kernel": ["linear", "rbf"],
                  "gamma": [0.2, 0.4, 0.6]}
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring="f1", return_train_score=True, cv=10, n_jobs=-1)
    grid_search.fit(x, y)
    __report(grid_search.cv_results_)
    __plot_support_vector_machine(grid_search.cv_results_)
    return grid_search.best_estimator_


def __plot_support_vector_machine(results):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(8, 6))
    for i in range(3):
        for j in range(2):
            indices = [0 + (i * 2) + j, 6 + (i * 2) + j, 12 + (i * 2) + j]

            axes[i, j].plot(results['param_C'].data[indices],
                            results['mean_train_score'][indices],
                            color='seagreen', label='Training score')

            axes[i, j].errorbar(results['param_C'].data[indices],
                                results['mean_train_score'][indices],
                                yerr=results['std_train_score'][indices],
                                color='seagreen')

            axes[i, j].plot(results['param_C'].data[indices],
                            results['mean_test_score'][indices],
                            color='steelblue', label='Test score')

            axes[i, j].errorbar(results['param_C'].data[indices],
                                results['mean_test_score'][indices],
                                yerr=results['std_test_score'][indices],
                                color='steelblue')
    plt.ylim(0, 1)
    axes[0, 0].legend(loc='best')
    axes[2, 0].set_xlabel('Regularization Multiplier')
    axes[2, 1].set_xlabel('Regularization Multiplier')
    axes[1, 0].set_ylabel("F Score")
    axes[0, 0].set_title("Gamma = 0.2 | Kernel = linear")
    axes[0, 1].set_title("Gamma = 0.2 | Kernel = RBF")
    axes[1, 0].set_title("Gamma = 0.4 | Kernel = linear")
    axes[1, 1].set_title("Gamma = 0.4 | Kernel = RBF")
    axes[2, 0].set_title("Gamma = 0.6 | Kernel = linear")
    axes[2, 1].set_title("Gamma = 0.6 | Kernel = RBF")
    figure.show()


def random_forest(x, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    clf = RandomForestClassifier(random_state=23)
    param_grid = {'n_estimators': [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]}
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring="f1", return_train_score=True, cv=10, n_jobs=-1)
    grid_search.fit(x, y)
    __report(grid_search.cv_results_)
    __plot_random_forest(grid_search.cv_results_)
    return grid_search.best_estimator_


def __plot_random_forest(results):
    import matplotlib.pyplot as plt

    plt.plot(results['param_n_estimators'].data,
             results['mean_train_score'].data,
             color='seagreen', label='Training score')
    plt.errorbar(results['param_n_estimators'].data,
                 results['mean_train_score'].data,
                 yerr=results['std_train_score'],
                 color='seagreen')
    plt.plot(results['param_n_estimators'].data,
             results['mean_test_score'].data,
             color='steelblue', label='Testing score')
    plt.errorbar(results['param_n_estimators'].data,
                 results['mean_test_score'].data,
                 yerr=results['std_test_score'],
                 color='steelblue')
    plt.ylim(0, 1)
    plt.xlabel('Number of Estimators')
    plt.ylabel('F Score')
    plt.legend(loc='best')
    plt.show()


def multilayer_perceptron(x, y):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV

    clf = MLPClassifier(solver='adam', activation="logistic", random_state=23)
    param_grid = {'alpha': [0.2, 0.4, 0.6, 0.8, 1],
                  'hidden_layer_sizes': [(50, 50, 50), (100, 100, 100), (100)]}
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring="f1", return_train_score=True, cv=10, n_jobs=-1)
    grid_search.fit(x, y)
    __report(grid_search.cv_results_)
    __plot_multilayer_perceptron(grid_search.cv_results_)
    return grid_search.best_estimator_


def __plot_multilayer_perceptron(results):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    for i in range(3):
        indices = [i, 3 + i, 6 + i, 9 + i, 12 + i]
        axes[i].plot(results['param_alpha'].data[indices],
                     results['mean_train_score'][indices],
                     color='seagreen', label='Training score')
        axes[i].errorbar(results['param_alpha'].data[indices],
                         results['mean_train_score'][indices],
                         yerr=results['std_train_score'][indices],
                         color='seagreen')
        axes[i].plot(results['param_alpha'].data[indices],
                     results['mean_test_score'][indices],
                     color='steelblue', label='Test score')
        axes[i].errorbar(results['param_alpha'].data[indices],
                         results['mean_test_score'][indices],
                         yerr=results['std_test_score'][indices],
                         color='steelblue')
    plt.ylim(0, 1)
    axes[2].set_xlabel('Regularization Multiplier')
    axes[1].set_ylabel("F Score")
    axes[0].set_title("Hidden Layers = (50,50,50)")
    axes[1].set_title("Hidden Layers = (100,100,100)")
    axes[2].set_title("Hidden Layers = (100)")
    axes[0].legend(loc='best')
    figure.show()
