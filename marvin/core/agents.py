"""
  MIT License
  Copyright (c) 2020 Marcus Vinicius D. B. Braga - mvbraga@gmail.com

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
"""

import threading
import warnings

from marvin.core.threads import ThreadTrainModel, KFoldParams


class Agent:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.models = []
        self.results = []
        self.__threads = []

    def init_models(self, model=None):
        """
        This method initializes the Machine Learning models that will be used in training and testing..

        :param
            model = Informs a Model object if you want to use only a specific model.
            Check marvin.core.models.py.

        :return
            self.
        """
        self.models = []
        if model is None:
            # Basic models of Machine Learning that will be initialized.
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.naive_bayes import GaussianNB
            from sklearn.svm import SVC
            from sklearn.neural_network import MLPClassifier

            self.models.append(("LogisticRegression", LogisticRegression(solver="lbfgs")))
            self.models.append(("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()))
            self.models.append(("KNeighborsClassifier", KNeighborsClassifier()))
            self.models.append(("DecisionTreeClassifier", DecisionTreeClassifier()))
            self.models.append(("GaussianNB", GaussianNB()))
            self.models.append(("SVM (SVC)", SVC(gamma="auto")))
            self.models.append(("MLPClassifier", MLPClassifier(hidden_layer_sizes=(200, 200, 200),
                                                               max_iter=10,
                                                               alpha=0.0001,
                                                               solver="sgd",
                                                               random_state=21,
                                                               tol=0.000000001)))
        else:
            # Initializes only the specific model stored in the model attribute.
            self.models.append((model.description, model.obj))

        return self

    def fit(self, scoring):
        """
        This method performs the training of the initialized models and prints its results in separate threads.

        :return: self
        """
        # Creates a semaphore to organize the inclusion of results in the list during multi threaded operation.
        semaphore = threading.Semaphore()

        warnings.filterwarnings("ignore")
        try:
            # Retrieves the name and model for training.
            for thread_number in range(self.models_count()):
                name, model = self.models[thread_number]
                self.__threads.insert(
                    thread_number,
                    ThreadTrainModel(
                        semaphore=semaphore,
                        agent=self,
                        model_name=name,
                        model=model,
                        scoring=scoring,
                        k_fold_params=KFoldParams(
                            n_splits=10,
                            random_state=3
                        )
                    )
                )
                self.__threads[thread_number].start()

            total = self.models_count()
            # Waits for all threads to finish.
            for thread_number in range(total):
                self.__threads[thread_number].join()

        finally:
            warnings.filterwarnings("default")

        return self

    def get_best_model(self):
        mean_values = [result.mean() for result in self.results]
        max_result = max(mean_values)
        position_index = mean_values.index(max_result)
        name_result = self.results[position_index].name
        best_result = mean_values[position_index]

        desc = ""
        best_model = None
        for desc, best_model in self.models:
            if desc == name_result:
                break

        return best_model, desc, best_result

    def models_count(self):
        return len(self.models)

    def results_count(self):
        return len(self.results)
