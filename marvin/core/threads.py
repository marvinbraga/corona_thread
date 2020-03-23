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

import time
from threading import Thread
from sklearn import model_selection
from marvin.core.results import Result


class KFoldParams:

    n_splits = 0
    random_state = 0

    def __init__(self, n_splits=10, random_state=7):
        self.n_splits = n_splits
        self.random_state = random_state


class ThreadTrainModel(Thread):

    def __init__(
            self,
            semaphore,
            agent,
            model_name,
            model,
            k_fold_params=KFoldParams(),
            group=None,
            target=None,
            name=None,
            scoring="accuracy"
    ):
        super().__init__(group=group, target=target, name=name)
        self.__model_name = model_name
        self.__model = model
        self.__semaphore = semaphore
        self.__agent = agent
        self.__scoring = scoring
        self.__k_fold_params = k_fold_params

    def __execute(self):
        """
        This method performs the training for each model.
        :return: self
        """
        # Separates data into validation folds.
        k_fold = model_selection.KFold(
            n_splits=self.__k_fold_params.n_splits,
            random_state=self.__k_fold_params.random_state
        )
        # Performs cross validation on the model.
        cv_results = model_selection.cross_val_score(
            estimator=self.__model,
            X=self.__agent.x_train,
            y=self.__agent.y_train,
            scoring=self.__scoring,
            cv=k_fold
        )
        # Retrieves the result.
        result = Result(self.__model_name, cv_results)

        # Saves the result obtained from the model.
        self.__semaphore.acquire()
        try:
            self.__agent.results.append(result)
        finally:
            self.__semaphore.release()

        # Displays messages synchronized with other secondary threads.
        self.__print(result)
        time.sleep(0.05)

        return self

    def __print(self, msg):
        import sys
        sys.stdout.write(str(msg))
        sys.stdout.write("\n")
        sys.stdout.flush()

        return self

    def run(self):
        self.__execute()
