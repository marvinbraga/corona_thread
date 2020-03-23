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

from sklearn.datasets import load_iris

from marvin.core.agents import Agent


class SampleIris:

    def __init__(self):
        self.iris = load_iris()

    def run(self):
        _X, _y = self.iris.data, self.iris.target

        agent = Agent(_X, _y).init_models().fit('accuracy')
        best_model, desc, best_result = agent.get_best_model()

        print(f'BEST RESULT: model={desc}, result={best_result}')


if __name__ == '__main__':
    SampleIris().run()
