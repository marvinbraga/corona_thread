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


class Result(object):
    """
    The Result class is used to store the values obtained in model training.
    It stores the model name and its results.
    """

    def __init__(self, name, results):
        self.name = name
        self.results = results

    def __str__(self):
        return "%s: weighted average(%f), min(%f), max(%f), std(%f)" % (
            self.name,
            self.mean(),
            self.min(),
            self.max(),
            self.std()
        )

    def mean(self):
        """
        This method returns the weighted average of the results retrieved.
        """
        return self.results.mean()

    def max(self):
        """
        This method returns the maximum value found among the retrieved results.
        """
        return max(self.results)

    def min(self):
        """
        This method returns the minimum value found among the retrieved results.
        """
        return min(self.results)

    def std(self):
        """
        This method returns the value of the standard deviation found among the retrieved results.
        """
        return self.results.std()
