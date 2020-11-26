import math


def mean(vec):
   return sum(vec) / len(vec)


def variance(vec):
   return mean([v ** 2 for v in vec]) - mean(vec) ** 2


def covariance(x, y):
   return mean([x_i * y_i for x_i, y_i in zip(x, y)]) - mean(x) * mean(y)


if __name__ == '__main__':
   m = int(input())
   x = list()
   y = list()

   for _ in range(m):
      x_i, y_i = list(map(int, input().split()))
      x.append(x_i)
      y.append(y_i)
   
   try:
      p = covariance(x, y) / (
         math.sqrt(variance(x)) * math.sqrt(variance(y)))
      print(p)
   except:
      print(0)


