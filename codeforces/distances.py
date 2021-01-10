def calc_distances(x, targets):
   """ within and between class distances """

   def c_s(l):
      l.sort()
      s, r = 0, 0
      n  = len(l)

      for i in range(n - 1):
         s += l[i]
         r += 2 * (n - i - 1) * l[i]
      s += l[-1]

      return (n - 1) * s - r

   within_class = 0

   for y, inclass_x in targets.items():
      within_class += c_s(inclass_x)

   within_class = 2 * within_class
   between_class = 2 * c_s(x) - within_class

   return within_class, between_class


if __name__ == '__main__':
   k = int(input())
   n = int(input())

   x = list()
   targets = dict()

   for _ in range(n):
      x_i, y_i = list(map(int, input().split()))
      x.append(x_i)

      if y_i in targets:
         targets[y_i].append(x_i)
      else:
         targets[y_i] = [x_i]

   within_class_d, between_class_d = calc_distances(x, targets)

   print(within_class_d)
   print(between_class_d)

