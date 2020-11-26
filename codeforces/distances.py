def calc_distances(x, targets):
   """ within and between class distances """
   within_class = 0
   between_class = 0

   for i in range(len(x)):
      for j in range(len(x)):
         abs_diff = abs(x[i] - x[j])
         if targets[i] == targets[j]:
            within_class += abs_diff
         else:
            between_class += abs_diff           
 
   return within_class, between_class


if __name__ == '__main__':
   k = int(input())
   n = int(input())
   
   x = list()
   y = list()
   
   for _ in range(n):
      x_i, y_i = list(map(int, input().split()))
      x.append(x_i)
      y.append(y_i)

   within_class_d, between_class_d = calc_distances(x, y)
   
   print(within_class_d)
   print(between_class_d)
