def get_ranks(vector):
   vector_with_indices = sorted(
      [(v, idx) for idx, v in enumerate(vector)], key=lambda x: x[0])

   prev_v = vector_with_indices[0][0]
   cur_rank = 0
   ranks = list()
   ranks.append(cur_rank)

   for v, _ in vector_with_indices[1:]:
      if prev_v != v:
         cur_rank += 1
      prev_v = v
      ranks.append(cur_rank)

   ind_to_rank = [0 for _ in range(len(vector))]
   for (v, i), r in zip(vector_with_indices, ranks):
      ind_to_rank[i] = r

   return ind_to_rank


def spearman(x, y):
   n = len(x)
   
   x_ranks = get_ranks(x)
   y_ranks = get_ranks(y)

   exp = sum([(r - s) ** 2 for r, s in zip(x_ranks, y_ranks)])
   p = 1 - 6 * exp / (n * (n - 1) * (n + 1))

   return p
   

if __name__ == '__main__':
   m = int(input())
   x, y = list(), list()   

   for _ in range(m):
      x_i, y_i = list(map(int, input().split()))
      x.append(x_i)
      y.append(y_i)

   result = spearman(x, y)

   print(result)
