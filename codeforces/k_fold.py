def main(n, m, k, samples):
   label_to_index = {}

   for idx, l in enumerate(samples):
      if l in label_to_index:
          label_to_index[l].append(idx)
      else:
          label_to_index[l] = [idx]

   folds = [[] for _ in range(k)]
   
   cur_fold = 0

   for l in label_to_index.keys():
      indices = label_to_index[l]
      
      for ind in indices:
         folds[cur_fold].append(ind + 1)
         cur_fold += 1
         if cur_fold == k:
            cur_fold = 0

   for fold in folds:
      print(len(fold), end=' ')
      print(*fold, sep=' ')


if __name__ == '__main__':
   n, m, k = map(int, input().split())
   samples = list(map(int, input().split()))
   
   main(n, m, k, samples)

