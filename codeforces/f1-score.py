# MACRO - calculate precision and recall for every class,
# then take their weigted sum and calculate ONE f1 score

# MICRO - calculate f1 scores for each class and take their weighted sum

def f1_score(precision, recall):
   f = 0 if precision + recall == 0 \
      else 2 * precision * recall / (precision + recall)
   return f


def main(k, confusion_matrix):
   num_samples = [sum(confusion_matrix[idx]) for idx in range(k)]
   weights = [s / sum(num_samples) for s in num_samples]
   
   precisions = []
   recalls = []

   for cls_idx in range(k):
      tp = confusion_matrix[cls_idx][cls_idx]
      fp = sum([confusion_matrix[idx][cls_idx] for idx in range(k)]) - tp
      fn = sum([confusion_matrix[cls_idx][idx] for idx in range(k)]) - tp

      precision = 0 if tp + fp == 0 else tp / (tp + fp)
      recall = 0 if tp + fn == 0 else tp / (tp + fn)

      precisions.append(precision)
      recalls.append(recall)

   weighted_precision = sum([
      weights[idx] * precisions[idx] for idx in range(k)
   ]) 
   weighted_recall = sum([
      weights[idx] * recalls[idx] for idx in range(k)
   ]) 
   macro_f1 = f1_score(weighted_precision, weighted_recall) 

   

   f1_scores = [f1_score(precisions[idx], recalls[idx]) for idx in range(k)]
   micro_f1 = sum([
       weights[idx] * f1_scores[idx] for idx in range(k)
   ])
   
   print(macro_f1)
   print(micro_f1)
   

if __name__ == '__main__':
   k = int(input())
   
   confusion_matrix = []

   for _ in range(k):
      values = list(map(int, input().split()))
      confusion_matrix.append(values)
   
   main(k, confusion_matrix)
