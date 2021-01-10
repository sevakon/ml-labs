import numpy as np


class Vectorizer:

    unique_words = dict()
    cur_idx = 0

    def fit(self, X):
        for x in X:
            for word in x:
                if word not in self.unique_words:
                    self.unique_words[word] = self.cur_idx
                    self.cur_idx += 1

        return self

    def transform(self, X):
        output = np.zeros((len(X), len(self.unique_words)), dtype=int)
        for idx, x in enumerate(X):
            for word in x:
                if word in self.unique_words:
                    output[idx, self.unique_words[word]] = 1

        return output


class NaiveBayesClassifier:

    def __init__(self, num_classes, alpha, penalties):
        self.alpha = alpha
        self.penalties = penalties
        self._is_fitted = False
        self.num_classes = num_classes
        self.prior_ = None
        self.word_likelihood_ = None
        self._Q = 2

    def fit(self, X, y):
        num_samples = X.shape[0]
        X_by_class = [
            [x for x, t in zip(X, y) if t == c] for c in range(self.num_classes)]

        self.prior_ = np.array([len(i) / num_samples for i in X_by_class])

        num_samples_per_class = np.array([len(x) for x in X_by_class])
        word_counts_per_class = np.array(
            [np.array(i).sum(axis=0) for i in X_by_class])

        # Bernoulli with Laplace Smoothing
        self.word_likelihood_ = (word_counts_per_class + self.alpha) / (
                num_samples_per_class.reshape(-1, 1) + self.alpha * self._Q)

        self._is_fitted = True

    def predict_one(self, x):
        probabilities = np.zeros(self.num_classes)

        for class_idx in range(self.num_classes):
            if class_idx >= len(self.prior_) or self.prior_[class_idx] == 0:
                continue

            temp = np.zeros(self.num_classes)

            for other_class_idx in range(self.num_classes):
                if other_class_idx >= len(self.prior_) or class_idx == other_class_idx:
                    continue

                t = self.penalties[other_class_idx] / self.penalties[class_idx] \
                    * (self.prior_[other_class_idx] / self.prior_[class_idx])

                for feature_idx in range(len(x)):
                    if x[feature_idx] == 0:
                        prob = 1 - self.word_likelihood_[class_idx, feature_idx]
                        other_prob = 1 - self.word_likelihood_[other_class_idx, feature_idx]
                    else:
                        prob = self.word_likelihood_[class_idx, feature_idx]
                        other_prob = self.word_likelihood_[other_class_idx, feature_idx]

                    t *= other_prob / prob

                temp[other_class_idx] = t

                probabilities[class_idx] = 1 / (1 + temp.sum())

        return probabilities


def main(X_train, y_train, X_test, used_labels, num_classes, lambda_c, alpha):
    v = Vectorizer().fit(X_train)

    X_train_vectorized = v.transform(X_train)
    X_test_vectorized = v.transform(X_test)

    for j in range(num_classes):
        if j not in used_labels:
            num_samples, num_words = X_train_vectorized.shape
            new_X = np.zeros(num_samples + 1, num_words)
            new_X[:num_samples, num_words] = X_train_vectorized
            new_X[num_samples, num_words] = np.zeros(num_words)
            X_train_vectorized = new_X

    classifier = NaiveBayesClassifier(num_classes, alpha, lambda_c)

    classifier.fit(X_train_vectorized, np.array(y_train))

    for X_sample in X_test_vectorized:
        print(*classifier.predict_one(X_sample))


if __name__ == '__main__':
    k = int(input())
    lambda_c = list(map(int, input().split()))
    alpha = int(input())
    N = int(input())

    X_train = list()
    y_train = list()

    used_labels = set()

    for _ in range(N):
        line = input().split()
        label = int(line[0])
        words = line[2:]

        X_train.append(words)
        y_train.append(label - 1)

        used_labels.add(label - 1)

    M = int(input())
    X_test = list()

    for _ in range(M):
        words = input().split()[1:]

        X_test.append(words)

    main(X_train, y_train, X_test, used_labels, k, lambda_c, alpha)
