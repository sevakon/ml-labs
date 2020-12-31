import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


class Vectorizer:

    word_to_idx = dict()
    last_index = 1

    def fit(self, words):
        for word in words:
            if word not in self.word_to_idx and word != '':
                self.word_to_idx[word] = self.last_index
                self.last_index += 1

    def transform(self, words):
        output = list()
        for w in words:
            if w in self.word_to_idx:
                output.append(self.word_to_idx[w])
        return output


class MyCountVectorizer:

    unique_words = set()

    def fit(self, X):
        for x in X:
            for word in x:
                self.unique_words.add(word)

        return self

    def transform(self, X):
        output = np.zeros((len(X), len(self.unique_words)))
        unique_words = list(self.unique_words)
        for idx, x in enumerate(X):
            for word in x:
                if word in self.unique_words:
                    output[idx, unique_words.index(word)] += 1

        return output


class NaiveBayesClassifier:

    def __init__(self, alpha, penalties):
        self.alpha = alpha
        self.penalties = penalties
        self._is_fitted = False
        self.num_classes = len(lambda_c)
        self.prior_ = None
        self.word_likelihood_ = None

    def fit(self, X, y):
        num_samples = X.shape[0]
        X_by_class = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]

        self.prior_ = np.array([np.log(len(i) / num_samples) for i in X_by_class])

        word_counts = np.array([np.array(i).sum(axis=0) for i in X_by_class]) + self.alpha
        self.word_likelihood_ = word_counts / word_counts.sum(axis=1).reshape(-1, 1)

        self._is_fitted = True

    def predict_proba(self, X):
        probabilities = np.zeros(shape=(X.shape[0], self.prior_.shape[0]))

        # nested loop goes brughhh
        for i, x in enumerate(X):

            lk_message = np.zeros(self.prior_.shape[0])
            for word in x:
                if word != 0.:
                    lk_message += np.log(self.word_likelihood_[:, x])

            probabilities[i] = lk_message + np.log(self.prior_) + np.log(self.penalties)

        def softmax(vector):
            exps = np.exp(vector)
            print(exps / np.sum(exps))

        for p in probabilities:
            softmax(p)

        return probabilities


def main(X_train, y_train, X_test, lambda_c, alpha, wctrain, wctest):
    v = MyCountVectorizer().fit(X_train)
    print(v.unique_words)
    print(v.transform(X_train))
    print(v.transform(X_test))

    X_train = v.transform(X_train)
    X_test = v.transform(X_test)

    print(v.unique_words)
    print(X_train)

    print(X_test)

    # for i in range(len(X_train)):
    #     x = np.zeros(padding)
    #     transformed = vectorizer.transform(X_train[i])
    #     x[:len(transformed)] = transformed
    #     X_train[i] = x
    #
    # for i in range(len(X_test)):
    #     x = np.zeros(padding)
    #     transformed = vectorizer.transform(X_test[i])
    #     x[:len(transformed)] = transformed
    #     X_test[i] = x
    #

    new_vec = Vectorizer()
    unique_ys = list(set(y_train))
    new_vec.fit(unique_ys)

    transformed_y_train = new_vec.transform(y_train)
    y_train = list(map(lambda x: x - 1, transformed_y_train))

    classifier = NaiveBayesClassifier(alpha, lambda_c)

    classifier.fit(X_train, np.array(y_train))

    print(classifier.predict_proba(X_test))


if __name__ == '__main__':
    k = int(input())
    lambda_c = list(map(int, input().split()))
    alpha = int(input())
    N = int(input())

    largest_word_count_in_train = 0
    largest_word_count_in_test = 0

    X_train = list()
    y_train = list()

    for _ in range(N):
        line = input().split()
        label = int(line[0])
        words = line[2:]

        if len(words) > largest_word_count_in_train:
            largest_word_count_in_train = len(words)
        
        X_train.append(words)
        y_train.append(label)        

    M = int(input())

    X_test = list()

    for _ in range(M):
        words = input().split()[1:]

        if len(words) > largest_word_count_in_test:
            largest_word_count_in_test = len(words)

        X_test.append(words)

    main(X_train, y_train, X_test, lambda_c, alpha,
         largest_word_count_in_train, largest_word_count_in_test)
