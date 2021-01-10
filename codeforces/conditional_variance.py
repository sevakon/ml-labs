def expectation(x):
    if len(x) == 0:
        return 0

    s = 0
    for sample in x:
        s += sample

    return s / len(x)


def variance(x):
    return expectation([s ** 2 for s in x]) - expectation(x) ** 2


def conditional_variance(y_conditional):
    class_to_variance = dict()

    for x, values in y_conditional.items():
        class_to_variance[x] = (variance(values), len(values))

    return class_to_variance


def conditional_variance_expectation(class_to_variance, n):
    exp = 0

    for x, t in class_to_variance.items():
        exp += t[1] / n * t[0]

    return exp


if __name__ == '__main__':
    k = int(input())
    n = int(input())

    y_conditional = {i+1:[] for i in range(k)}

    for _ in range(n):
        x, y = list(map(int, input().split()))
        y_conditional[x].append(y)

    d = conditional_variance(y_conditional)
    e = conditional_variance_expectation(d, n)

    print(e)

