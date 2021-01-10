from math import log


def cond_entropy(kx, ky, xs, ys):
    assert len(xs) == len(ys)

    n = len(xs)
    prob_x = dict()
    prob_xy = dict()

    for x, y in zip(xs, ys):
        if x not in prob_x:
            prob_x[x] = 1 / n
        else:
            prob_x[x] += 1 / n

        if (x, y) not in prob_xy:
            prob_xy[(x, y)] = 1 / n
        else:
            prob_xy[(x, y)] += 1 / n

    result = .0

    for (x, y) in prob_xy.keys():
        p = -prob_xy[(x, y)] * (log(prob_xy[(x, y)]) - log(prob_x[x]))
        result += p

    return result


if __name__ == '__main__':
    k_x, k_y = map(int, input().split())
    n = int(input())

    xs = list()
    ys = list()

    for _ in range(n):
        x, y = map(int, input().split())
        xs.append(x - 1)
        ys.append(y - 1)

    res = cond_entropy(k_x, k_y, xs, ys)
    print(res)

