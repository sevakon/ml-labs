def chi_squared(k1, k2, xs, ys):
    stats_x = [0 for _ in range(k1)]
    stats_y  = [0 for _ in range(k2)]

    d = dict()

    for x, y in zip(xs, ys):

        if (x,y) in d:
            d[(x, y)] += 1
        else:
            d[(x, y)] = 1

        stats_x[x] += 1
        stats_y[y] += 1

    result = 0.

    for (x, y) in d.keys():
        e = stats_x[x] * stats_y[y] / n
        s = (d[(x, y)] - e) ** 2 / e - e
        result += s

    return result


if __name__ == '__main__':
    k1, k2 = map(int, input().split())
    n = int(input())

    xs = list()
    ys = list()

    for _ in range(n):
        x, y = map(int, input().split())

        xs.append(x - 1)
        ys.append(y - 1)

    res = chi_squared(k1, k2, xs, ys)

    print(res + n)

