def bin_count(x):
    return bin(x).count('1')


def logic(n, m, y):
    print(2)
    print(n, 1)

    if n == 0:
        print(1)
        print(1)
        for i in range(m):
            print(-1.0, end=' ')
        print(-1.0)
        return

    for idx in range(n):
        for binary in range(m):
            if idx & (2 ** binary) > 0:
                print(1.0, end=' ')
            else:
                print(-1e7, end=' ')
        print(0.5 - bin_count(idx))

    print(*y)
    print(-0.5)


if __name__ == '__main__':
    m = int(input())
    y = list()

    for _ in range(2 ** m):
        y.append(int(input()))

    n = 2 ** m

    logic(n, m, y)
