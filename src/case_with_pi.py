import random


def main():
    pi = count_pi(1000000)
    print(f"По нашим расчетам число пи равно {pi}")


def count_pi(n: int) -> float:
    i = 0
    count = 0
    while i < n:
        if (pow(random.random(), 2) + pow(random.random(), 2)) < 1:
            count += 1
        i += 1
    return 4 * (count / n)


if __name__ == "__main__":
    main()
