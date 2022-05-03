from typing import *


def main():
    n: int = int(input("체스판의 크기를 입력해주세요: "))

    print("n * n 체스판에 넣을 점수를 입력해주세요")
    board: List[List[int]] = [list(map(int, input(f"{i}번째 줄: "))) for i in range(n)]
