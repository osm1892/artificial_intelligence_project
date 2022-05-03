import game
from typing import *


class Knight:
    def __init__(self, n: int, y: int, x: int):
        self.size = n
        self.pos: List[int, int] = [y, x]

    def set_pos(self, y, x):
        # y, x가 범위 내인지 체크
        assert 0 <= y < self.size
        assert 0 <= x < self.size

        self.pos = [y, x]

    def get_pos(self):
        return self.pos

