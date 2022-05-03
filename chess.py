import collections
import copy
from typing import *

dy = [-1, 1, -2, 2, -1, 1, -2, 2]
dx = [-2, -2, -1, -1, 2, 2, 1, 1]


class Game:
    """문제를 정의하기 위한 Problem 클래스에 대응되는 게임 정의를 위한 클래스.
    경로 비용과 목표 검사 대신 각 상태에 대한 효용 함수와 종료 검사로 구성됨.
    게임을 정의하려면 이 클래스의 서브 클래스를 만들어서
    actions, result, is_terminal, utility를 구현하면 됨.
    필요에 따라 게임의 초기 상태를 지정하려면,
    클래스 생성자에서 초기 상태를 initial 에 세팅하면 됨."""

    def actions(self, state):
        """주어진 상태에서 허용 가능한 수(move) 리스트"""
        raise NotImplementedError

    def result(self, state, move):
        """주어진 상태(state)에서 수(move)를 두었을 때의 결과 상태 리턴"""
        raise NotImplementedError

    def is_terminal(self, state):
        """state가 종료 상태이면 True 리턴"""
        return not self.actions(state)

    def utility(self, state, player):
        """종료 상태 state에서 게임이 종료됐을 때 player의 효용 함수 값"""
        raise NotImplementedError


class Node:
    """
    상태공간 노드입니다.
    """

    def __init__(self, size: int, y: int, x: int, board: List[List], point: int, turn: int):
        self.size: int = size
        self.y: int = y
        self.x: int = x
        self.board: List[List] = board
        self.point: int = point
        self.turn: int = turn  # 현재 말의 턴을 저장합니다. 0: max, 1: min


class ChessGame(Game):
    def __init__(self):
        self.initial = Node()
    def actions(self, state) -> List[Tuple[int, int]]:
        result = []

        for i in range(8):
            ny, nx = state.y + dy[i], state.x + dx[i]

            if not (0 <= ny < state.size):
                continue
            if not (0 <= nx < state.size):
                continue
            if state.board[ny][nx] == -1:
                continue

            result.append((ny, nx))

        return result

    def result(self, state: Node, move: Tuple[int, int]):
        next_state = copy.deepcopy(state)
        y, x = move
        next_state.point += next_state.board[y][x]
        next_state.board[y][x] = -1
        return next_state

    def utility(self, state: Node, player: int):
        return state.point

    def alpha_beta(self, state: Node):
        ans = int(-1e9)
        size = state.size

        for i in range(size):
            for j in range(size):
                state.y, state.x = i, j
                ans = max(ans, self.max_value(state, int(-1e9), int(1e9)))

    def max_value(self, state: Node, alpha: int, beta: int):
        if self.is_terminal(state):
            return self.utility(state, state.turn), None

        value = int(-1e9)
        move: Tuple[int, int] = (0, 0)

        for action in self.actions(state):
            value2, _ = self.min_value(self.result(state, action), alpha, beta)

            if value < value2:
                value, move = value2, action
                alpha = max(alpha, value)

            if value >= beta:
                return value, move

        return value, move

    def min_value(self, state: Node, alpha: int, beta: int):
        if self.is_terminal(state):
            return self.utility(state, state.turn), None

        value = int(1e9)
        move: Tuple[int, int] = (0, 0)

        for action in self.actions(state):
            value2, _ = self.max_value(self.result(state, action), alpha, beta)

            if value2 < value:
                value, move = value2, action
                beta = min(beta, value)

            if value <= alpha:
                return value, move

        return value, move


if __name__ == '__main__':
    print(f"플레이어 A와 B의 최대 점수차이는 {play_game(ChessGame()).utility()}점 입니다.")