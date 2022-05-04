import copy
import math
from typing import *

dy = [-1, 1, -2, 2, -1, 1, -2, 2]
dx = [-2, -2, -1, -1, 2, 2, 1, 1]

infinity = math.inf


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

    def utility(self):
        return self.point


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


def alphabeta_search(game, state):
    """알파-베타 가지치기를 사용하여 최고의 수를 결정하기 위한 게임 트리 탐색."""

    player = state.to_move

    def max_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    def min_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    return max_value(state, -infinity, +infinity)


def play_game(game, strategies: dict, verbose=False) -> Node:
    """번갈아 가면서 두는 게임 진행.
    strategies: {참가자 이름: 함수} 형태의 딕셔너리.
    함수(game, state)는 상태 state에서 참가자의 수를 찾는 함수"""
    state = game.initial
    while not game.is_terminal(state):
        player = state.to_move
        move = strategies[player](game, state)
        state = game.result(state, move)
        if verbose:
            print('Player', player, 'move:', move)
            print(state)
    return state


def game_player(search_algorithm):
    """지정된 탐색 알고리즘을 사용하는 플레이어: (game, state)를 입력 받아 move를 리턴하는 함수."""
    return lambda game, state: search_algorithm(game, state)[1]


if __name__ == '__main__':
    n: int = int(input("체스판의 크기를 입력해주세요: "))

    print("n * n 체스판에 넣을 점수를 입력해주세요")
    board: List[List[int]] = [list(map(int, input(f"{i}번째 줄: "))) for i in range(n)]

    print(
        f"플레이어 A와 B의 최대 점수차이는 {play_game(ChessGame(), {'a': game_player(alphabeta_search), 'b': game_player(alphabeta_search)}).utility()}점 입니다.")
