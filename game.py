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
