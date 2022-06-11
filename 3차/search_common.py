"""탐색을 통한 문제 해결을 위해 필요한 기반 구조들.
GitHub의 aima-python 코드를 기반으로 일부 내용을 수정하였음."""
import math


class Problem:
    """해결할 문제에 대한 추상 클래스
    다음 절차에 따라 이 클래스를 활용하여 문제해결하면 됨.
    1. 이 클래스의 서브클래스 생성 (이 서브클래스를 편의상 YourProblem이라고 하자)
    2. 다음 메쏘드들 구현
       - actions
       - result
       - 필요에 따라 h, __init__, is_goal, action_cost도
    3. YourProblem의 인스턴스를 생성
    4. 다양한 탐색 함수들을 사용해서 YourProblem을 해결"""

    def __init__(self, initial=None, goal=None, **kwds):
        """초기 상태(initial), 목표 상태(goal) 지정.
        필요에 따라 다른 파라미터들 추가"""
        self.initial = initial
        self.goal = goal

        self.__dict__.update(**kwds)  # __dict__: 객체의 속성 정보를 담고 있는 딕셔너리

    def actions(self, state):
        """행동: 주어진 상태(state)에서 취할 수 있는 행동들을 리턴함.
        대개 리스트 형태로 리턴하면 될 것임.
        한꺼번에 리턴하기에 너무 많은 행동들이 있을 경우, yield 사용을 검토할 것."""
        raise NotImplementedError

    def result(self, state, action):
        """이행모델: 주어진 상태(state)에서 주어진 행동(action)을 취했을 때의 결과 상태를 리턴함.
        리턴되는 상태는 self.actions(state) 중 하나여야 함."""
        raise NotImplementedError

    def is_goal(self, state):
        """목표검사: 상태가 목표 상태이면 True를 리턴함.
        상태가 self.goal과 일치하는지 혹은 self.goal이 리스트인 경우 그 중의 하나인지 체크함.
        더 복잡한 목표검사가 필요할 경우 이 메쏘드를 오버라이드하면 됨."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    @staticmethod
    def action_cost(state1, action, state2):
        """행동 비용: state1에서 action을 통해 state2에 이르는 비용을 리턴함.
        경로가 중요치 않은 문제의 경우에는 state2만을 고려한 함수가 될 것임.
        현재 구현된 기본 버전은 모든 상태에서 행동 비용을 1로 산정함."""
        return 1

    @staticmethod
    def h(node):
        """휴리스틱 함수:
        문제에 따라 휴리스틱 함수를 적절히 변경해줘야 함."""
        return 0

    def __str__(self):
        return f'{type(self).__name__}({self.initial!r}, {self.goal!r})'


def is_in(elt, seq):
    """elt가 seq의 원소인지 체크.
    (elt in seq)와 유사하나 ==(값의 비교)이 아닌 is(객체의 일치 여부)로 비교함."""
    return any(x is elt for x in seq)


class Node:
    """탐색 트리의 노드. 다음 요소들로 구성됨.
    - 이 노드를 생성한 부모에 대한 포인터
    - 이 노드에 대응되는 상태(한 상태에 여러 노드가 대응될 수도 있음)
    - 이 상태에 이르게 한 행동
    - 경로 비용(g)
    이 클래스의 서브클래스를 만들 필요는 없을 것임."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """parent에서 action을 취해 만들어지는 탐색 트리의 노드 생성"""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __repr__(self):
        return f"<{self.state}>"

    def __len__(self):  # 노드의 깊이
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other):
        return self.path_cost < other.path_cost


failure = Node('failure', path_cost=math.inf.__int__())  # 알고리즘이 해결책을 찾을 수 없음을 나타냄
cutoff = Node('cutoff', path_cost=math.inf.__int__())  # 반복적 깊이 증가 탐색이 중단(cut off)됐음을 나타냄


def expand(problem, node):
    """노드 확장: 이 노드에서 한 번의 움직임으로 도달 가능한 자식 노드들을 생성하여 yield함"""
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


def path_actions(node):
    """루트 노드에서부터 이 노드까지 이르는 행동 시퀀스. 결국 node가 목표 상태라면 이 행동 시퀀스는 해결책임."""
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    """루트 노드에서부터 이 노드까지 이르는 상태 시퀀스"""
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]
