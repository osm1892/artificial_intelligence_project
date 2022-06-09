# 고전적 계획수립 관련 코드가 planning.py에 저장되어 있음
from collections import deque

from planning import *
from search_common import Node


class HLA(Action):
    """고수준 행동"""
    unique_group = 1

    def __init__(self, action, precond=None, effect=None, duration=0, consume=None, use=None):
        """
        기본 행동(Action)에 제약조건 추가.
        - duration: 태스크 실행에 요구되는 시간 기간
        - consume: 태스크가 소모하는 소모성 자원을 표현하는 사전
        - use: 태스크가 사용하는 재사용 가능 자원을 표현하는 사전
        """
        precond = precond or [None]
        effect = effect or [None]
        super().__init__(action, precond, effect)
        self.duration = duration
        self.consumes = consume or {}
        self.uses = use or {}
        self.completed = False

    def do_action(self, job_order, available_resources, kb, args):
        """
        HLA에 기반한 act.
        지식베이스 업데이트뿐만 아니라, 자원을 체크하고 행동이 올바른 순서로 실행되도록 보장함.
        """
        if not self.has_usable_resource(available_resources):
            raise Exception('Not enough usable resources to execute {}'.format(self.name))
        if not self.has_consumable_resource(available_resources):
            raise Exception('Not enough consumable resources to execute {}'.format(self.name))
        if not self.inorder(job_order):
            raise Exception("Can't execute {} - execute prerequisite actions first".
                            format(self.name))
        kb = super().act(kb, args)  # 지식베이스 업데이트
        for resource in self.consumes:  # 소모된 자원 제거
            available_resources[resource] -= self.consumes[resource]
        self.completed = True  # 작업 상태를 완료로 설정
        return kb

    def has_consumable_resource(self, available_resources):
        """
        이 행동이 실행되는데 필요한 소모성 자원이 충분한지 확인
        """
        for resource in self.consumes:
            if available_resources.get(resource) is None:
                return False
            if available_resources[resource] < self.consumes[resource]:
                return False
        return True

    def has_usable_resource(self, available_resources):
        """
        이 행동이 실행되는데 필요한 재사용 가능 자원이 충분한지 확인
        """
        for resource in self.uses:
            if available_resources.get(resource) is None:
                return False
            if available_resources[resource] < self.uses[resource]:
                return False
        return True

    def inorder(self, job_order):
        """
        현재 작업 전에 실행되었어야 할 모든 작업들이 성공적으로 실행됐는지 확인
        """
        for jobs in job_order:
            if self in jobs:
                for job in jobs:
                    if job is self:
                        return True
                    if not job.completed:
                        return False
        return True


class RealWorldPlanningProblem(PlanningProblem):
    """
    자원을 이름 대신 수량으로 종합하여 표현함.
    HLA로 표현된 자원과 순서 조건을 처리하도록 act 함수를 오버로딩.
    """

    def __init__(self, initial, goals, actions, jobs=None, resources=None):
        super().__init__(initial, goals, actions)
        self.jobs = jobs
        self.resources = resources or {}

    def act(self, action):
        """
        HLA를 수행함.
        """
        args = action.args
        list_action = first(a for a in self.actions if a.name == action.name)
        if list_action is None:
            raise Exception("Action '{}' not found".format(action.name))
        self.initial = list_action.do_action(self.jobs, self.resources, self.initial, args).clauses

    def refinements(self, library):
        """
        library: 모든 가능한 세분들(refinements)에 대한 상세사항들을 포함하는 사전
        예:
        {
        'HLA': [
            'Go(Home, SFO)',
            'Go(Home, SFO)',
            'Drive(Home, SFOLongTermParking)',
            'Shuttle(SFOLongTermParking, SFO)',
            'Taxi(Home, SFO)'
            ],
        'steps': [
            ['Drive(Home, SFOLongTermParking)', 'Shuttle(SFOLongTermParking, SFO)'],
            ['Taxi(Home, SFO)'],
            [],
            [],
            []
            ],
        # 빈 refinement는 기본 행동을 의미함.
        'precond': [
            ['At(Home) & Have(Car)'],
            ['At(Home)'],
            ['At(Home) & Have(Car)'],
            ['At(SFOLongTermParking)'],
            ['At(Home)']
            ],
        'effect': [
            ['At(SFO) & ~At(Home)'],
            ['At(SFO) & ~At(Home)'],
            ['At(SFOLongTermParking) & ~At(Home)'],
            ['At(SFO) & ~At(SFOLongTermParking)'],
            ['At(SFO) & ~At(Home)']
            ]}
        """
        indices = [i for i, x in enumerate(library['HLA']) if expr(x).op == self.name]
        for i in indices:
            actions = []
            for j in range(len(library['steps'][i])):
                # HLA의 step[j]의 인텍스를 찾음
                index_step = [k for k, x in enumerate(library['HLA']) if x == library['steps'][i][j]][0]
                precond = library['precond'][index_step][0]  # step[j]의 전제조건
                effect = library['effect'][index_step][0]  # step[j]의 결과
                actions.append(HLA(library['steps'][i][j], precond, effect))
            yield actions

    def hierarchical_search(self, hierarchy):
        """
        계층적 탐색: 계층적 순방향 계획수립 탐색(BFS 버전).
        hierarchy: HLA - 세분들(refinements)의 사전구조
        """
        act = Node(self.initial, None, [self.actions[0]])
        frontier = deque()
        frontier.append(act)
        while True:
            if not frontier:
                return None
            plan = frontier.popleft()
            # plan의 행동들 중 첫번째 (기본 행동이 아닌) HLA를 찾아냄
            (hla, index) = RealWorldPlanningProblem.find_hla(plan, hierarchy)
            prefix = plan.action[:index]
            outcome = RealWorldPlanningProblem(
                RealWorldPlanningProblem.result(self.initial, prefix), self.goals, self.actions)
            suffix = plan.action[index + 1:]
            if not hla:
                if outcome.goal_test():
                    return plan.action
            else:
                for sequence in RealWorldPlanningProblem.refinements(hla, hierarchy):
                    frontier.append(Node(outcome.initial, plan, prefix + sequence + suffix))

    def result(state, actions):
        """문제에 행동을 적용했을 때의 결과"""
        for a in actions:
            if a.check_precond(state, a.args):
                state = a(state, a.args).clauses
        return state

    def angelic_search(self, hierarchy, initial_plan):
        """
        목표를 달성하는 고수준 계획을 인식하여 그 계획을 세분화하는 계층적 계획수립 알고리즘.
        목표를 달성하지 않는 고수준 계획은 회피함.
        initial_plan에 초기 계획을 세팅하여 호출하면 됨.
        initial_plan에 있는 angelic HLA의 결과 예:
        ~ : 결과 삭제
        $+: 결과 추가 가능
        $-: 결과 삭제 가능
        $$: 추가나 삭제 가능
        """
        frontier = deque(initial_plan)
        while True:
            if not frontier:
                return None
            plan = frontier.popleft()  # HLA 및 angelic HLA의 시퀀스
            opt_reachable_set = RealWorldPlanningProblem.reach_opt(self.initial, plan)
            pes_reachable_set = RealWorldPlanningProblem.reach_pes(self.initial, plan)
            if self.intersects_goal(opt_reachable_set):
                if RealWorldPlanningProblem.is_primitive(plan, hierarchy):
                    return [x for x in plan.action]
                guaranteed = self.intersects_goal(pes_reachable_set)
                if guaranteed and RealWorldPlanningProblem.making_progress(plan, initial_plan):
                    final_state = guaranteed[0]  # guaranteed 중 임의의 한 항목
                    return RealWorldPlanningProblem.decompose(hierarchy, plan, final_state, pes_reachable_set)
                hla, index = RealWorldPlanningProblem.find_hla(plan, hierarchy)
                prefix = plan.action[:index]
                suffix = plan.action[index + 1:]
                outcome = RealWorldPlanningProblem(
                    RealWorldPlanningProblem.result(self.initial, prefix), self.goals, self.actions)
                for sequence in RealWorldPlanningProblem.refinements(hla, hierarchy):
                    frontier.append(
                        AngelicNode(outcome.initial, plan, prefix + sequence + suffix, prefix + sequence + suffix))

    def intersects_goal(self, reachable_set):
        """
        도달 가능 집합과 목표의 교집합을 찾음.
        """
        return [y for x in list(reachable_set.keys())
                for y in reachable_set[x]
                if all(goal in y for goal in self.goals)]

    def is_primitive(plan, library):
        """
        plan이 기본(primitive) 계획인지 확인.
        plan의 action에 HLA가 하나라도 포함되어 있으면 False,
        HLA가 하나도 없으면(모두 기본 행동이면) True 리턴.
        """
        for hla in plan.action:
            indices = [i for i, x in enumerate(library['HLA']) if expr(x).op == hla.name]
            for i in indices:
                if library["steps"][i]:
                    return False
        return True

    def reach_opt(init, plan):
        """
        plan의 행동 시퀀스에 대해 낙관적 도달 가능 집합을 찾음.
        """
        reachable_set = {0: [init]}
        optimistic_description = plan.action  # list of angelic actions with optimistic description
        return RealWorldPlanningProblem.find_reachable_set(reachable_set, optimistic_description)

    def reach_pes(init, plan):
        """
        plan의 행동 시퀀스에 대해 비관적 도달 가능 집합을 찾음.
        """
        reachable_set = {0: [init]}
        pessimistic_description = plan.action_pes  # list of angelic actions with pessimistic description
        return RealWorldPlanningProblem.find_reachable_set(reachable_set, pessimistic_description)

    def find_reachable_set(reachable_set, action_description):
        """
        도달 가능 집합의 각 상태에 행동 표현 action_description을 적용했을 때 도달 가능한 상태를 찾음.
        """
        for i in range(len(action_description)):
            reachable_set[i + 1] = []
            if type(action_description[i]) is AngelicHLA:
                possible_actions = action_description[i].angelic_action()
            else:
                possible_actions = action_description
            for action in possible_actions:
                for state in reachable_set[i]:
                    if action.check_precond(state, action.args):
                        if action.effect[0]:
                            new_state = action(state, action.args).clauses
                            reachable_set[i + 1].append(new_state)
                        else:
                            reachable_set[i + 1].append(state)
        return reachable_set

    def find_hla(plan, hierarchy):
        """
        plan.action에서 기본 행동이 아닌 첫번째 HLA와 그 인덱스를 찾음.
        """
        hla = None
        index = len(plan.action)
        for i in range(len(plan.action)):  # find the first HLA in plan, that is not primitive
            if not RealWorldPlanningProblem.is_primitive(Node(plan.state, plan.parent, [plan.action[i]]), hierarchy):
                hla = plan.action[i]
                index = i
                break
        return hla, index

    def making_progress(plan, initial_plan):
        """
        refinement regression이 무한 루프에 빠지지 않도록 방지함.
        """
        for i in range(len(initial_plan)):
            if plan == initial_plan[i]:
                return False
        return True

    def decompose(hierarchy, plan, s_f, reachable_set):
        solution = []
        i = max(reachable_set.keys())
        while plan.action_pes:
            action = plan.action_pes.pop()
            if i == 0:
                return solution
            s_i = RealWorldPlanningProblem.find_previous_state(s_f, reachable_set, i, action)
            problem = RealWorldPlanningProblem(s_i, s_f, plan.action)
            angelic_call = RealWorldPlanningProblem.angelic_search(problem, hierarchy,
                                                                   [AngelicNode(s_i, Node(None), [action], [action])])
            if angelic_call:
                for x in angelic_call:
                    solution.insert(0, x)
            else:
                return None
            s_f = s_i
            i -= 1
        return solution

    def find_previous_state(s_f, reachable_set, i, action):
        """
        regression(역행).
        상태 s_i에 행동 action이 적용됐을 때 s_f(최종 상태)를 리턴하는 상태 s_i를 도달 가능 집합에서 찾음.
        """
        s_i = reachable_set[i - 1][0]
        for state in reachable_set[i - 1]:
            if s_f in [x for x in RealWorldPlanningProblem.reach_pes(
                    state, AngelicNode(state, None, [action], [action]))[1]]:
                s_i = state
                break
        return s_i


class AngelicHLA(HLA):
    """
    천사적 표현에 기반한 HLA
    """

    def __init__(self, action, precond, effect, duration=0, consume=None, use=None):
        super().__init__(action, precond, effect, duration, consume, use)

    def convert(self, clauses):
        """
        문자열을 Expr로 변환.
        AngelicHLA는 HLA의 결과뿐만 아니라 변수에 대한 다음과 같은 결과를 추가적으로 포함할 수 있음:
        - 변수 추가 가능 ( $+ )
        - 변수 삭제 가능 ( $- )
        - 변수 추가/삭제 가능 ( $$ )
        """
        lib = {'~': 'Not',
               '$+': 'PosYes',
               '$-': 'PosNot',
               '$$': 'PosYesNot'}

        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                for ch in lib.keys():
                    if clauses[i].op == ch:
                        clauses[i] = expr(lib[ch] + str(clauses[i].args[0]))

        elif isinstance(clauses, str):
            for ch in lib.keys():
                clauses = clauses.replace(ch, lib[ch])
            if len(clauses) > 0:
                clauses = expr(clauses)

            try:
                clauses = conjuncts(clauses)
            except AttributeError:
                pass

        return clauses

    def angelic_action(self):
        """
        AngelicHLA를 상응하는 HLA들로 변환.
        - 변수 추가 가능 ( $+: 'PosYes' )          -->  HLA_1: 변수 추가
                                                        HLA_2: 변수 그대로 유지
        - 변수 삭제 가능 ( $-: 'PosNot' )          -->  HLA_1: 변수 삭제
                                                        HLA_2: 변수 그대로 유지
        - 변수 추가/삭제 가능 ( $$: 'PosYesNot' )  -->  HLA_1: 변수 추가
                                                        HLA_2: 변수 삭제
                                                        HLA_3: 변수 그대로 유지
        예:
            '$+A & $$B':    HLA_1: 'A & B'   (add A and add B)
                            HLA_2: 'A & ~B'  (add A and remove B)
                            HLA_3: 'A'       (add A)
                            HLA_4: 'B'       (add B)
                            HLA_5: '~B'      (remove B)
                            HLA_6: ' '       (no effect)
        """

        effects = [[]]
        for clause in self.effect:
            (n, w) = AngelicHLA.compute_parameters(clause)
            effects = effects * n  # effects를 n개 복사
            it = range(1)
            if len(effects) != 0:
                # effects를 n개의 서브리스트로 분리
                it = range(len(effects) // n)
            for i in it:  # effects의 i번째 항목 수정
                if effects[i]:
                    if clause.args:
                        effects[i] = expr(str(effects[i]) + '&' + str(
                            Expr(clause.op[w:], clause.args[0])))
                        if n == 3:
                            effects[i + len(effects) // 3] = expr(
                                str(effects[i + len(effects) // 3]) + '&' + str(Expr(clause.op[6:], clause.args[0])))
                    else:
                        effects[i] = expr(
                            str(effects[i]) + '&' + str(expr(clause.op[w:])))
                        if n == 3:
                            effects[i + len(effects) // 3] = expr(
                                str(effects[i + len(effects) // 3]) + '&' + str(expr(clause.op[6:])))

                else:
                    if clause.args:
                        effects[i] = Expr(clause.op[w:], clause.args[0])
                        if n == 3:
                            effects[i + len(effects) // 3] = Expr(clause.op[6:], clause.args[0])

                    else:
                        effects[i] = expr(clause.op[w:])
                        if n == 3:
                            effects[i + len(effects) // 3] = expr(clause.op[6:])

        return [HLA(Expr(self.name, self.args), self.precond, effects[i]) for i in range(len(effects))]

    def compute_parameters(clause):
        """
        n = angelic HLA에 상응하는 HLA 결과의 수
        w = angelic HLA 결과 표현의 길이
                    n = 1, if effect is add
                    n = 1, if effect is remove
                    n = 2, if effect is possibly add
                    n = 2, if effect is possibly remove
                    n = 3, if effect is possibly add or remove
        """
        if clause.op[:9] == 'PosYesNot':
            # 변수 추가/삭제 가능: 변수에 대한 3가지 결과 가능
            n = 3
            w = 9
        elif clause.op[:6] == 'PosYes':  # 변수 추가 가능: 변수에 대한 2가지 결과 가능
            n = 2
            w = 6
        elif clause.op[:6] == 'PosNot':  # 변수 삭제 가능: 변수에 대한 2가지 결과 가능
            n = 2
            w = 3  # We want to keep 'Not' from 'PosNot' when adding action
        else:  # variable or ~variable
            n = 1
            w = 0
        return n, w


class AngelicNode(Node):
    """
    angelic HLA를 반영할 수 있도록 Node 클래스를 확장.
    self.action: angelic HLA의 낙관적 표현을 포함함.
    self.action_pes: angelic HLA의 비관적 표현을 포함함.
    """

    def __init__(self, state, parent=None, action_opt=None, action_pes=None, path_cost=0):
        super().__init__(state, parent, action_opt, path_cost)
        self.action_pes = action_pes

def main():
    library = {
        'HLA': ['Make(Pasta)', 'Boil(Water)', 'Boil(Noodle)', 'Prepare(TomatoSauce)', 'Prepare(Ingredients)', 'Make(Sauce)', 'Mix(Noodle, Sauce)'],
        'steps': [['Boil(Noodle)', 'Make(Sauce)', 'Mix(Noodle, Sauce)'], []],
        'precond': [],
        'effect': [],
    }

    boil_water = HLA('Boil(Water)', precond='Clean(Pot) & Empty(Pot)', effect='~Empty(Pot) & In(Pot, Water) & Hot(Pot)')
    boil_noodle = HLA('Boil(Noodle)', precond='Boil(Water) & Clean(Pot)', effect='~Clean(Pot) & In(Pot, Noodle)')
    clean_pot = HLA('Clean(Pot)', precond='', effect='Clean(Pot) & Empty(Pot) & ~Hot(Pot)')
