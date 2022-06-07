# 고전적 계획수립 문제 정의하기. 코드는 GitHub aima-python의 코드를 기반으로 일부 수정한 것임.
from infer import *


# 계획수립 문제 정의
class PlanningProblem:
    """
    Planning Domain Definition Language(PDDL): 계획수립을 탐색 문제로 정의하는데 사용됨.
    일차논리 문장으로 구성된 지식베이스에 상태들을 저장함.
    이 논리 문장들의 논리곱(conjunction)이 상태를 정의함.
    """

    def __init__(self, initial, goals, actions, domain=None):
        """
        initial: 초기 상태(초기 지식베이스 구성)
        goals: 문제에서 달성하고자 하는 목표 표현
        actions: 탐색 공간에서 실행될 Action 객체들의 리스트
        """
        self.initial = self.convert(initial) if domain is None else self.convert(initial) + self.convert(domain)
        self.goals = self.convert(goals)
        self.actions = actions
        self.domain = domain

    def convert(self, clauses):
        """문자열들을 expr들로 변환함"""
        if not isinstance(clauses, Expr):
            if len(clauses) > 0:
                clauses = expr(clauses)
            else:
                clauses = []
        try:
            clauses = conjuncts(clauses)
        except AttributeError:
            pass

        new_clauses = []
        for clause in clauses:
            if clause.op == '~':
                new_clauses.append(expr('Not' + str(clause.args[0])))
            else:
                new_clauses.append(clause)
        return new_clauses

    def expand_fluents(self, name=None):
        kb = None
        if self.domain:
            kb = FolKB(self.convert(self.domain))
            for action in self.actions:
                if action.precond:
                    for fests in set(action.precond).union(action.effect).difference(self.convert(action.domain)):
                        if fests.op[:3] != 'Not':
                            kb.tell(expr(str(action.domain) + ' ==> ' + str(fests)))

        objects = set(arg for clause in set(self.initial + self.goals) for arg in clause.args)
        fluent_list = []
        if name is not None:
            for fluent in self.initial + self.goals:
                if str(fluent) == name:
                    fluent_list.append(fluent)
                    break
        else:
            fluent_list = list(map(lambda fluent: Expr(fluent[0], *fluent[1]),
                                   {fluent.op: fluent.args for fluent in self.initial + self.goals +
                                    [clause for action in self.actions for clause in action.effect if
                                     clause.op[:3] != 'Not']}.items()))

        expansions = []
        for fluent in fluent_list:
            for permutation in itertools.permutations(objects, len(fluent.args)):
                new_fluent = Expr(fluent.op, *permutation)
                if (self.domain and kb.ask(new_fluent) is not False) or not self.domain:
                    expansions.append(new_fluent)

        return expansions

    def expand_actions(self, name=None):
        """전제조건의 변수 대입을 통해 모든 가능한 행동들 생성"""

        has_domains = all(action.domain for action in self.actions if action.precond)
        kb = None
        if has_domains:
            kb = FolKB(self.initial)
            for action in self.actions:
                if action.precond:
                    kb.tell(expr(str(action.domain) + ' ==> ' + str(action)))

        objects = set(arg for clause in self.initial for arg in clause.args)
        expansions = []
        action_list = []
        if name is not None:
            for action in self.actions:
                if str(action.name) == name:
                    action_list.append(action)
                    break
        else:
            action_list = self.actions

        for action in action_list:
            for permutation in itertools.permutations(objects, len(action.args)):
                bindings = unify_mm(Expr(action.name, *action.args), Expr(action.name, *permutation))
                if bindings is not None:
                    new_args = []
                    for arg in action.args:
                        if arg in bindings:
                            new_args.append(bindings[arg])
                        else:
                            new_args.append(arg)
                    new_expr = Expr(str(action.name), *new_args)
                    if (has_domains and kb.ask(new_expr) is not False) or (
                            has_domains and not action.precond) or not has_domains:
                        new_preconds = []
                        for precond in action.precond:
                            new_precond_args = []
                            for arg in precond.args:
                                if arg in bindings:
                                    new_precond_args.append(bindings[arg])
                                else:
                                    new_precond_args.append(arg)
                            new_precond = Expr(str(precond.op), *new_precond_args)
                            new_preconds.append(new_precond)
                        new_effects = []
                        for effect in action.effect:
                            new_effect_args = []
                            for arg in effect.args:
                                if arg in bindings:
                                    new_effect_args.append(bindings[arg])
                                else:
                                    new_effect_args.append(arg)
                            new_effect = Expr(str(effect.op), *new_effect_args)
                            new_effects.append(new_effect)
                        expansions.append(Action(new_expr, new_preconds, new_effects))

        return expansions

    def is_strips(self):
        """전제조건과 목표에 음리터럴을 포함하고 있지 않으면 True 리턴"""
        return (all(clause.op[:3] != 'Not' for clause in self.goals) and
                all(clause.op[:3] != 'Not' for action in self.actions for clause in action.precond))

    def goal_test(self):
        """목표 상태에 도달했는지 체크"""
        return all(goal in self.initial for goal in self.goals)

    def act(self, action):
        """행동 수행.
        행동은 Expr임. 예: expr('Remove(Glass, Table)') 또는 expr('Eat(Sandwich)')"""
        action_name = action.op
        args = action.args
        list_action = first(a for a in self.actions if a.name == action_name)
        if list_action is None:
            raise Exception("Action '{}' not found".format(action_name))
        if not list_action.check_precond(self.initial, args):
            raise Exception("Action '{}' pre-conditions not satisfied".format(action))
        self.initial = list_action(self.initial, args).clauses


class Action:
    """
    전제조건과 결과를 사용하는 행동 스키마 정의.
    이것을 사용하여 PlanningProblem에서 행동들을 표현함.
    행동은 변수가 인자로 주어지는 Expr임.
    전제조건과 결과는 양/음리터럴들로 구성되는 리스트.
    전제조건과 결과에서 음리터럴은 내부적으로 절 이름 앞에 'Not'을 붙인 형태로 저장됨.
    입력할 때는 ~형태로 입력해도 됨.
    예:
    precond = [expr("Human(person)"), expr("Hungry(Person)"), expr("NotEaten(food)")]
    effect = [expr("Eaten(food)"), expr("Hungry(person)")]
    eat = Action(expr("Eat(person, food)"), precond, effect)
    """

    def __init__(self, action, precond, effect, domain=None):
        if isinstance(action, str):
            action = expr(action)
        self.name = action.op
        self.args = action.args
        self.precond = self.convert(precond) if domain is None else self.convert(precond) + self.convert(domain)
        self.effect = self.convert(effect)
        self.domain = domain

    def __call__(self, kb, args):
        return self.act(kb, args)

    def __repr__(self):
        return '{}'.format(Expr(self.name, *self.args))

    def convert(self, clauses):
        """문자열들을 expr들로 변환함"""
        if isinstance(clauses, Expr):
            clauses = conjuncts(clauses)
            for i in range(len(clauses)):
                if clauses[i].op == '~':
                    clauses[i] = expr('Not' + str(clauses[i].args[0]))

        elif isinstance(clauses, str):
            clauses = clauses.replace('~', 'Not')
            if len(clauses) > 0:
                clauses = expr(clauses)

            try:
                clauses = conjuncts(clauses)
            except AttributeError:
                pass

        return clauses

    def relaxed(self):
        """행동의 결과에서 모든 음 리터럴을 제거하여 삭제 리스트를 행동에서 제거함"""
        return Action(Expr(self.name, *self.args), self.precond,
                      list(filter(lambda effect: effect.op[:3] != 'Not', self.effect)))

    def substitute(self, e, args):
        """변수들을 각각의 명제 기호로 교체"""
        new_args = list(e.args)
        for num, x in enumerate(e.args):
            for i, _ in enumerate(self.args):
                if self.args[i] == x:
                    new_args[num] = args[i]
        return Expr(e.op, *new_args)

    def check_precond(self, kb, args):
        """현 상태에서 전제조건이 충족됐는지 체크"""
        if isinstance(kb, list):
            kb = FolKB(kb)
        for clause in self.precond:
            if self.substitute(clause, args) not in kb.clauses:
                return False
        return True

    def act(self, kb, args):
        """상태의 지식베이스에 행동 실행"""
        if isinstance(kb, list):
            kb = FolKB(kb)

        if not self.check_precond(kb, args):
            raise Exception('Action pre-conditions not satisfied')
        for clause in self.effect:
            kb.tell(self.substitute(clause, args))
            if clause.op[:3] == 'Not':
                new_clause = Expr(clause.op[3:], *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:
                    kb.retract(self.substitute(new_clause, args))
            else:
                new_clause = Expr('Not' + clause.op, *clause.args)

                if kb.ask(self.substitute(new_clause, args)) is not False:
                    kb.retract(self.substitute(new_clause, args))

        return kb
