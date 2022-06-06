import itertools

from logic import *


class KB:
    """신규 문장을 추가(tell)하거나 알려진 것을 질의(ask)할 수 있는 지식베이스(knowledge base).
    지식베이스를 생성하려면 이 클래스의 서브클래스로 정의하고 tell, ask_generator, retract 등을 구현하면 됨.
    ask_generator는 문장이 참이 되도록 하는 대입들을 찾고, ask는 이 중 첫번째를 리턴하거나 False 리턴."""

    def __init__(self, sentence):
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        """지식베이스에 문장 추가"""
        raise NotImplementedError

    def ask(self, query):
        """query를 참이 되게 하는 대입을 리턴함. 없으면 False 리턴."""
        return first(self.ask_generator(query), default=False)

    def ask_generator(self, query):
        """query가 참이 되는 모든 대입들을 생성"""
        raise NotImplementedError

    def retract(self, sentence):
        """지식베이스에서 문장 삭제"""
        raise NotImplementedError


class FolKB(KB):
    """일차논리 한정 절(definite clause)로 구성된 지식베이스.
    >>> kb0 = FolKB([expr('Farmer(Mac)'), expr('Rabbit(Pete)'),
    ...              expr('(Rabbit(r) & Farmer(f)) ==> Hates(f, r)')])
    >>> kb0.tell(expr('Rabbit(Flopsie)'))
    >>> kb0.retract(expr('Rabbit(Pete)'))
    >>> kb0.ask(expr('Hates(Mac, x)'))[x]
    Flopsie
    >>> kb0.ask(expr('Wife(Pete, x)'))
    False
    """

    def __init__(self, clauses=None):
        super().__init__(sentence=None)
        self.clauses = []
        if clauses:
            for clause in clauses:
                self.tell(clause)

    def tell(self, sentence):
        if is_definite_clause(sentence):
            self.clauses.append(sentence)
        else:
            raise Exception(f'Not a definite clause: {sentence}')

    def ask_generator(self, query):
        return fol_fc_ask(self, query)

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def fetch_rules_for_goal(self, goal):
        return self.clauses


def is_definite_clause(s):
    """Expr s가 한정 절이면 True를 리턴함.
    A & B & ... & C ==> D  (모두 양 리터럴)
    절 형식으로 표현하면,
    ~A | ~B | ... | ~C | D   (하나의 양 리터럴을 갖는 절)
    >>> is_definite_clause(expr('Farmer(Mac)'))
    True
    """
    if is_symbol(s.op):
        return True
    elif s.op == '==>':
        antecedent, consequent = s.args
        return is_symbol(consequent.op) and all(is_symbol(arg.op) for arg in conjuncts(antecedent))
    else:
        return False


def parse_definite_clause(s):
    """한정 절의 전제와 결론을 리턴"""
    assert is_definite_clause(s)
    if is_symbol(s.op):
        return [], s
    else:
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent


def conjuncts(s):
    """문장 s를 논리곱으로 해석 했을 때의 구성요소를 리스트로 리턴함.
    >>> conjuncts(A & B)
    [A, B]
    >>> conjuncts(A | B)
    [(A | B)]
    """
    return dissociate('&', [s])


def disjuncts(s):
    """문장 s를 논리합으로 해석했을 때의 구성요소를 리스트로 리턴함.
    >>> disjuncts(A | B)
    [A, B]
    >>> disjuncts(A & B)
    [(A & B)]
    """
    return dissociate('|', [s])


def dissociate(op, args):
    """op를 기준으로 인자들의 리스트를 리턴.
    >>> dissociate('&', [A & B])
    [A, B]
    """
    result = []

    def collect(subargs):
        for arg in subargs:
            if arg.op == op:
                collect(arg.args)
            else:
                result.append(arg)

    collect(args)
    return result


def fol_fc_ask(kb, alpha):
    """순방향 연쇄(forward chaining) 알고리즘"""
    kb_consts = list({c for clause in kb.clauses for c in constant_symbols(clause)})

    def enum_subst(p):
        query_vars = list({v for clause in p for v in variables(clause)})
        for assignment_list in itertools.product(kb_consts, repeat=len(query_vars)):
            theta = {x: y for x, y in zip(query_vars, assignment_list)}
            yield theta

    # 새로운 추론 없이도 답변할 수 있는지 체크
    for q in kb.clauses:
        phi = unify_mm(q, alpha)
        if phi is not None:
            yield phi

    while True:
        new = []
        for rule in kb.clauses:
            p, q = parse_definite_clause(rule)
            for theta in enum_subst(p):
                if set(subst(theta, p)).issubset(set(kb.clauses)):
                    q_ = subst(theta, q)
                    if all([unify_mm(x, q_) is None for x in kb.clauses + new]):
                        new.append(q_)
                        phi = unify_mm(q_, alpha)
                        if phi is not None:
                            yield phi
        if not new:
            break
        for clause in new:
            kb.tell(clause)
    return None


def unify_mm(x, y, s={}):
    """단일화. 규칙 기반으로 효율성을 개선한 알고리즘(Martelli & Montanari).
    >>> unify_mm(x, 3, {})
    {x: 3}
    """
    set_eq = extend(s, x, y)
    s = set_eq.copy()
    while True:
        trans = 0
        for x, y in set_eq.items():
            if x == y:
                # if x = y this mapping is deleted (rule b)
                del s[x]
            elif not is_variable(x) and is_variable(y):
                # if x is not a variable and y is a variable, rewrite it as y = x in s (rule a)
                if s.get(y, None) is None:
                    s[y] = x
                    del s[x]
                else:
                    # if a mapping already exist for variable y then apply
                    # variable elimination (there is a chance to apply rule d)
                    s[x] = vars_elimination(y, s)
            elif not is_variable(x) and not is_variable(y):
                # in which case x and y are not variables, if the two root function symbols
                # are different, stop with failure, else apply term reduction (rule c)
                if x.op is y.op and len(x.args) == len(y.args):
                    term_reduction(x, y, s)
                    del s[x]
                else:
                    return None
            elif isinstance(y, Expr):
                # in which case x is a variable and y is a function or a variable (e.g. F(z) or y),
                # if y is a function, we must check if x occurs in y, then stop with failure, else
                # try to apply variable elimination to y (rule d)
                if occur_check(x, y, s):
                    return None
                s[x] = vars_elimination(y, s)
                if y == s.get(x):
                    trans += 1
            else:
                trans += 1
        if trans == len(set_eq):
            # if no transformation has been applied, stop with success
            return s
        set_eq = s.copy()


def term_reduction(x, y, s):
    """x, y가 모두 함수이고 함수 기호가 동일한 경우 항 축소(term reduction)를 적용.
    예: x = F(x1, x2, ..., xn), y = F(x1', x2', ..., xn')
    x: y를 {x1: x1', x2: x2', ..., xn: xn'}로 대체한 새로운 매핑을 리턴.
    """
    for i in range(len(x.args)):
        if x.args[i] in s:
            s[s.get(x.args[i])] = y.args[i]
        else:
            s[x.args[i]] = y.args[i]


def vars_elimination(x, s):
    """변수 제거를 x에 적용함.
    x가 변수이고 s에 등장하면, x에 매핑된 항을 리턴함.
    x가 함수이면 함수의 각 항에 순환적으로 적용함."""
    if not isinstance(x, Expr):
        return x
    if is_variable(x):
        return s.get(x, x)
    return Expr(x.op, *[vars_elimination(arg, s) for arg in x.args])


def fol_bc_ask(kb, query):
    """역방향 연쇄(backward chaining) 알고리즘.
    kb는 FolKB 인스턴스이어야 하고, query는 기본 문장이어야 함.
    """
    return fol_bc_or(kb, query, {})


def fol_bc_or(kb, goal, theta):
    for rule in kb.fetch_rules_for_goal(goal):
        lhs, rhs = parse_definite_clause(standardize_variables(rule))
        for theta1 in fol_bc_and(kb, lhs, unify_mm(rhs, goal, theta)):
            yield theta1


def fol_bc_and(kb, goals, theta):
    if theta is None:
        pass
    elif not goals:
        yield theta
    else:
        first, rest = goals[0], goals[1:]
        for theta1 in fol_bc_or(kb, subst(theta, first), theta):
            for theta2 in fol_bc_and(kb, rest, theta1):
                yield theta2


def standardize_variables(sentence, dic=None):
    """변수 표준화: 문장의 모든 변수를 새로운 변수로 바꿈."""
    if dic is None:
        dic = {}
    if not isinstance(sentence, Expr):
        return sentence
    elif is_var_symbol(sentence.op):
        if sentence in dic:
            return dic[sentence]
        else:
            v = Expr('v_{}'.format(next(standardize_variables.counter)))
            dic[sentence] = v
            return v
    else:
        return Expr(sentence.op, *[standardize_variables(a, dic) for a in sentence.args])


standardize_variables.counter = itertools.count()


def problem_maker(people: int, liar: int, max_state: int):
    pass


def main():
    # 참고: https://m.blog.naver.com/dylan0301/221562384708
    print("거짓말쟁이 찾기 게임\n")

    # 문장을 저장하는 배열입니다.
    alpha_clauses = [
        expr('T(A) ==> T(A)'),  # A: A는 거짓말쟁이가 아니다.
        expr('F(A) ==> T(B)'),  # B: A는 거짓말쟁이다.
        expr('F(A) & T(B) ==> T(C)'),  # C: A는 거짓말쟁이고, B는 거짓말쟁이가 아니다.
        expr('T(B) & T(C) ==> T(D)'),  # D: B와 C는 거짓말쟁이가 아니다.
    ]

    # 문제의 답을 저장하는 배열입니다.
    # 답이 1개가 맞는지 검증하기 위함입니다.
    answers = []

    # 고전적인 방법으로, 각각의 화자를 거짓말쟁이로 가정한 후, 추론을 수행합니다.
    # 거짓말쟁이 1명에, 나머지가 다 진실을 말하는 것으로 나온다면 그것이 정답입니다.
    for i in ['A', 'B', 'C', 'D']:
        liar_kb = FolKB(clauses=alpha_clauses)
        liar_kb.tell(expr(f'F({i})'))

        print(f"{i}가 거짓말쟁이일 경우")
        liar_result = [list(i.values())[0] for i in liar_kb.ask_generator(expr('F(x)'))]
        truth_result = [list(i.values())[0] for i in liar_kb.ask_generator(expr('T(x)'))]

        print("거짓말쟁이 목록:", liar_result)
        print("진실쟁이 목록:", truth_result)
        print('\n')

        i_symbol = expr(i)
        if len(liar_result) != 1:  # 거짓말쟁이가 0명 혹은 2명 이상이 나온다면 오답입니다.
            continue
        if len(truth_result) != 3:  # 진실쟁이가 3명이 아니라면 오답입니다.
            continue
        if i_symbol not in liar_result:  # 선택한 화자가 거짓말쟁이가 아닌 것으로 바뀌었다면 오답입니다.
            continue
        if i_symbol in truth_result:  # 선택한 화자가 진실을 말하는 것으로 판명되었다면 오답입니다.
            continue
        if set(liar_result) & set(truth_result):  # 거짓말쟁이와 진실쟁이가 겹칠 경우에도 오답입니다.
            continue
        answers.append(i_symbol)  # 정답 목록에 추가합니다.

    print("최종 답:", answers)


if __name__ == "__main__":
    main()
