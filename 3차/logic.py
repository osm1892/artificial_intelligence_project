# 논리적 에이전트, 명제논리, 일차논리 표현과 관련된 사항들. 코드는 GitHub aima-python의 코드를 기반으로 일부 수정한 것임.

import collections


# # 논리 문장 표현을 위한 준비
# `Expr`: 논리 문장을 포함한 수학식 표현을 위한 클래스
class Expr:
    """논리 문장을 포함한 수학식(연산자와 0개 이상의 피연산자 포함) 표현을 위한 클래스.
    op: 문자열 (예: '+', 'sin')
    args: Expression"""

    def __init__(self, op, *args):
        self.op = str(op)
        self.args = args

    # 각종 연산자를 Expr에서 사용하기 위한 연산자 오버로딩
    def __neg__(self):
        return Expr('-', self)

    def __pos__(self):
        return Expr('+', self)

    def __invert__(self):
        return Expr('~', self)

    def __add__(self, rhs):
        return Expr('+', self, rhs)

    def __sub__(self, rhs):
        return Expr('-', self, rhs)

    def __mul__(self, rhs):
        return Expr('*', self, rhs)

    def __pow__(self, rhs):
        return Expr('**', self, rhs)

    def __mod__(self, rhs):
        return Expr('%', self, rhs)

    def __and__(self, rhs):
        return Expr('&', self, rhs)

    def __xor__(self, rhs):
        return Expr('^', self, rhs)

    def __rshift__(self, rhs):
        return Expr('>>', self, rhs)

    def __lshift__(self, rhs):
        return Expr('<<', self, rhs)

    def __truediv__(self, rhs):
        return Expr('/', self, rhs)

    def __floordiv__(self, rhs):
        return Expr('//', self, rhs)

    def __matmul__(self, rhs):
        return Expr('@', self, rhs)

    def __or__(self, rhs):
        """P | Q 형식이나 P |'==>'| Q 형식을 허용하도록."""
        if isinstance(rhs, Expression):
            return Expr('|', self, rhs)
        else:
            return PartialExpr(rhs, self)

    # reverse 연산자 오버로딩: Expr 객체가 연산자의 오른쪽 피연산자로 사용됐을 때도 정상 동작하도록.
    def __radd__(self, lhs):
        return Expr('+', lhs, self)

    def __rsub__(self, lhs):
        return Expr('-', lhs, self)

    def __rmul__(self, lhs):
        return Expr('*', lhs, self)

    def __rdiv__(self, lhs):
        return Expr('/', lhs, self)

    def __rpow__(self, lhs):
        return Expr('**', lhs, self)

    def __rmod__(self, lhs):
        return Expr('%', lhs, self)

    def __rand__(self, lhs):
        return Expr('&', lhs, self)

    def __rxor__(self, lhs):
        return Expr('^', lhs, self)

    def __ror__(self, lhs):
        return Expr('|', lhs, self)

    def __rrshift__(self, lhs):
        return Expr('>>', lhs, self)

    def __rlshift__(self, lhs):
        return Expr('<<', lhs, self)

    def __rtruediv__(self, lhs):
        return Expr('/', lhs, self)

    def __rfloordiv__(self, lhs):
        return Expr('//', lhs, self)

    def __rmatmul__(self, lhs):
        return Expr('@', lhs, self)

    def __call__(self, *args):
        """'f'가 기호라면, f(0) == Expr('f', 0)"""
        if self.args:
            raise ValueError('Can only do a call for a Symbol, not an Expr')
        else:
            return Expr(self.op, *args)

    # ==, repr
    def __eq__(self, other):
        """x == y: op와 args가 모두 동일한 Expr일 경우 True를 리턴."""
        return isinstance(other, Expr) and self.op == other.op and self.args == other.args

    def __lt__(self, other):
        return isinstance(other, Expr) and str(self) < str(other)

    def __hash__(self):
        return hash(self.op) ^ hash(self.args)

    def __repr__(self):
        op = self.op
        args = [str(arg) for arg in self.args]
        if op.isidentifier():  # f(x) or f(x, y)
            return f"{op}({', '.join(args)})" if args else op
        elif len(args) == 1:  # -x or -(x + 1)
            return op + args[0]
        else:  # (x - y)
            opp = (' ' + op + ' ')
            return '(' + opp.join(args) + ')'


# Expression은 Expr 또는 숫자(Number)
Number = (int, float, complex)
Expression = (Expr, Number)


# 기호(Symbol): 인자가 없는 Expr
def Symbol(name):
    """인자가 없는 Expr"""
    return Expr(name)


def symbols(names):
    """여러 기호 생성 시 사용. Symbol들의 튜플을 리턴함.
    이름은 콤마나 공백으로 구분된 문자열."""
    return tuple(Symbol(name) for name in names.replace(',', ' ').split())


def subexpressions(x):
    """Expression x를 구성하는 부분 Expression(인자)들을 리턴함(자기 자신 포함)."""
    yield x
    if isinstance(x, Expr):
        for arg in x.args:
            yield from subexpressions(arg)


def arity(expression):
    """expression의 인자 개수"""
    if isinstance(expression, Expr):
        return len(expression.args)
    else:  # 숫자인 경우
        return 0


# Python에 정의되어 있지 않은 연산자를 사용하기 위해 새로운 infixOps 정의
class PartialExpr:
    """조건문 'P |'==>'| Q은 먼저 PartialExpr('==>', P)를 생성한 후 Q를 결합함."""

    def __init__(self, op, lhs):
        self.op, self.lhs = op, lhs

    def __or__(self, rhs):
        return Expr(self.op, self.lhs, rhs)

    def __repr__(self):
        return "PartialExpr('{}', {})".format(self.op, self.lhs)


def expr(x):
    """Expression 생성을 간편하게 수행하기 위한 함수. x는 문자열.
    x 문자열에 포함된 식별자는 자동으로 Symbol로 정의됨.
    ==>, <==, <=>는 각각의 infix 연산자(예: |'==>'|)로 취급됨.
    x가 이미 Expression이면 그대로 리턴.
    >>> expr('P & Q ==> Q')
    ((P & Q) ==> Q)
    """
    return eval(expr_handle_infix_ops(x), defaultkeydict(Symbol)) if isinstance(x, str) else x


infix_ops = '==> <== <=>'.split()


def expr_handle_infix_ops(x):
    """infix 연산자로 변환. 예: ==> 를 |'==>'| 로 변환)
    >>> expr_handle_infix_ops('P ==> Q')
    "P |'==>'| Q"
    """
    for op in infix_ops:
        x = x.replace(op, '|' + repr(op) + '|')
    return x


class defaultkeydict(collections.defaultdict):
    """default_factory가 key의 함수임.
    >>> d = defaultkeydict(len); d['abcde']
    5
    """

    def __missing__(self, key):
        self[key] = result = self.default_factory(key)
        return result


def is_symbol(s):
    """기호: 알파벳 문자로 시작하는 문자열.
    >>> is_symbol('R2D2')
    True
    """
    return isinstance(s, str) and s[:1].isalpha()


def is_var_symbol(s):
    """논리 변수 기호: 소문자로 시작하는 문자열.
    >>> is_var_symbol('EXE')
    False
    """
    return is_symbol(s) and s[0].islower()


def is_prop_symbol(s):
    """명제논리 기호: 대문자로 시작하는 문자열.
    >>> is_prop_symbol('exe')
    False
    """
    return is_symbol(s) and s[0].isupper()


def variables(s):
    """Expr s에 등장하는 변수 집합을 리턴함.
    >>> variables(expr('F(x, x) & G(x, y) & H(y, z) & R(A, z, 2)')) == {x, y, z}
    True
    """
    return {x for x in subexpressions(s) if is_variable(x)}


def is_variable(x):
    """변수: args가 없고 op가 소문자 기호로 구성된 Expr"""
    return isinstance(x, Expr) and not x.args and x.op[0].islower()


def constant_symbols(x):
    """x에 존재하는 모든 상수 기호의 집합을 리턴"""
    if not isinstance(x, Expr):
        return set()
    elif is_prop_symbol(x.op) and not x.args:
        return {x}
    else:
        return {symbol for arg in x.args for symbol in constant_symbols(arg)}


def predicate_symbols(x):
    """x에 존재하는 (기호명, 인자수)의 집합을 리턴함.
    인자의 개수가 >0인 모든 기호(함수 포함)를 고려함."""
    if not isinstance(x, Expr) or not x.args:
        return set()
    pred_set = {(x.op, len(x.args))} if is_prop_symbol(x.op) else set()
    pred_set.update({symbol for arg in x.args for symbol in predicate_symbols(arg)})
    return pred_set


# # 대입 및 단일화
# 대입은 변수:값 형식의 사전구조로 구현됨. (예: {x:1, y:x})
def subst(s, x):
    """x에 대입 s를 적용함.
    >>> subst({x: 42, y:0}, F(x) + y)
    (F(42) + 0)
    """
    if isinstance(x, list):
        return [subst(s, xi) for xi in x]
    elif isinstance(x, tuple):
        return tuple([subst(s, xi) for xi in x])
    elif not isinstance(x, Expr):
        return x
    elif is_var_symbol(x.op):
        return s.get(x, x)
    else:
        return Expr(x.op, *[subst(s, arg) for arg in x.args])


def unify(x, y, s={}):
    """x, y를 동일하게 만드는 대입을 찾아 리턴. 동일하게 만드는 대입이 없으면 None 리턴.
    x, y는 변수(예: Expr('x')), 상수, 리스트, Expr들이 가능함.
    >>> unify(x, 3, {})
    {x: 3}
    """
    if s is None:
        return None
    elif x == y:
        return s
    elif is_variable(x):
        return unify_var(x, y, s)
    elif is_variable(y):
        return unify_var(y, x, s)
    elif isinstance(x, Expr) and isinstance(y, Expr):
        return unify(x.args, y.args, unify(x.op, y.op, s))
    elif isinstance(x, str) or isinstance(y, str):
        return None
    elif is_sequence(x) and is_sequence(y) and len(x) == len(y):
        if not x:
            return s
        return unify(x[1:], y[1:], unify(x[0], y[0], s))
    else:
        return None


def unify_var(var, x, s):
    if var in s:
        return unify(s[var], x, s)
    elif x in s:
        return unify(var, s[x], s)
    elif occur_check(var, x, s):
        return None
    else:
        new_s = extend(s, var, x)
        cascade_substitution(new_s)
        return new_s


def occur_check(var, x, s):
    """x(또는 x에 대입 s를 적용한 결과)에 var가 존재하면 true를 리턴함."""
    if var == x:
        return True
    elif is_variable(x) and x in s:
        return occur_check(var, s[x], s)
    elif isinstance(x, Expr):
        return (occur_check(var, x.op, s) or
                occur_check(var, x.args, s))
    elif isinstance(x, (list, tuple)):
        return first(e for e in x if occur_check(var, e, s))
    else:
        return False


def cascade_substitution(s):
    """정규형에서도 올바른 단일자(unifier)를 리턴하도록 하기 위해 s에 연쇄적인 대입을 수행함.
    >>> s = {x: y, y: G(z)}
    >>> cascade_substitution(s)
    >>> s == {x: G(z), y: G(z)}
    True
    """
    for x in s:
        s[x] = subst(s, s.get(x))
        if isinstance(s.get(x), Expr) and not is_variable(s.get(x)):
            # 함수 항이 올바르게 업데이트되도록 다시 패싱함.
            s[x] = subst(s, s.get(x))


def extend(s, var, val):
    """dict s를 복사하고 var의 값을 val로 세팅하여 확장한 후 리턴"""
    try:  # Python 3.5 and later
        return eval('{**s, var: val}')
    except SyntaxError:  # Python 3.4
        s2 = s.copy()
        s2[var] = val
        return s2


def is_sequence(x):
    """x가 시퀀스인가?"""
    return isinstance(x, collections.abc.Sequence)


def first(iterable, default=None):
    """첫번째 원소 리턴"""
    return next(iter(iterable), default)
