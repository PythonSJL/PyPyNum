from .trees import BTNode, BinaryTree
from .types import Union

__OPERATORS = {"**", "*", "//", "/", "%", "+", "-"}
__BASIC = "%()*+-./0123456789"
__ENGLISH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
__GREEK = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
__VALID = set(__BASIC + __ENGLISH + __GREEK)
_PRIORITY = {"**": 4, "*": 3, "//": 3, "/": 3, "%": 3, "+": 2, "-": 2}
_ASSOCIATIVITY = {"**": 0, "*": 1, "//": 1, "/": 1, "%": 1, "+": 1, "-": 1}


def tokenize(expr: str) -> list:
    def identify(e):
        if e and e not in __OPERATORS:
            if e.count(".") <= 1:
                if e.replace(".", "").isdigit():
                    if expr[index] == "(":
                        raise ValueError("Syntax error in expression")
                    result.append((e, number))
                else:
                    result.append((e, function if expr[index] == "(" else symbol))
            else:
                result.append((e, function if expr[index] == "(" else symbol))

    expr = expr.replace(" ", "")
    if any([item not in __VALID for item in expr]):
        raise ValueError("Characters other than Arabic numerals, English letters, Greek letters, operators, "
                         "decimal points, and parentheses cannot appear in expressions")
    depth = 0
    pointer = 0
    result = []
    bracket = "B"
    function = "F"
    number = "N"
    operator = "O"
    symbol = "S"
    unary = "U"
    for index in range(len(expr)):
        char = expr[index]
        if char in "()" or char in __OPERATORS:
            unknown = expr[pointer:index]
            if unknown:
                identify(unknown)
            pointer = index + 1
            if char == "(":
                result.append(("(", bracket))
                depth += 1
            elif char == ")":
                if depth == 0:
                    raise ValueError("The parentheses in the expression are not paired")
                depth -= 1
                result.append((")", bracket))
            else:
                is_unary = False
                if char in "+-":
                    if index == 0 or expr[index - 1] == "(" or expr[index - 1] in __OPERATORS:
                        is_unary = True
                if char in "*/" and index > 0 and char == expr[index - 1]:
                    if result and result[-1][-1] == operator:
                        if char == result[-1][0]:
                            result[-1] = (char * 2, operator)
                        else:
                            raise ValueError("Syntax error in expression")
                elif index != 0:
                    if result and result[-1][-1] == operator and not is_unary:
                        raise ValueError("Syntax error in expression")
                    op_type = unary if is_unary else operator
                    result.append((char, op_type))
                elif index == 0:
                    result.append((char, unary if char in "+-" else operator))
    unknown = expr[pointer:]
    if unknown:
        identify(unknown)
    if depth != 0:
        raise ValueError("The parentheses in the expression are not paired")
    return result


def infix2postfix(infix_expr: list) -> list:
    postfix = []
    op_stack = []
    for token, t_type in infix_expr:
        if t_type == "B":
            if token == "(":
                op_stack.append((token, t_type))
            elif token == ")":
                while op_stack and op_stack[-1][0] != "(":
                    postfix.append(op_stack.pop())
                if op_stack and op_stack[-1][0] == "(":
                    op_stack.pop()
                if op_stack and op_stack[-1][1] in ("F", "U"):
                    postfix.append(op_stack.pop())
        elif t_type == "O":
            is_left_assoc = True
            current_prio = _PRIORITY[token]
            while op_stack and op_stack[-1][0] != "(":
                top_token, top_type = op_stack[-1]
                if top_type == "U":
                    postfix.append(op_stack.pop())
                    continue
                if top_type == "F":
                    break
                top_prio = _PRIORITY.get(top_token, 0)
                pop_condition = current_prio < top_prio or (is_left_assoc and current_prio == top_prio)
                if pop_condition:
                    postfix.append(op_stack.pop())
                else:
                    break
            op_stack.append((token, t_type))
        elif t_type in ("F", "U"):
            op_stack.append((token, t_type))
        else:
            postfix.append((token, t_type))
    while op_stack:
        postfix.append(op_stack.pop())
    return postfix


def build_expr_tree(postfix_expr: list) -> BinaryTree:
    node_stack = []
    for token, t_type in postfix_expr:
        new_node = BTNode((token, t_type))
        if t_type == "O":
            right_child = node_stack.pop() if node_stack else None
            left_child = node_stack.pop() if node_stack else None
            new_node.add_right(right_child)
            new_node.add_left(left_child)
            node_stack.append(new_node)
        elif t_type in ("F", "U"):
            arg_child = node_stack.pop() if node_stack else None
            new_node.add_right(arg_child)
            node_stack.append(new_node)
        else:
            node_stack.append(new_node)
    return BinaryTree(node_stack.pop()) if node_stack else None


class Expr:
    def __init__(self, tree: Union[str, BinaryTree] = None):
        self.tree = tree if tree is None or isinstance(tree, BinaryTree) else build_expr_tree(
            infix2postfix(tokenize(tree)))

    def __repr__(self) -> str:
        return "Expr(tree={})".format(self.tree)

    def __str__(self) -> str:
        return "Expr(empty)" if self.is_empty() else self.to_string()

    def is_empty(self) -> bool:
        return self.tree is None

    def traverse(self, method: str = "inorder") -> list:
        if self.tree is None:
            return []
        return self.tree.traverse(method)

    def to_string(self) -> str:
        if self.is_empty():
            return ""

        def dfs(node):
            if node is None:
                return ""
            token, t_type = node.data
            if t_type in ("N", "S"):
                return token
            if t_type == "F":
                arg_str = dfs(node.right)
                return "{}({})".format(token, arg_str)
            if t_type == "U":
                arg_str = dfs(node.right)
                if node.right and node.right.data[1] in ("O", "U"):
                    return "{}({})".format(token, arg_str)
                return token + arg_str
            if t_type == "O":
                left_str = dfs(node.left)
                right_str = dfs(node.right)
                p_curr = _PRIORITY.get(token, -1)
                final_left = left_str
                if node.left and node.left.data[1] == "O":
                    p_left = _PRIORITY.get(node.left.data[0], -1)
                    if p_left < p_curr or p_left == p_curr and _ASSOCIATIVITY[token] == 0:
                        final_left = "({})".format(left_str)
                final_right = right_str
                if node.right and node.right.data[1] == "O":
                    p_right = _PRIORITY.get(node.right.data[0], -1)
                    if p_right < p_curr or p_right == p_curr and _ASSOCIATIVITY[token] == 1:
                        final_right = "({})".format(right_str)
                return "".join((final_left, token, final_right))
            return token

        return dfs(self.tree.root)

    def __deep_copy_node(self, node: Union[BTNode, None]) -> Union[BTNode, None]:
        if node is None:
            return None
        new_node = BTNode(node.data)
        # Get copies of children
        left_child = self.__deep_copy_node(node.left)
        right_child = self.__deep_copy_node(node.right)
        # Add children only if they are not None to avoid errors in trees.py
        if left_child is not None:
            new_node.add_left(left_child)
        if right_child is not None:
            new_node.add_right(right_child)
        return new_node

    def __binary_op(self, other, op_token):
        other_expr = other if isinstance(other, Expr) else parse_expr(str(other))
        if self.is_empty():
            return other_expr
        if other_expr.is_empty():
            return self
        left_copy = self.__deep_copy_node(self.tree.root)
        right_copy = self.__deep_copy_node(other_expr.tree.root)
        new_node = BTNode((op_token, "O"))
        new_node.add_left(left_copy)
        new_node.add_right(right_copy)
        return Expr(BinaryTree(new_node))

    def __rbinary_op(self, other, op_token):
        other_expr = other if isinstance(other, Expr) else parse_expr(str(other))
        if self.is_empty():
            return other_expr
        if other_expr.is_empty():
            return self
        left_copy = self.__deep_copy_node(other_expr.tree.root)
        right_copy = self.__deep_copy_node(self.tree.root)
        new_node = BTNode((op_token, "O"))
        new_node.add_left(left_copy)
        new_node.add_right(right_copy)
        return Expr(BinaryTree(new_node))

    def __unary_op(self, op_token):
        if self.is_empty():
            return self
        child_copy = self.__deep_copy_node(self.tree.root)
        new_node = BTNode((op_token, "U"))
        new_node.add_right(child_copy)
        return Expr(BinaryTree(new_node))

    def __add__(self, other):
        return self.__binary_op(other, "+")

    def __radd__(self, other):
        return self.__binary_op(other, "+")

    def __sub__(self, other):
        return self.__binary_op(other, "-")

    def __rsub__(self, other):
        return self.__rbinary_op(other, "-")

    def __mul__(self, other):
        return self.__binary_op(other, "*")

    def __rmul__(self, other):
        return self.__binary_op(other, "*")

    def __truediv__(self, other):
        return self.__binary_op(other, "/")

    def __rtruediv__(self, other):
        return self.__rbinary_op(other, "/")

    def __floordiv__(self, other):
        return self.__binary_op(other, "//")

    def __rfloordiv__(self, other):
        return self.__rbinary_op(other, "//")

    def __mod__(self, other):
        return self.__binary_op(other, "%")

    def __rmod__(self, other):
        return self.__rbinary_op(other, "%")

    def __pow__(self, other):
        return self.__binary_op(other, "**")

    def __rpow__(self, other):
        return self.__rbinary_op(other, "**")

    def __neg__(self):
        return self.__unary_op("-")

    def __pos__(self):
        return self.__unary_op("+")


def parse_expr(expr: str) -> Expr:
    return Expr(build_expr_tree(infix2postfix(tokenize(expr))))
