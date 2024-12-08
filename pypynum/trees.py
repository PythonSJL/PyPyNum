class RBTNode:
    def __init__(self, data, color="red"):
        self.data = data
        self.color = color
        self.parent = None
        self.left = None
        self.right = None

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.data)


class RedBlackTree:
    def __init__(self):
        self.nil = RBTNode(None, "black")
        self.root = self.nil

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.root)

    def _fix_delete(self, x):
        while x != self.root and x.color == "black":
            if x == x.parent.left:
                s = x.parent.right
                if s.color == "red":
                    s.color = "black"
                    x.parent.color = "red"
                    self._left_rotate(x.parent)
                    s = x.parent.right
                if s.left.color == "black" and s.right.color == "black":
                    s.color = "red"
                    x = x.parent
                else:
                    if s.right.color == "black":
                        s.left.color = "black"
                        s.color = "red"
                        self._right_rotate(s)
                        s = x.parent.right
                    s.color = x.parent.color
                    x.parent.color = "black"
                    s.right.color = "black"
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                s = x.parent.left
                if s.color == "red":
                    s.color = "black"
                    x.parent.color = "red"
                    self._right_rotate(x.parent)
                    s = x.parent.left
                if s.right.color == "black" and s.left.color == "black":
                    s.color = "red"
                    x = x.parent
                else:
                    if s.left.color == "black":
                        s.right.color = "black"
                        s.color = "red"
                        self._left_rotate(s)
                        s = x.parent.left
                    s.color = x.parent.color
                    x.parent.color = "black"
                    s.left.color = "black"
                    self._right_rotate(x.parent)
                    x = self.root
        x.color = "black"

    def _fix_insert(self, node):
        while node != self.root and node.parent.color == "red":
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == "red":
                    node.parent.color = "black"
                    uncle.color = "black"
                    node.parent.parent.color = "red"
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self._left_rotate(node)
                    node.parent.color = "black"
                    node.parent.parent.color = "red"
                    self._right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == "red":
                    node.parent.color = "black"
                    uncle.color = "black"
                    node.parent.parent.color = "red"
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)
                    node.parent.color = "black"
                    node.parent.parent.color = "red"
                    self._left_rotate(node.parent.parent)
        self.root.color = "black"

    def _inorder_helper(self, node, result):
        stack = []
        current = node
        while stack or current != self.nil:
            if current != self.nil:
                stack.append(current)
                current = current.left
            else:
                current = stack.pop()
                result.append(current.data)
                current = current.right

    def _left_rotate(self, x):
        if x == self.nil:
            raise ValueError("Cannot rotate a nil node.")
        y = x.right
        x.right = y.left
        if y.left != self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _levelorder_helper(self):
        result = []
        if self.root == self.nil:
            return result
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            result.append(node.data)
            if node.left != self.nil:
                queue.append(node.left)
            if node.right != self.nil:
                queue.append(node.right)
        return result

    def _postorder_helper(self, node, result):
        stack = []
        last_visited = self.nil
        current = node
        while stack or current != self.nil:
            if current != self.nil:
                stack.append(current)
                current = current.left
            else:
                peek_node = stack[-1]
                if peek_node.right != self.nil and last_visited != peek_node.right:
                    current = peek_node.right
                else:
                    result.append(peek_node.data)
                    last_visited = stack.pop()

    def _preorder_helper(self, node, result):
        stack = [node]
        while stack:
            current = stack.pop()
            if current != self.nil:
                result.append(current.data)
                stack.append(current.right)
                stack.append(current.left)

    def _print_tree_helper(self, node, indent, is_left):
        result = ""
        if node != self.nil:
            result += self._print_tree_helper(node.right, indent + ("│   " if is_left else "    "), False)
            result += indent + ("└── " if is_left else "┌── ") + node.data.__str__().__repr__()[1:-1] + (
                "(R)" if node.color == "red" else "(B)") + "\n"
            result += self._print_tree_helper(node.left, indent + ("    " if is_left else "│   "), True)
        return result

    def _range_search_helper(self, node, low, high, result):
        if node == self.nil:
            return
        if low < node.data:
            self._range_search_helper(node.left, low, high, result)
        if low <= node.data <= high:
            result.append(node.data)
        if high > node.data:
            self._range_search_helper(node.right, low, high, result)

    def _right_rotate(self, x):
        if x == self.nil:
            raise ValueError("Cannot rotate a nil node.")
        y = x.left
        x.left = y.right
        if y.right != self.nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def _transplant(self, u, v):
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def black_height(self, node=None):
        if node is None:
            node = self.root
        if node == self.nil:
            return 0
        left_height = self.black_height(node.left)
        right_height = self.black_height(node.right)
        if left_height != right_height:
            return -1
        return left_height + (1 if node.color == "black" else 0)

    def count_nodes(self, node=None):
        if node is None:
            node = self.root
        count = 0
        stack = [node]
        while stack:
            current = stack.pop()
            if current != self.nil:
                count += 1
                stack.append(current.left)
                stack.append(current.right)
        return count

    def delete(self, data):
        node_to_delete = self.root
        while node_to_delete != self.nil:
            if node_to_delete.data == data:
                break
            elif data < node_to_delete.data:
                node_to_delete = node_to_delete.left
            else:
                node_to_delete = node_to_delete.right
        if node_to_delete == self.nil:
            raise ValueError("Node to delete not found in the tree.")
        original_color = node_to_delete.color
        if node_to_delete.left == self.nil:
            x = node_to_delete.right
            self._transplant(node_to_delete, node_to_delete.right)
        elif node_to_delete.right == self.nil:
            x = node_to_delete.left
            self._transplant(node_to_delete, node_to_delete.left)
        else:
            successor = self.minimum()
            original_color = successor.color
            x = successor.right
            if successor.parent == node_to_delete:
                x.parent = successor
            else:
                self._transplant(successor, successor.right)
                successor.right = node_to_delete.right
                successor.right.parent = successor
            self._transplant(node_to_delete, successor)
            successor.left = node_to_delete.left
            successor.left.parent = successor
            successor.color = node_to_delete.color
        if original_color == "black":
            self._fix_delete(x)

    def get_depth(self, node):
        if node is None:
            return 0
        depth = 0
        while node != self.root and node != self.nil:
            node = node.parent
            depth += 1
        return depth

    def get_height(self, node):
        if node == self.nil:
            return 0
        max_height = 0
        stack = [(node, 1)]
        while stack:
            current, height = stack.pop()
            if current != self.nil:
                max_height = max(max_height, height)
                stack.append((current.left, height + 1))
                stack.append((current.right, height + 1))
        return max_height

    def get_max(self):
        return self.maximum().data

    def get_min(self):
        return self.minimum().data

    def insert(self, data):
        new_node = RBTNode(data)
        new_node.left = self.nil
        new_node.right = self.nil
        parent = None
        current = self.root
        while current != self.nil:
            parent = current
            if new_node.data < current.data:
                current = current.left
            else:
                current = current.right
        new_node.parent = parent
        if parent is None:
            self.root = new_node
        elif new_node.data < parent.data:
            parent.left = new_node
        else:
            parent.right = new_node
        new_node.color = "red"
        self._fix_insert(new_node)

    def is_red_black_tree(self):
        def is_red(node):
            return node.color == "red"

        def check_properties(node):
            if node == self.nil:
                return True, 1
            if is_red(node):
                if is_red(node.left) or is_red(node.right):
                    return False, 0
            left_valid, left_black_height = check_properties(node.left)
            right_valid, right_black_height = check_properties(node.right)
            return left_valid and right_valid and left_black_height == right_black_height, left_black_height + (
                1 if node.color == "black" else 0)

        return check_properties(self.root)[0]

    def lowest_common_ancestor(self, node1, node2):
        def lca_helper(root, a, b):
            if root == self.nil or root == a or root == b:
                return root
            left = lca_helper(root.left, a, b)
            right = lca_helper(root.right, a, b)
            if left is not None and right is not None:
                return root
            return left if left else right

        return lca_helper(self.root, node1, node2)

    def maximum(self):
        node = self.root
        while node.right != self.nil:
            node = node.right
        return node

    def minimum(self):
        node = self.root
        while node.left != self.nil:
            node = node.left
        return node

    def node_height(self, node):
        if node == self.nil:
            return -1
        max_height = -1
        stack = [(node, 0)]
        while stack:
            current, height = stack.pop()
            if current != self.nil:
                max_height = max(max_height, height)
                stack.append((current.left, height + 1))
                stack.append((current.right, height + 1))
        return max_height

    def node_rank(self, data):
        if self.search(data) is None:
            return -1
        rank = 0
        node = self.root
        while node != self.nil:
            if data < node.data:
                node = node.left
            elif data > node.data:
                rank += 1 + self.count_nodes(node.left)
                node = node.right
            else:
                rank += self.count_nodes(node.left)
                break
        return rank

    def path_length(self, data):
        length = 0
        node = self.root
        while node != self.nil:
            length += 1
            if data == node.data:
                return length
            elif data < node.data:
                node = node.left
            else:
                node = node.right
        return -1

    def path_to_node(self, data):
        path = []
        node = self.root
        while node != self.nil:
            path.append(node.data)
            if data == node.data:
                return path
            elif data < node.data:
                node = node.left
            else:
                node = node.right

    def predecessor(self, node):
        if node == self.nil:
            return None
        if node.left != self.nil:
            node = node.left
            while node.right != self.nil:
                node = node.right
            return node
        while node.parent != self.nil and node == node.parent.left:
            node = node.parent
        return node.parent

    def print_tree(self, to_console=True):
        tree_str = self._print_tree_helper(self.root, "", True)
        if to_console:
            print(tree_str)
        else:
            return tree_str

    def range_search(self, low, high):
        if low > high:
            raise ValueError("Low value must be less than or equal to high value.")
        result = []
        self._range_search_helper(self.root, low, high, result)
        return result

    def search(self, data):
        current = self.root
        while current != self.nil:
            if current.data == data:
                return current
            elif data < current.data:
                current = current.left
            else:
                current = current.right

    def subtree_maximum(self, node):
        while node.right != self.nil:
            node = node.right
        return node

    def subtree_minimum(self, node):
        while node.left != self.nil:
            node = node.left
        return node

    def subtree_node_count(self, node):
        if node == self.nil:
            return 0
        count = 0
        stack = [node]
        while stack:
            current = stack.pop()
            if current != self.nil:
                count += 1
                stack.append(current.left)
                stack.append(current.right)
        return count

    def successor(self, node):
        if node == self.nil:
            return None
        if node.right != self.nil:
            node = node.right
            while node.left != self.nil:
                node = node.left
            return node
        while node.parent != self.nil and node == node.parent.right:
            node = node.parent
        return node.parent

    def traverse(self, traversal_type="preorder"):
        result = []
        if traversal_type == "preorder":
            self._preorder_helper(self.root, result)
        elif traversal_type == "inorder":
            self._inorder_helper(self.root, result)
        elif traversal_type == "postorder":
            self._postorder_helper(self.root, result)
        elif traversal_type == "levelorder":
            self._levelorder_helper()
        else:
            raise ValueError("Invalid traversal type. Choose from 'preorder', 'inorder', 'postorder', 'levelorder'.")
        return result

    def tree_height(self):
        return self.node_height(self.root)

    def update(self, old_data, new_data):
        self.delete(old_data)
        self.insert(new_data)


class MTNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.data)

    def add_child(self, node):
        if node.parent is not None:
            node.parent.remove_child(node)
        node.parent = self
        self.children.append(node)

    def remove_child(self, node):
        if node in self.children:
            node.parent = None
            self.children.remove(node)


class MultiTree:
    def __init__(self, root=None):
        self.root = root if root is None or isinstance(root, MTNode) else MTNode(root)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.root)

    def add_node(self, data, parent=None):
        new_node = data if isinstance(data, MTNode) else MTNode(data)
        if parent is None:
            if self.root is None:
                self.root = new_node
            else:
                raise ValueError("Root node already exists. Cannot add another root.")
        else:
            new_node.parent = parent
            parent.add_child(new_node)
        return new_node

    def apply_function(self, func):
        if self.root is None:
            return
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            node.data = func(node.data)
            queue.extend(node.children)

    def clone_subtree(self, data):
        node = self.find_node(data)
        if node is None:
            return None
        cloned_root = MTNode(node.data)
        stack = [(node, cloned_root)]
        while stack:
            original, cloned = stack.pop()
            for child in original.children:
                cloned_child = MTNode(child.data)
                cloned_child.parent = cloned
                cloned.children.append(cloned_child)
                stack.append((child, cloned_child))
        return cloned_root

    def count_nodes(self):
        if self.root is None:
            return 0
        count = 0
        queue = [self.root]
        while queue:
            current_node = queue.pop(0)
            count += 1
            queue.extend(current_node.children)
        return count

    def find_leaves(self):
        if self.root is None:
            return []
        leaves = []
        queue = [self.root]
        while queue:
            current_node = queue.pop(0)
            if current_node.children is None:
                leaves.append(current_node.data)
            queue.extend(current_node.children)
        return leaves

    def find_node(self, data):
        def find_helper(node, target_data):
            if node.data == target_data:
                return node
            for child in node.children:
                result = find_helper(child, target_data)
                if result is not None:
                    return result

        if self.root is not None:
            return find_helper(self.root, data)

    def find_path(self, data):
        if self.root is None:
            return None
        path = []
        stack = [(self.root, path)]
        while stack:
            current_node, current_path = stack.pop()
            if current_node is None:
                continue
            new_path = current_path + [current_node.data]
            if current_node.data == data:
                return new_path
            for child in current_node.children:
                stack.append((child, new_path))

    def is_balanced(self):
        def check_height(node):
            if node is None:
                return 0
            children_heights = tuple(map(check_height, node.children))
            if children_heights is None:
                return 0
            if max(children_heights) - min(children_heights) > 1:
                return -1
            return 1 + max(children_heights)

        return check_height(self.root) != -1

    def node_depth(self, data):
        if self.root is None:
            return -1
        queue = [(self.root, 0)]
        while queue:
            current_node, current_depth = queue.pop(0)
            if current_node.data == data:
                return current_depth
            for child in current_node.children:
                queue.append((child, current_depth + 1))
        return -1

    def node_height(self, data):
        node = self.find_node(data)
        if node is None:
            return -1
        max_height = 0
        stack = [(node, 0)]
        while stack:
            current_node, current_height = stack.pop()
            if current_node.children is None:
                max_height = max(max_height, current_height)
            for child in current_node.children:
                stack.append((child, current_height + 1))
        return max_height

    def print_tree(self, to_console=True):
        def string(obj):
            return obj.__str__().__repr__()[1:-1]

        def print_tree_helper(node, indent, is_last):
            if node is None:
                return ""
            line = indent
            if is_last:
                line += "└── " + string(node.data)
                indent += "    "
            else:
                line += "├── " + string(node.data)
                indent += "│   "
            result = line + "\n"
            num_children = len(node.children)
            for i, child in enumerate(node.children):
                result += print_tree_helper(child, indent, i == num_children - 1)
            return result

        tree_str = print_tree_helper(self.root, "", True)
        if to_console:
            print(tree_str)
        else:
            return tree_str

    def prune_tree(self, predicate):
        if self.root is None:
            return
        queue = [(self.root, None)]
        while queue:
            node, parent = queue.pop(0)
            if predicate(node.data):
                if parent is not None:
                    parent.children.remove(node)
            else:
                queue.extend([(child, node) for child in node.children])

    def remove_node(self, data):
        def remove_helper(node, target_data):
            if node.data == target_data:
                if node == self.root:
                    self.root = None
                else:
                    node.parent.remove_child(node)
                return True
            for child in node.children:
                if remove_helper(child, target_data):
                    return True
            return False

        if self.root is None:
            raise ValueError("Tree is empty. Cannot remove node.")
        return remove_helper(self.root, data)

    def replace_node(self, old_data, new_data):
        node = self.find_node(old_data)
        if node is None:
            return False
        new_node = MTNode(new_data)
        new_node.children = node.children
        new_node.parent = node.parent
        if node.parent is not None:
            node.parent.children[node.parent.children.index(node)] = new_node
        for child in node.children:
            child.parent = new_node
        if node == self.root:
            self.root = new_node
        return True

    def search_nodes(self, predicate):
        if self.root is None:
            return []
        queue = [self.root]
        matching_nodes = []
        while queue:
            node = queue.pop(0)
            if predicate(node.data):
                matching_nodes.append(node)
            queue.extend(node.children)
        return matching_nodes

    def sort_tree(self, key_func=None):
        def default(x):
            return x

        if key_func is None:
            key_func = default
        elif not callable(key_func):
            raise ValueError("The 'key_func' parameter must be a callable function that takes a node's data as input "
                             "and returns a comparable value.")
        if self.root is None:
            return
        queue = [(self.root, 0)]
        while queue:
            node, level = queue.pop(0)
            node.children.sort(key=lambda child: key_func(child.data))
            queue.extend([(child, level + 1) for child in node.children])

    def to_list(self):
        def to_list_helper(node):
            node_list = [node.data]
            if node.children is not None:
                node_list.extend([to_list_helper(child) for child in node.children])
            return node_list

        if self.root is not None:
            return to_list_helper(self.root)
        return []

    def traverse(self, traversal_type="preorder"):
        def preorder(node):
            result = [node.data]
            for child in node.children:
                result.extend(preorder(child))
            return result

        def postorder(node):
            result = []
            for child in node.children:
                result.extend(postorder(child))
            result.append(node.data)
            return result

        def levelorder(node):
            result = []
            queue = [node]
            while queue:
                current_node = queue.pop(0)
                result.append(current_node.data)
                queue.extend(current_node.children)
            return result

        if self.root is None:
            raise ValueError("Tree is empty. Cannot perform traversal.")
        if traversal_type == "preorder":
            return preorder(self.root)
        elif traversal_type == "postorder":
            return postorder(self.root)
        elif traversal_type == "levelorder":
            return levelorder(self.root)
        else:
            raise ValueError("Invalid traversal type. Choose from 'preorder', 'postorder', 'levelorder'.")


class BTNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.parent = None

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.data)

    def add_left(self, node):
        if node.parent is not None:
            node.parent.remove_child(node)
        node.parent = self
        self.left = node

    def add_right(self, node):
        if node.parent is not None:
            node.parent.remove_child(node)
        node.parent = self
        self.right = node

    def remove_child(self, node):
        if self.left == node:
            self.left = None
        elif self.right == node:
            self.right = None
        if node.parent == self:
            node.parent = None


class BinaryTree:
    def __init__(self, root=None):
        self.root = root if root is None or isinstance(root, BTNode) else BTNode(root)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.root)

    def add_left(self, parent, data):
        if parent is None:
            if self.root is None:
                self.root = BTNode(data)
                return self.root
            else:
                raise ValueError("Root node already exists. Cannot add another root.")
        elif not isinstance(parent, BTNode):
            raise ValueError("Parent must be a BTNode instance")
        new_node = BTNode(data)
        if parent.left is not None:
            parent.left.remove_child(parent.left)
        parent.left = new_node
        new_node.parent = parent
        return new_node

    def add_right(self, parent, data):
        if parent is None:
            if self.root is None:
                self.root = BTNode(data)
                return self.root
            else:
                raise ValueError("Root node already exists. Cannot add another root.")
        elif not isinstance(parent, BTNode):
            raise ValueError("Parent must be a BTNode instance")
        new_node = BTNode(data)
        if parent.right is not None:
            parent.right.remove_child(parent.right)
        parent.right = new_node
        new_node.parent = parent
        return new_node

    def apply_function(self, func):
        def apply_helper(node):
            if node is None:
                return
            node.data = func(node.data)
            apply_helper(node.left)
            apply_helper(node.right)

        apply_helper(self.root)

    def clone_subtree(self, data):
        node = self.find_node(data)
        if node is None:
            return None
        return self._clone_node(node)

    def _clone_node(self, node):
        if node is None:
            return None
        cloned_node = BTNode(node.data)
        cloned_node.left = self._clone_node(node.left)
        cloned_node.right = self._clone_node(node.right)
        if cloned_node.left is not None:
            cloned_node.left.parent = cloned_node
        if cloned_node.right is not None:
            cloned_node.right.parent = cloned_node
        return cloned_node

    def count_nodes(self):
        def count_helper(node):
            if node is None:
                return 0
            left_count = count_helper(node.left)
            right_count = count_helper(node.right)
            return 1 + left_count + right_count

        return count_helper(self.root)

    def find_leaves(self):
        def leaves_helper(node):
            if node is None:
                return []
            if node.left is None and node.right is None:
                return [node.data]
            return leaves_helper(node.left) + leaves_helper(node.right)

        return leaves_helper(self.root)

    def find_node(self, data):
        def find_helper(node):
            if node is None:
                return None
            if node.data == data:
                return node
            left_result = find_helper(node.left)
            if left_result is not None:
                return left_result
            return find_helper(node.right)

        return find_helper(self.root)

    def find_path(self, data):
        def path_helper(node, target_data, path):
            if node is None:
                return None
            new_path = path + [node.data]
            if node.data == target_data:
                return new_path
            left_result = path_helper(node.left, target_data, new_path)
            if left_result is not None:
                return left_result
            return path_helper(node.right, target_data, new_path)

        return path_helper(self.root, data, [])

    def is_balanced(self):
        def balanced_helper(node):
            if node is None:
                return 0, True
            left_height, left_balanced = balanced_helper(node.left)
            right_height, right_balanced = balanced_helper(node.right)
            return 1 + max(left_height, right_height), left_balanced and right_balanced and abs(
                left_height - right_height) <= 1

        return balanced_helper(self.root)[1]

    def node_depth(self, data):
        def depth_helper(node, target_data, depth):
            if node is None:
                return -1
            if node.data == target_data:
                return depth
            left_result = depth_helper(node.left, target_data, depth + 1)
            if left_result != -1:
                return left_result
            return depth_helper(node.right, target_data, depth + 1)

        return depth_helper(self.root, data, 0)

    def node_height(self, data):
        def height_helper(node):
            if node is None:
                return 0
            return 1 + max(height_helper(node.left), height_helper(node.right))

        result = self.find_node(data)
        if result is None:
            return -1
        return height_helper(result)

    def print_tree(self, to_console=True):
        def print_tree_helper(node, indent, is_last):
            if node is None:
                return ""
            line = indent
            if is_last:
                line += "└── " + str(node.data)
                indent += "    "
            else:
                line += "├── " + str(node.data)
                indent += "│   "
            result = line + "\n"
            if node.left is not None:
                result += print_tree_helper(node.left, indent, not node.right)
            if node.right is not None:
                result += print_tree_helper(node.right, indent, True)
            return result

        tree_str = print_tree_helper(self.root, "", True)
        if to_console:
            print(tree_str)
        else:
            return tree_str

    def prune_tree(self, predicate):
        def prune_helper(node, parent):
            if node is None:
                return
            if predicate(node.data):
                if parent is not None:
                    if parent.left == node:
                        parent.left = None
                    elif parent.right == node:
                        parent.right = None
                return
            prune_helper(node.left, node)
            prune_helper(node.right, node)

        prune_helper(self.root, None)

    def remove_node(self, data):
        def remove_helper(node, target_data, parent):
            if node is None:
                return False
            if node.data == target_data:
                if parent is not None:
                    if parent.left == node:
                        parent.left = None
                    elif parent.right == node:
                        parent.right = None
                return True
            if remove_helper(node.left, target_data, node):
                return True
            return remove_helper(node.right, target_data, node)

        if self.root is None:
            raise ValueError("Tree is empty. Cannot remove node.")
        if remove_helper(self.root, data, None):
            if self.root.left is None and self.root.right is None:
                self.root = None
            return True
        return False

    def replace_node(self, old_data, new_data):
        node = self.find_node(old_data)
        if node is None:
            return False
        new_node = BTNode(new_data)
        new_node.left = node.left
        new_node.right = node.right
        if node.left is not None:
            node.left.parent = new_node
        if node.right is not None:
            node.right.parent = new_node
        if node.parent is not None:
            if node.parent.left == node:
                node.parent.left = new_node
            elif node.parent.right == node:
                node.parent.right = new_node
        else:
            self.root = new_node
        new_node.parent = node.parent
        return True

    def search_nodes(self, predicate):
        def search_helper(node):
            if node is None:
                return []
            result = []
            if predicate(node.data):
                result.append(node)
            result.extend(search_helper(node.left))
            result.extend(search_helper(node.right))
            return result

        return search_helper(self.root)

    def sort_tree(self, key_func=None):
        def default(x):
            return x

        def sort_helper(node):
            if node is None:
                return
            sort_helper(node.left)
            sort_helper(node.right)
            if node.left and key_func(node.left.data) > key_func(node.data):
                node.left, node.data = node.data, node.left.data
            if node.right and key_func(node.right.data) < key_func(node.data):
                node.right, node.data = node.data, node.right.data

        if key_func is None:
            key_func = default
        elif not callable(key_func):
            raise ValueError("The 'key_func' parameter must be a callable function that takes a node's data as input "
                             "and returns a comparable value.")
        if self.root is None:
            return
        sort_helper(self.root)

    def to_list(self):
        def to_list_helper(node):
            if node is None:
                return []
            return [node.data, to_list_helper(node.left), to_list_helper(node.right)]

        return to_list_helper(self.root)

    def traverse(self, traversal_type="inorder"):
        def inorder(node):
            if node is None:
                return []
            return inorder(node.left) + [node.data] + inorder(node.right)

        def preorder(node):
            if node is None:
                return []
            return [node.data] + preorder(node.left) + preorder(node.right)

        def postorder(node):
            if node is None:
                return []
            return postorder(node.left) + postorder(node.right) + [node.data]

        def levelorder(node):
            if node is None:
                return []
            result = []
            queue = [node]
            while queue:
                current_node = queue.pop(0)
                result.append(current_node.data)
                if current_node.left is not None:
                    queue.append(current_node.left)
                if current_node.right is not None:
                    queue.append(current_node.right)
            return result

        if self.root is None:
            raise ValueError("Tree is empty. Cannot perform traversal.")
        if traversal_type == "inorder":
            return inorder(self.root)
        elif traversal_type == "preorder":
            return preorder(self.root)
        elif traversal_type == "postorder":
            return postorder(self.root)
        elif traversal_type == "levelorder":
            return levelorder(self.root)
        else:
            raise ValueError("Invalid traversal type. Choose from 'inorder', 'preorder', 'postorder', 'levelorder'.")
