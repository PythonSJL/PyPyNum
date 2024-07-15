class MultiTreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None

    def add_child(self, node):
        if node.parent:
            node.parent.remove_child(node)
        node.parent = self
        self.children.append(node)

    def remove_child(self, node):
        if node in self.children:
            node.parent = None
            self.children.remove(node)

    def find_node(self, data):
        if self.data == data:
            return self
        for child in self.children:
            result = child.find_node(data)
            if result:
                return result
        return None

    def traverse(self, traversal_type="preorder"):
        if traversal_type == "preorder":
            result = [self.data]
            for child in self.children:
                result.extend(child.traverse(traversal_type))
            return result
        elif traversal_type == "postorder":
            result = []
            for child in self.children:
                result.extend(child.traverse(traversal_type))
            result.append(self.data)
            return result
        elif traversal_type == "levelorder":
            result = []
            queue = [self]
            while queue:
                current_node = queue.pop(0)
                result.append(current_node.data)
                queue.extend(current_node.children)
            return result
        else:
            raise ValueError("Invalid traversal type")

    def to_list(self):
        node_list = [self.data]
        if self.children:
            node_list.extend([child.to_list() for child in self.children])
        return node_list

    def print_tree(self, node=None, level=0, prefix="", use_repr=True):
        if node is None:
            node = self
            if node is None:
                print("The tree is empty")
                return
            prefix = "Root\n└── "
        print("    " * level + prefix + (repr if use_repr else str)(node.data))
        children_prefix = "│   " if node.children else ""  # TODO 前缀暂未正确处理
        for i, child in enumerate(node.children):
            child_prefix = "├── " if i < len(node.children) - 1 else "└── "
            child.print_tree(child, level + 1, child_prefix, use_repr)

    def __str__(self):
        return str(self.data)


class MultiTree:
    def __init__(self, root=None):
        self.root = root if root is None or isinstance(root, MultiTreeNode) else MultiTreeNode(root)

    def add_node(self, data, parent=None):
        new_node = data if isinstance(data, MultiTreeNode) else MultiTreeNode(data)
        if parent is None:
            if self.root is None:
                self.root = new_node
            else:
                raise ValueError("Root already exists")
        else:
            if parent not in self.root.children and parent != self.root:
                raise ValueError("Parent node is not part of the tree")
            new_node.parent = parent
            parent.add_child(new_node)
        return new_node

    def remove_node(self, data):
        def remove_helper(node, target_data):
            if node.data == target_data:
                parent = node.parent
                if parent:
                    parent.remove_child(node)
                return True
            for child in node.children:
                if remove_helper(child, target_data):
                    return True
            return False

        if not self.root:
            raise ValueError("Tree is empty")
        if remove_helper(self.root, data):
            if not self.root.children:
                self.root = None
            return True
        return False

    def find_node(self, data):
        if self.root:
            return self.root.find_node(data)
        return None

    def traverse(self, traversal_type="preorder"):
        if self.root:
            return self.root.traverse(traversal_type)
        else:
            raise ValueError("Tree is empty")

    def to_list(self):
        return self.root.to_list()

    def print_tree(self, use_repr=True):
        self.root.print_tree(use_repr=use_repr)
