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
            print(self.data)
            for child in self.children:
                child.traverse(traversal_type)
        elif traversal_type == "postorder":
            for child in self.children:
                child.traverse(traversal_type)
            print(self.data)
        else:
            raise ValueError("Invalid traversal type")

    def __str__(self):
        return str(self.data)


class MultiTree:
    def __init__(self, root=None):
        self.root = root if root is None else MultiTreeNode(root)

    def add_node(self, data, parent=None):
        new_node = MultiTreeNode(data)
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
            self.root.traverse(traversal_type)
        else:
            raise ValueError("Tree is empty")
