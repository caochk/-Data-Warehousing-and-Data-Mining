class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = [] #对象自带的属性，下面新建了两个Tree对象：tree、parent
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def print_tree(root, indent=0):
    print(' ' * indent, root) #先是打印的根，不会先打印children列表内的东西
    if len(root.children) > 0: #若children列表内有东西
        for child in root.children: #把children列表内的东西遍历一遍
            print_tree(child, indent+4) #4是固定间隔

def make_tree(tokens): # do not change the heading of the function
    root = Tree(tokens[0])
    node = root
    root_list = []
    i = 1
    while i < len(tokens):
        if tokens[i] == "[":
            root_list.append(node)
            i += 1
        elif tokens[i] == "]":
            root = root_list.pop()
            i += 1
        else:
            root = root_list[-1]
            node = Tree(tokens[i])
            root.add_child(node)
            i += 1
    return root

def max_depth(root):
    i = 1
    depth = a(root, i)
    return depth

def a(root, i):
    if len(root.children) > 0:
        i += 1
        for child in root.children:
            if len(child.children) > 0:
                i += 1
                a(child, i)
    return i


tt = make_tree(['1', '[', '2', '[', '3', '4', '5', ']', '6', '[', '7', '8', '[', '9', ']', '10', '[', '11', '12', ']', ']', '13', ']'])
print(max_depth(tt))