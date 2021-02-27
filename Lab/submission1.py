import math ## import modules here

################# Question 0 #################

def add(a, b): # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x): # do not change the heading of the function
    if x < 0:
        return -1

    else:
        left = 0
        right = x
        while left <= right:
            mid = math.floor((left + right) / 2)
            if mid ** 2 == x:
                return mid
            elif mid ** 2 < x:
                left = mid + 1
                ans = mid
            else:
                right = mid - 1
        return ans


################# Question 2 #################


# x_0: initial guess
# EPSILON: stop when abs(x - x_new) < EPSILON
# MAX_ITER: maximum number of iterations

## NOTE: you must use the default values of the above parameters, do not change them

def find_root(f, fprime, x_0=1.0, EPSILON = 1E-7, MAX_ITER = 1000): # do not change the heading of the function
    x_i = x_0

    for i in range(MAX_ITER):
        x_j = x_i - f(x_i) / fprime(x_i)
        if abs(x_i - x_j) < EPSILON:
            return x_j
        x_i = x_j

    return -1


################# Question 3 #################

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

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

def findDepth(root, i):
    if len(root.children) > 0:
        i += 1
        for child in root.children:
            if len(child.children) > 0:
                i += 1
                findDepth(child, i)
    return i

def max_depth(root): # do not change the heading of the function
    i = 1
    depth = findDepth(root, i)
    return depth
