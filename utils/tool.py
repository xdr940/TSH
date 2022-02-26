

import time
def time_stat(last_end):
    end = time.time()
    running_time = end - last_end
    print('time cost : %.5f sec' % running_time)
    return end

def get_now():
    return time.time()

max_d = []

# dfs_depth base Code reference networkx

def dfs_depth(G, source=None, depth_limit=None):
    if source is None:
        nodes = G
    else:
        nodes = [source]
    visited = set()
    if depth_limit is None:
        depth_limit = len(G)
    for start in nodes:
        # print(start)
        if start in visited:
            continue
        max_depth = 0
        visited.add(start)
        stack = [(start, depth_limit, iter(G[start]))]
        while stack:
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    yield parent, child
                    visited.add(child)
                    if depth_now > 1:
                        if ((depth_limit - depth_now + 1) > max_depth):
                            max_depth = depth_limit - depth_now + 1
                        stack.append((child, depth_now - 1, iter(G[child])))
            except StopIteration:
                stack.pop()
    global max_d
    max_d.append(max_depth)


