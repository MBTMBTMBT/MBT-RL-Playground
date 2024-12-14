from pyvis.network import Network
import networkx as nx

mdp = {
    'S1': {'a1': [('S2', 0.8, 10), ('S3', 0.2, 5)]},
    'S2': {'a2': [('S3', 1.0, 15)]},
    'S3': {'a3': [('S1', 0.6, -5), ('S2', 0.4, 0)]}
}

# 创建一个有向图
G = nx.DiGraph()

# 添加节点和边
for state, actions in mdp.items():
    for action, transitions in actions.items():
        for next_state, prob, reward in transitions:
            G.add_edge(state, next_state, label=f"{action}\nP={prob}, R={reward}")

# 使用 Pyvis 进行可视化
net = Network(height='750px', width='100%', directed=True)
net.from_nx(G)

# 自定义节点和边的样式
for node in G.nodes():
    net.get_node(node)['title'] = f"State: {node}"
    net.get_node(node)['color'] = '#87CEEB'

# 保存并在浏览器中打开
net.show("graph.html", notebook=False)
