import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 1. 加载数据集并构建有向图
edges = pd.read_csv('PageRank_Dataset.csv', header=None, names=['source', 'target'])
G = nx.DiGraph()
G.add_edges_from(edges.values)

# 2. 计算 PageRank 值（默认 β = 0.85）
beta = 0.85
pagerank_values = nx.pagerank(G, alpha=beta)

# 3. 输出结果
# 3.1 提取 PageRank 值最高的 20 个节点
sorted_pagerank = sorted(pagerank_values.items(), key=lambda x: x[1], reverse=True)
top_20_nodes = sorted_pagerank[:20]

print("PageRank 值最高的 20 个节点：")
for node, score in top_20_nodes:
    print(f"节点 ID: {node}, PageRank 值: {score}")

# 3.2 保存所有节点的 PageRank 值到 CSV 文件
pagerank_df = pd.DataFrame(pagerank_values.items(), columns=['NodeID', 'PageRankValue'])
pagerank_df.to_csv('PageRank_Results.csv', index=False)

# 4. 探索不同的阻尼因子（β）对结果的影响（可选）
beta_values = [0.6, 0.75, 0.85, 0.95]
pagerank_results = {}

for beta in beta_values:
    # 计算不同 β 值下的 PageRank
    pr_values = nx.pagerank(G, alpha=beta)
    pagerank_results[beta] = pr_values

    # 排序并提取前 20 个节点
    sorted_pr = sorted(pr_values.items(), key=lambda x: x[1], reverse=True)
    top_20 = sorted_pr[:20]

    print(f"\n当 β = {beta} 时，PageRank 值最高的 20 个节点：")
    for node, score in top_20:
        print(f"节点 ID: {node}, PageRank 值: {score}")

    # 设置字体为 SimHei（黑体）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

    # 绘制 PageRank 值分布直方图
    plt.figure()
    plt.hist(pr_values.values(), bins=50)
    plt.title(f'当 β = {beta} 时的 PageRank 值分布')
    plt.xlabel('PageRank 值')
    plt.ylabel('频率')
    plt.show()
