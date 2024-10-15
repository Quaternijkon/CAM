import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#  加载数据集并构建有向图
edges = pd.read_csv('PageRank_Dataset.csv', header=None, names=['source', 'target'])
G = nx.DiGraph()
G.add_edges_from(edges.values)

#  探索不同的β对结果的影响
beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
pagerank_results = {}

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))

#  计算不同 β 值下的 PageRank 并绘制分布图
for beta in beta_values:
    # 计算不同 β 值下的 PageRank
    pr_values = nx.pagerank(G, alpha=beta)
    pagerank_results[beta] = pr_values

    values = list(pr_values.values())
    counts, bin_edges = np.histogram(values, bins=50, range=(0, 0.0002))

    # 绘制折线图
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.plot(bin_centers, counts, label=f'β = {beta}', marker='o')

plt.xlim(0, 0.0002)

plt.title('不同 β 值下的 PageRank 值分布')
plt.xlabel('PageRank 值')
plt.ylabel('频率')
plt.legend()
plt.tight_layout()
plt.show()

#  输出每种 β 值下 PageRank 分数最高的 20 个节点
for beta, pr in pagerank_results.items():
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    top_20 = sorted_pr[:20]

    print(f'\nβ = {beta} 时 PageRank 分数最高的 20 个节点:')
    print('-' * 50)
    print(f'{"排名":<5} {"节点":<15} {"PageRank 分数":<20}')
    print('-' * 50)
    for rank, (node, score) in enumerate(top_20, start=1):
        print(f'{rank:<5} {node:<15} {score:<20.8f}')
    print('-' * 50)
