import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer
from glob import glob
import matplotlib.pyplot as plt

def get_global_featnames(folder_path):
    featname_files = glob(os.path.join(folder_path, "*.featnames"))
    all_featnames = set()
    for fn in featname_files:
        with open(fn) as f:
            all_featnames.update(line.strip().split()[1] for line in f)
    all_featnames = sorted(all_featnames)
    return all_featnames

def load_single_ego_graph(folder_path, ego_id, featname2idx, global_feat_dim):
    edge_file = os.path.join(folder_path, f"{ego_id}.edges")
    feat_file = os.path.join(folder_path, f"{ego_id}.feat")
    circle_file = os.path.join(folder_path, f"{ego_id}.circles")
    featname_file = os.path.join(folder_path, f"{ego_id}.featnames")

    # Load node features
    feats = np.loadtxt(feat_file, dtype=str)
    node_ids = feats[:, 0]
    x_local = feats[:, 1:].astype(float)
    # local featname 순서대로
    with open(featname_file) as f:
        ego_featnames = [line.strip().split()[1] for line in f]
    ego_feat_idx_map = [featname2idx[f] for f in ego_featnames]
    # global feature 공간에 맞게 0-padding
    x = np.zeros((x_local.shape[0], global_feat_dim), dtype=float)
    x[:, ego_feat_idx_map] = x_local
    x = torch.tensor(x, dtype=torch.float)

    # Load edge list
    edges = np.loadtxt(edge_file, dtype=str)
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    edge_index = torch.tensor([
        [node_ids.tolist().index(src) if src in node_ids else -1 for src in edges[:, 0]],
        [node_ids.tolist().index(dst) if dst in node_ids else -1 for dst in edges[:, 1]]
    ], dtype=torch.long)
    mask = (edge_index[0] >= 0) & (edge_index[1] >= 0)
    edge_index = edge_index[:, mask]
    if edge_index.shape[1] == 0:
        return None, None, None

    # Load circles
    labels = [[] for _ in range(len(node_ids))]
    found_label = False
    with open(circle_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            circle_id = parts[0]
            circle_members = parts[1:]
            for nid in circle_members:
                if nid in node_ids:
                    labels[node_ids.tolist().index(nid)].append(circle_id)
                    found_label = True
    if not found_label:
        return None, None, None

    return x, edge_index, labels

def load_all_ego_graphs(folder_path, topk=2000, importance_mode="var"):
    # 전체 featname 기준 global feature space 생성
    all_featnames = get_global_featnames(folder_path)
    featname2idx = {name: i for i, name in enumerate(all_featnames)}
    global_feat_dim = len(all_featnames)

    # 모든 ego 그래프를 global feature로 쌓기
    ego_ids = [os.path.basename(f).split('.')[0] for f in glob(os.path.join(folder_path, '*.edges'))]
    all_x, all_edges, all_labels = [], [], []
    offset = 0
    node_count_per_ego = []
    for ego_id in ego_ids:
        x, edge_index, labels = load_single_ego_graph(folder_path, ego_id, featname2idx, global_feat_dim)
        if x is None:
            print(f"[SKIP] {ego_id}: invalid graph")
            continue
        print(f"[LOAD] {ego_id}: x={x.shape}, edge_index={edge_index.shape}, #labels with values={sum([len(l) > 0 for l in labels])}")
        all_x.append(x)
        all_labels.extend(labels)
        edge_index = edge_index + offset
        all_edges.append(edge_index)
        offset += x.size(0)
        node_count_per_ego.append(x.size(0))
    x = torch.cat(all_x, dim=0)  # [N, 19044]
    edge_index = torch.cat(all_edges, dim=1)

    # 중요도 기반 feature selection (분산 또는 등장빈도)
    x_np = x.numpy()
    if importance_mode == "var":
        feat_importance = x_np.var(axis=0)
    elif importance_mode == "mean":
        feat_importance = np.abs(x_np.mean(axis=0))
    elif importance_mode == "count":
        feat_importance = (x_np != 0).sum(axis=0)
    else:
        raise ValueError("importance_mode should be 'var', 'mean' or 'count'")

    topk_idx = np.argsort(feat_importance)[::-1][:topk]

    if isinstance(x, torch.Tensor):
        x_np = x.numpy()
    else:
        x_np = x
    x_selected = torch.tensor(x_np[:, topk_idx].copy(), dtype=torch.float)

    mlb = MultiLabelBinarizer()
    y = torch.tensor(mlb.fit_transform(all_labels), dtype=torch.float)
    return Data(x=x_selected, edge_index=edge_index, y=y), mlb, all_labels

def visualize_label_distribution(mlb, all_labels, title="Google+ Circle Label Distribution"):
    label_names = mlb.classes_
    label_counts = [sum(label in labels for labels in all_labels) for label in label_names]
    plt.figure(figsize=(10, max(6, len(label_names) // 8)))  # 라벨이 많으면 더 세로로 길게
    plt.barh(range(len(label_names)), label_counts, tick_label=label_names)
    plt.yticks(fontsize=8)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Circle Labels")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def save_edge_list_for_gephi(data, file_path="graph_edges.csv"):
    edge_index = data.edge_index.cpu().numpy()
    edges = edge_index.T  # shape: [num_edges, 2]

    # CSV 저장: Source,Target 형식
    np.savetxt(file_path, edges, fmt='%d', delimiter=',', header='Source,Target', comments='')
    print(f"Edge list saved to {file_path}")


if __name__ == '__main__':
    
    # google
    dir = "./project/dataset/google/gplus"
    save_path = "./project/dataset/google/count/gplus_graph.pt"
    featname_txt_path = "./project/dataset/google/count/gplus_feature.txt"
    csv_path = "./project/csv/google_count.csv"
    
    data, mlb, all_labels = load_all_ego_graphs(dir, topk=2000, importance_mode="count")

    # 저장
    torch.save((data, mlb, all_labels), save_path)
    print(f"Saved to: {save_path}")

    save_edge_list_for_gephi(data, csv_path)
    print(f"Saved to: {csv_path}")

    # 라벨별 노드 수 시각화
    visualize_label_distribution(mlb, all_labels)