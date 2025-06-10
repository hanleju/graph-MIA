import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


def confidence_adaptive_noise(logits, base_scale=0.2, min_scale=0.05, gamma=1.0):
    """
    logits: (N, C) torch.Tensor (N: 노드 수, C: label 수)
    base_scale: 최대 노이즈 scale (낮은 confidence에 적용)
    min_scale: 최소 노이즈 scale (높은 confidence에 적용)
    gamma: non-linearity 제어 (1이면 선형, 2면 비선형)
    반환: noise가 더해진 logits
    """
    probs = torch.sigmoid(logits)
    # scale: 높은 확률(확신)일수록 noise 적게, 낮을수록 noise 크게
    scale = min_scale + (base_scale - min_scale) * (1 - probs) ** gamma  # (N, C)

    noise = torch.distributions.Laplace(0, scale).sample().to(logits.device)
    logits_noisy = logits + noise
    return logits_noisy

def train_batchwise(model, loader, device, epochs, learning_rate, pos_weight, noise=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    if pos_weight is not None:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(1, epochs)):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            
            # ----- 노이즈 적용 -----
            if noise is True:
                out = confidence_adaptive_noise(out)
            
            loss = loss_fn(out[batch.batch_size:], batch.y[batch.batch_size:])  # target nodes만 학습
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss: {total_loss:.4f}")

def test_batchwise(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)

            # batch.input_id: loader에 따라 target 노드 인덱스를 의미
            target_mask = torch.arange(batch.batch_size, device=device)
            logits = out[target_mask]
            targets = batch.y[target_mask]

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds_np = torch.cat(all_preds, dim=0).numpy()
    targets_np = torch.cat(all_targets, dim=0).numpy()

    # Multi-label 평가
    f1_micro = f1_score(targets_np, preds_np, average='micro', zero_division=0)
    f1_macro = f1_score(targets_np, preds_np, average='macro', zero_division=0)
    f1_per_label = f1_score(targets_np, preds_np, average=None, zero_division=0)

    precision_macro = precision_score(targets_np, preds_np, average='macro', zero_division=0)
    recall_macro = recall_score(targets_np, preds_np, average='macro', zero_division=0)

    print(f"[Test] F1-micro: {f1_micro:.4f} | F1-macro: {f1_macro:.4f} | "
          f"Precision-macro: {precision_macro:.4f} | Recall-macro: {recall_macro:.4f}")
    print(f"F1 per label:\n{np.round(f1_per_label, 3)}")

    return f1_micro, f1_macro, precision_macro, recall_macro, f1_per_label

def extract_attack_features(model, x, edge_index, indices, top_k=1, device='cpu'):
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(indices), 512):  # 배치 처리 (속도 및 메모리 절약)
            batch_idx = indices[i:i+512].to(device)
            logits = model(x.to(device), edge_index.to(device))[batch_idx]
            probs = torch.sigmoid(logits).cpu()

            # Top-k probability 값만 feature로 사용
            topk_vals, _ = probs.topk(k=top_k, dim=1)
            features.append(topk_vals)

    return torch.cat(features, dim=0)

def train_attack_model(shadow_model, data, shadow_train_idx, shadow_test_idx, top_k=1, device='cpu'):
    # Shadow member / non-member 특징 추출
    X_member = extract_attack_features(shadow_model, data.x, data.edge_index, shadow_train_idx, top_k, device)
    X_nonmember = extract_attack_features(shadow_model, data.x, data.edge_index, shadow_test_idx, top_k, device)

    # 데이터 구성
    X = torch.cat([X_member, X_nonmember], dim=0).numpy()
    y = torch.cat([torch.ones(len(X_member)), torch.zeros(len(X_nonmember))], dim=0).numpy()

    # 로지스틱 회귀 기반 Attack 모델 학습
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def evaluate_attack_model(clf, target_model, data, target_train_idx, target_test_idx, top_k=1, device='cpu'):
    X_member = extract_attack_features(target_model, data.x, data.edge_index, target_train_idx, top_k, device)
    X_nonmember = extract_attack_features(target_model, data.x, data.edge_index, target_test_idx, top_k, device)

    X_test = torch.cat([X_member, X_nonmember], dim=0).numpy()
    y_test = torch.cat([torch.ones(len(X_member)), torch.zeros(len(X_nonmember))], dim=0).numpy()

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"[Attack Evaluation] Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    return acc, auc