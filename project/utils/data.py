import torch
from sklearn.model_selection import train_test_split

def split_multilabel_dataset(data, seed=42):
    """
    Multi-label classification에 맞는 dataset 분할 함수

    전체 노드를 반으로 나누고, 각 그룹을 7:3으로 train/test로 나눔.
    반환: target_train, target_test, shadow_train, shadow_test
    """
    torch.manual_seed(seed)

    all_indices = torch.arange(data.num_nodes)
    all_indices_np = all_indices.cpu().numpy()

    # Step 1: 전체 노드를 50%씩 → target, shadow
    target_indices, shadow_indices = train_test_split(
        all_indices_np,
        train_size=0.6,
        random_state=seed
    )

    # Step 2: 각 그룹에서 70:30 비율로 train/test 나누기
    target_train, target_test = train_test_split(
        target_indices,
        train_size=0.7,
        random_state=seed
    )

    shadow_train, shadow_test = train_test_split(
        shadow_indices,
        train_size=0.7,
        random_state=seed
    )

    # torch.Tensor로 변환하여 반환
    return (
        torch.tensor(target_train, dtype=torch.long),
        torch.tensor(target_test, dtype=torch.long),
        torch.tensor(shadow_train, dtype=torch.long),
        torch.tensor(shadow_test, dtype=torch.long)
    )