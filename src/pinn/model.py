"""PINN에서 사용하는 기본 다층 퍼셉트론(MLP) 모델 정의.

핵심 포인트:
1. 활성화 함수로 ReLU가 아닌 Tanh를 사용합니다.
   - ReLU는 2차 미분이 0이라서, 2계 미분방정식(예: 가속도 d²u/dt², 열확산 ∂²u/∂x²)을
     학습할 때 물리 손실(Physics Loss)의 기울기가 사라져 버립니다.
   - Tanh는 무한 번 매끄럽게 미분 가능(C∞)해서 PINN의 사실상 표준 활성화 함수입니다.
2. 가중치 초기화는 Tanh에 잘 맞는 Xavier(Glorot) 초기화를 사용합니다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """PINN용 완전 연결 신경망(Fully Connected Neural Network).

    예시:
        - 1D ODE (시간 t -> 변위 u): in_dim=1, out_dim=1
        - 1D PDE (위치 x, 시간 t -> 온도 u): in_dim=2, out_dim=1
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 32,
        num_hidden_layers: int = 3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        # 1) 입력층 -> 첫 번째 은닉층
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())

        # 2) 중간 은닉층들
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        # 3) 마지막 은닉층 -> 출력층 (물리량은 범위 제한이 없으므로 활성화 함수 없음)
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Tanh 활성화 함수에 적합한 Xavier Uniform 초기화."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
