"""PyTorch 자동미분(Autograd)을 이용해 물리 미분값을 계산하는 유틸리티.

PINN의 가장 중요한 아이디어:
- 보통 딥러닝에서는 Loss를 '가중치(weights)'에 대해 미분합니다.
- PINN에서는 여기에 더해, 신경망 출력 u를 '입력 좌표(t 또는 x)'에 대해 미분하여
  du/dt, d²u/dt², ∂²u/∂x² 같은 물리 미분값을 정확하게 구합니다.
- 이때 `create_graph=True`를 반드시 켜야 합니다!
  그래야 미분값 자체도 연산 그래프에 남아, 이후 `loss.backward()`를 호출할 때
  신경망 가중치까지 역전파(2차 미분 그래프 통과)가 정상적으로 일어납니다.
"""

from __future__ import annotations

import torch


def derivative(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """출력 텐서 y를 입력 텐서 x에 대해 order번 미분합니다 (dy/dx, d²y/dx² 등).

    Args:
        y: 신경망 출력 텐서, shape (N, 1)
        x: 미분 기준 입력 텐서 (requires_grad=True 필요), shape (N, 1)
        order: 미분 차수 (1이면 1차 미분, 2이면 2차 미분)

    Returns:
        y와 동일한 shape (N, 1)의 미분값 텐서
    """
    if order < 1:
        raise ValueError(f"미분 차수(order)는 1 이상이어야 합니다. 받은 값: {order}")

    current = y
    for _ in range(order):
        # grad_outputs=torch.ones_like(current):
        # 배치 크기 N개의 각 점에 대해 독립적으로 ∂y_i/∂x_i를 한 번에 계산하기 위해
        # 벡터-야코비안 곱(VJP)의 가중치를 모두 1로 줍니다.
        grad = torch.autograd.grad(
            outputs=current,
            inputs=x,
            grad_outputs=torch.ones_like(current),
            create_graph=True,
            retain_graph=True,
        )[0]
        current = grad

    return current
