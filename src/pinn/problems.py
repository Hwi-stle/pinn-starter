"""PINN 학습용 대표 물리 문제 정의 (해석해 + 물리 방정식 잔차).

포함된 문제:
1. DampedHarmonicOscillator (1D 감쇠 조화 진동자 - ODE)
   - m * u'' + mu * u' + k * u = 0
2. HeatEquation1D (1차원 열전도 방정식 - PDE)
   - ∂u/∂t - alpha * ∂²u/∂x² = 0
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .autograd import derivative


@dataclass(frozen=True)
class DampedHarmonicOscillator:
    """1D 과소감쇠(underdamped) 조화 진동자 문제.

    지배 방정식 (ODE):
        m * (d²u/dt²) + mu * (du/dt) + k * u = 0
        (편의상 m = 1, mu = 2 * delta, k = omega_0² 로 둡니다.)

    초기 조건:
        u(0) = 1,  u'(0) = 0
    """

    delta: float = 2.0  # 감쇠 계수 (클수록 빨리 멈춤)
    omega_0: float = 20.0  # 고유 각진동수 (클수록 빠르게 진동)

    def __post_init__(self) -> None:
        if self.delta >= self.omega_0:
            raise ValueError("과소감쇠(underdamped) 조건은 delta < omega_0 이어야 합니다.")

    @property
    def mu(self) -> float:
        """마찰(감쇠) 계수 mu = 2 * delta."""
        return 2.0 * self.delta

    @property
    def k(self) -> float:
        """용수철 상수 k = omega_0²."""
        return self.omega_0**2

    @property
    def omega(self) -> float:
        """감쇠 진동수 omega = sqrt(omega_0² - delta²)."""
        return math.sqrt(self.omega_0**2 - self.delta**2)

    def exact_solution(self, t: torch.Tensor) -> torch.Tensor:
        """해석해(정확한 수학적 정답): u(t) = exp(-delta*t) * (cos(w*t) + (delta/w)*sin(w*t))."""
        w = self.omega
        return torch.exp(-self.delta * t) * (
            torch.cos(w * t) + (self.delta / w) * torch.sin(w * t)
        )

    def physics_residual(self, model: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
        """물리 방정식 잔차(residual)를 계산합니다.

        물리 법칙이 완벽하게 지켜지면 아래 식의 값은 0이 됩니다:
            r(t) = d²u/dt² + mu * du/dt + k * u
        """
        u = model(t)
        du_dt = derivative(u, t, order=1)
        d2u_dt2 = derivative(u, t, order=2)
        return d2u_dt2 + self.mu * du_dt + self.k * u


@dataclass(frozen=True)
class HeatEquation1D:
    """1차원 막대기의 열전도(Heat Diffusion) 문제.

    지배 방정식 (PDE):
        ∂u/∂t - alpha * (∂²u/∂x²) = 0,   x ∈ [0, 1], t ∈ [0, 1]

    초기 조건 (IC, t = 0):
        u(x, 0) = sin(pi * x)   (가운데가 뜨겁고 양 끝이 0도인 상태)

    경계 조건 (BC, x = 0, x = 1):
        u(0, t) = 0,  u(1, t) = 0   (양 끝은 항상 0도로 고정)
    """

    alpha: float = 0.1  # 열확산 계수 (클수록 열이 빨리 식음)

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """해석해(정확한 수학적 정답): u(x, t) = exp(-alpha * pi² * t) * sin(pi * x)."""
        return torch.exp(-self.alpha * (math.pi**2) * t) * torch.sin(math.pi * x)

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """t = 0 일 때 초기 온도 분포 u(x, 0) = sin(pi * x)."""
        return torch.sin(math.pi * x)

    def physics_residual(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """열전도 PDE 잔차 r(x, t) = ∂u/∂t - alpha * ∂²u/∂x² 를 계산합니다.

        x와 t를 각각 requires_grad=True 상태로 받은 뒤 concat해서 모델에 넣으면,
        PyTorch autograd가 ∂u/∂t 와 ∂²u/∂x² 를 독립적으로 정확하게 추적해줍니다.
        """
        xt = torch.cat([x, t], dim=1)
        u = model(xt)
        du_dt = derivative(u, t, order=1)
        d2u_dx2 = derivative(u, x, order=2)
        return du_dt - self.alpha * d2u_dx2
