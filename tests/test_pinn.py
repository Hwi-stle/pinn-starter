"""PINN 핵심 컴포넌트(모델, 자동미분, 물리 잔차)를 검증하는 단위 테스트."""

import math
import torch

from pinn import DampedHarmonicOscillator, HeatEquation1D, MLP, derivative


def test_mlp_output_shape() -> None:
    model = MLP(in_dim=2, out_dim=1, hidden_dim=16, num_hidden_layers=2)
    x = torch.randn(10, 2)
    y = model(x)
    assert y.shape == (10, 1)


def test_autograd_first_and_second_derivative() -> None:
    """f(x) = x^3 의 1차 미분(3x^2)과 2차 미분(6x)이 정확히 계산되는지 검증."""
    x = torch.linspace(-2.0, 2.0, 25).view(-1, 1).requires_grad_(True)
    y = x**3

    dy_dx = derivative(y, x, order=1)
    d2y_dx2 = derivative(y, x, order=2)

    assert torch.allclose(dy_dx, 3.0 * (x**2), atol=1e-5)
    assert torch.allclose(d2y_dx2, 6.0 * x, atol=1e-5)


def test_harmonic_oscillator_exact_solution_satisfies_ode() -> None:
    """해석해 자체를 ODE에 대입하면 잔차(residual)가 0에 가까워야 함."""
    problem = DampedHarmonicOscillator(delta=2.0, omega_0=20.0)
    t = torch.linspace(0.0, 1.0, 50).view(-1, 1).requires_grad_(True)

    # 해석해를 모델처럼 감싸서 residual 계산
    class ExactModel(torch.nn.Module):
        def forward(self, t_in: torch.Tensor) -> torch.Tensor:
            return problem.exact_solution(t_in)

    residual = problem.physics_residual(ExactModel(), t)
    assert torch.max(torch.abs(residual)).item() < 1e-3

    # 초기조건 u(0) = 1, u'(0) = 0 확인
    t0 = torch.tensor([[0.0]], requires_grad=True)
    u0 = problem.exact_solution(t0)
    du0 = derivative(u0, t0, order=1)
    assert math.isclose(u0.item(), 1.0, abs_tol=1e-6)
    assert math.isclose(du0.item(), 0.0, abs_tol=1e-5)


def test_heat_equation_exact_solution_satisfies_pde() -> None:
    """열전도 방정식 해석해가 PDE, 초기조건, 경계조건을 모두 만족하는지 검증."""
    problem = HeatEquation1D(alpha=0.1)
    x = torch.linspace(0.05, 0.95, 30).view(-1, 1).requires_grad_(True)
    t = torch.linspace(0.05, 0.95, 30).view(-1, 1).requires_grad_(True)

    class ExactHeatModel(torch.nn.Module):
        def forward(self, xt: torch.Tensor) -> torch.Tensor:
            return problem.exact_solution(xt[:, 0:1], xt[:, 1:2])

    residual = problem.physics_residual(ExactHeatModel(), x, t)
    assert torch.max(torch.abs(residual)).item() < 1e-5

    # 경계조건 u(0, t) = 0, u(1, t) = 0 검증
    zeros = torch.zeros_like(t)
    ones = torch.ones_like(t)
    assert torch.allclose(problem.exact_solution(zeros, t), zeros, atol=1e-6)
    assert torch.allclose(problem.exact_solution(ones, t), zeros, atol=1e-6)
