"""예제 1: 1D 감쇠 조화 진동자 (Damped Harmonic Oscillator) — 일반 NN vs PINN 비교.

이 스크립트를 통해 확인할 수 있는 핵심 내용:
1. 일반 신경망(Standard NN)은 관측 데이터(t ∈ [0, 0.4])가 있는 구간만 외우고,
   데이터가 없는 미래 구간(t ∈ [0.4, 1.0])에서는 완전히 엉뚱한 예측을 합니다.
2. 반면 PINN은 똑같은 소량의 데이터(10개)만 보고도,
   물리 법칙(d²u/dt² + mu*du/dt + k*u = 0)을 Loss에 추가함으로써
   데이터가 전혀 없는 미래 구간까지 정확하게 예측해냅니다.

실행 방법:
    python examples/01_harmonic_oscillator.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # GUI가 없는 서버/리모트 환경에서도 안전하게 이미지 저장
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 프로젝트 루트 기준 src/ 경로 추가 (별도 패키지 설치 없이도 바로 실행 가능)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pinn import DampedHarmonicOscillator, MLP  # noqa: E402


def train_standard_nn(
    t_data: torch.Tensor,
    u_data: torch.Tensor,
    epochs: int = 3000,
    lr: float = 1e-3,
) -> tuple[MLP, list[float]]:
    """물리 법칙 없이 관측 데이터(t_data, u_data)만으로 일반 신경망을 학습합니다."""
    torch.manual_seed(42)
    model = MLP(in_dim=1, out_dim=1, hidden_dim=32, num_hidden_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    loss_history: list[float] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        u_pred = model(t_data)
        loss = mse(u_pred, u_data)

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 1000 == 0 or epoch == 1:
            print(f"[Standard NN] Epoch {epoch:5d}/{epochs} | Data Loss: {loss.item():.6f}")

    return model, loss_history


def train_pinn(
    problem: DampedHarmonicOscillator,
    t_data: torch.Tensor,
    u_data: torch.Tensor,
    t_physics: torch.Tensor,
    epochs: int = 12000,
    lr: float = 1e-3,
    lambda_physics: float = 1e-4,
) -> tuple[MLP, dict[str, list[float]]]:
    """관측 데이터 손실(Loss_data) + 물리 법칙 손실(Loss_physics)로 PINN을 학습합니다.

    참고 — 왜 lambda_physics = 1e-4 를 곱할까요?
        우리 문제에서 k = omega_0² = 400 이므로, 물리 잔차 r = u'' + mu*u' + k*u 는
        초기에 수백 단위 크기를 가집니다. 이를 제곱(MSE)하면 ~10^5 수준이 되어
        ~1 수준인 Data Loss를 압도해버립니다.
        따라서 1e-4 (≈ 1/k에 대응하는 스케일 보정)를 곱해 두 Loss의 균형을 맞춥니다.
    """
    torch.manual_seed(42)
    model = MLP(in_dim=1, out_dim=1, hidden_dim=32, num_hidden_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    history: dict[str, list[float]] = {"total": [], "data": [], "physics": []}

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # 1) 데이터 손실: 실제 관측된 10개 점의 값을 맞추도록 학습
        u_data_pred = model(t_data)
        loss_data = mse(u_data_pred, u_data)

        # 2) 물리 손실: 전체 구간(t ∈ [0, 1])의 collocation 점에서 ODE 잔차가 0이 되도록 학습
        #    (주의: t_physics에는 정답 u 값이 전혀 들어가지 않습니다!)
        residual = problem.physics_residual(model, t_physics)
        loss_physics = torch.mean(residual**2)

        # 3) 총 손실 = 데이터 손실 + 가중치 * 물리 손실
        loss = loss_data + lambda_physics * loss_physics

        loss.backward()
        optimizer.step()

        history["total"].append(loss.item())
        history["data"].append(loss_data.item())
        history["physics"].append((lambda_physics * loss_physics).item())

        if epoch % 2000 == 0 or epoch == 1:
            print(
                f"[PINN]        Epoch {epoch:5d}/{epochs} | "
                f"Total: {loss.item():.6f} | "
                f"Data: {loss_data.item():.6f} | "
                f"Physics(weighted): {(lambda_physics * loss_physics).item():.6f}"
            )

    return model, history


def plot_results(
    t_test: torch.Tensor,
    u_exact: torch.Tensor,
    t_data: torch.Tensor,
    u_data: torch.Tensor,
    t_physics: torch.Tensor,
    u_nn: torch.Tensor,
    u_pinn: torch.Tensor,
    pinn_history: dict[str, list[float]],
    save_path: Path,
) -> None:
    """일반 NN vs PINN 예측 비교 및 학습 손실 곡선을 저장합니다."""
    fig = plt.figure(figsize=(13, 8.5), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.35, wspace=0.22)

    t_np = t_test.numpy().squeeze()
    exact_np = u_exact.numpy().squeeze()
    td_np = t_data.numpy().squeeze()
    ud_np = u_data.numpy().squeeze()
    tp_np = t_physics.detach().numpy().squeeze()

    # 1) 좌상단: 일반 신경망(Standard NN)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axvspan(0.4, 1.0, color="#fde2e4", alpha=0.35, label="Extrapolation zone (no data)")
    ax1.plot(t_np, exact_np, color="#222222", linestyle="--", linewidth=2.0, label="Exact solution")
    ax1.plot(t_np, u_nn.numpy().squeeze(), color="#d62828", linewidth=2.2, label="Standard NN")
    ax1.scatter(td_np, ud_np, color="#003049", s=45, zorder=5, label="Training data (10 pts)")
    ax1.set_title("1. Standard Neural Network (Data Only)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time t")
    ax1.set_ylabel("Displacement u(t)")
    ax1.set_ylim(-1.3, 1.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)

    # 2) 우상단: PINN
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axvspan(0.4, 1.0, color="#d8f3dc", alpha=0.35, label="Extrapolation zone (no data)")
    ax2.plot(t_np, exact_np, color="#222222", linestyle="--", linewidth=2.0, label="Exact solution")
    ax2.plot(t_np, u_pinn.numpy().squeeze(), color="#1d4ed8", linewidth=2.2, label="PINN prediction")
    ax2.scatter(td_np, ud_np, color="#003049", s=45, zorder=5, label="Training data (10 pts)")
    ax2.scatter(
        tp_np,
        [-1.18] * len(tp_np),
        color="#2a9d8f",
        marker="|",
        s=50,
        alpha=0.8,
        label="Physics collocation pts",
    )
    ax2.set_title("2. PINN (Data + Physics ODE Loss)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Displacement u(t)")
    ax2.set_ylim(-1.3, 1.3)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8.5)

    # 3) 하단 전체: PINN 학습 손실 곡선 (로그 스케일)
    ax3 = fig.add_subplot(gs[1, :])
    epochs = range(1, len(pinn_history["total"]) + 1)
    ax3.semilogy(epochs, pinn_history["total"], color="#111827", linewidth=1.8, label="Total Loss")
    ax3.semilogy(epochs, pinn_history["data"], color="#2563eb", linewidth=1.4, alpha=0.85, label="Data Loss")
    ax3.semilogy(
        epochs,
        pinn_history["physics"],
        color="#059669",
        linewidth=1.4,
        alpha=0.85,
        label="Weighted Physics Loss",
    )
    ax3.set_title("PINN Training Loss Curve (Log Scale)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss (log scale)")
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "1D Damped Harmonic Oscillator: Standard NN vs. PINN",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 68)
    print("예제 1: 1D 감쇠 조화 진동자 (Standard NN vs PINN)")
    print("=" * 68)

    problem = DampedHarmonicOscillator(delta=2.0, omega_0=20.0)

    # 평가용 전체 시간 격자: t ∈ [0, 1]
    t_test = torch.linspace(0.0, 1.0, 500).view(-1, 1)
    u_exact = problem.exact_solution(t_test)

    # 관측 데이터: 앞부분 t ∈ [0, 0.4] 구간에서만 10개 점 관측
    t_data = torch.linspace(0.0, 0.4, 10).view(-1, 1)
    u_data = problem.exact_solution(t_data)

    # 물리 법칙 검사 지점(Collocation points): 전체 구간 t ∈ [0, 1]에 40개 배치
    # requires_grad=True가 있어야 PyTorch가 입력 t에 대한 미분(du/dt, d²u/dt²)을 계산 가능
    t_physics = torch.linspace(0.0, 1.0, 40).view(-1, 1).requires_grad_(True)

    print("\n--- 1단계: 일반 신경망(Standard NN) 학습 ---")
    nn_model, _ = train_standard_nn(t_data, u_data, epochs=3000, lr=1e-3)

    print("\n--- 2단계: PINN 학습 ---")
    pinn_model, pinn_history = train_pinn(
        problem,
        t_data,
        u_data,
        t_physics,
        epochs=12000,
        lr=1e-3,
        lambda_physics=1e-4,
    )

    # 평가
    with torch.no_grad():
        u_nn = nn_model(t_test)
        u_pinn = pinn_model(t_test)

    extrap_mask = (t_test >= 0.4).squeeze()
    nn_rmse_all = torch.sqrt(torch.mean((u_nn - u_exact) ** 2)).item()
    pinn_rmse_all = torch.sqrt(torch.mean((u_pinn - u_exact) ** 2)).item()
    nn_rmse_extrap = torch.sqrt(torch.mean((u_nn[extrap_mask] - u_exact[extrap_mask]) ** 2)).item()
    pinn_rmse_extrap = torch.sqrt(
        torch.mean((u_pinn[extrap_mask] - u_exact[extrap_mask]) ** 2)
    ).item()

    print("\n" + "=" * 68)
    print("최종 정확도 비교 (RMSE, 낮을수록 정확)")
    print("-" * 68)
    print(f"전체 구간 t ∈ [0.0, 1.0]  | Standard NN: {nn_rmse_all:.5f}  | PINN: {pinn_rmse_all:.5f}")
    print(
        f"미래 구간 t ∈ [0.4, 1.0]  | Standard NN: {nn_rmse_extrap:.5f}  | PINN: {pinn_rmse_extrap:.5f}"
    )
    print("=" * 68)

    output_path = PROJECT_ROOT / "results" / "01_harmonic_oscillator.png"
    plot_results(
        t_test=t_test,
        u_exact=u_exact,
        t_data=t_data,
        u_data=u_data,
        t_physics=t_physics,
        u_nn=u_nn,
        u_pinn=u_pinn,
        pinn_history=pinn_history,
        save_path=output_path,
    )
    print(f"결과 그래프 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
