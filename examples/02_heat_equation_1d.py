"""예제 2: 1차원 열전도 방정식 (1D Heat Equation, PDE) — 관측 데이터 없이(Data-free) 풀기.

이 스크립트를 통해 확인할 수 있는 핵심 내용:
1. 입력 변수가 2개(공간 x, 시간 t)인 편미분 방정식(PDE)을 PINN으로 푸는 표준 패턴.
2. 내부 구간(0 < x < 1, t > 0)의 정답 데이터를 단 1개도 주지 않고,
   오직 초기조건(IC) + 경계조건(BC) + 열전도 방정식(PDE)만으로
   시공간 전체의 온도 분포 u(x, t)를 정확하게 예측합니다.

실행 방법:
    python examples/02_heat_equation_1d.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pinn import HeatEquation1D, MLP  # noqa: E402


def sample_training_points(
    problem: HeatEquation1D,
    n_ic: int = 100,
    n_bc: int = 100,
    n_pde: int = 2000,
) -> dict[str, torch.Tensor]:
    """초기조건(IC), 경계조건(BC), 내부 PDE 검사점(Collocation points)을 생성합니다."""
    # 1) 초기조건 (Initial Condition): t = 0, x ∈ [0, 1]
    x_ic = torch.linspace(0.0, 1.0, n_ic).view(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = problem.initial_condition(x_ic)

    # 2) 경계조건 (Boundary Condition): x = 0 및 x = 1, t ∈ [0, 1]
    t_bc = torch.linspace(0.0, 1.0, n_bc).view(-1, 1)
    x_bc_left = torch.zeros_like(t_bc)
    x_bc_right = torch.ones_like(t_bc)

    # 3) 내부 물리 검사점 (Collocation Points): (x, t) ∈ (0, 1) × (0, 1)
    #    requires_grad=True 를 켜서 ∂u/∂t 와 ∂²u/∂x² 를 자동미분으로 구합니다.
    x_pde = torch.rand(n_pde, 1, requires_grad=True)
    t_pde = torch.rand(n_pde, 1, requires_grad=True)

    return {
        "x_ic": x_ic,
        "t_ic": t_ic,
        "u_ic": u_ic,
        "t_bc": t_bc,
        "x_bc_left": x_bc_left,
        "x_bc_right": x_bc_right,
        "x_pde": x_pde,
        "t_pde": t_pde,
    }


def train_heat_pinn(
    problem: HeatEquation1D,
    epochs: int = 4000,
    lr: float = 1e-3,
) -> tuple[MLP, dict[str, list[float]]]:
    """1D 열전도 방정식 PINN을 학습합니다 (내부 정답 데이터 0개!)."""
    torch.manual_seed(42)

    # 입력 2개 (x, t) -> 출력 1개 (온도 u)
    model = MLP(in_dim=2, out_dim=1, hidden_dim=32, num_hidden_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    pts = sample_training_points(problem)
    xt_ic = torch.cat([pts["x_ic"], pts["t_ic"]], dim=1)
    xt_bc_left = torch.cat([pts["x_bc_left"], pts["t_bc"]], dim=1)
    xt_bc_right = torch.cat([pts["x_bc_right"], pts["t_bc"]], dim=1)

    history: dict[str, list[float]] = {"total": [], "ic": [], "bc": [], "pde": []}

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # 1) 초기조건 손실 (t = 0에서 u(x, 0) = sin(pi * x))
        loss_ic = mse(model(xt_ic), pts["u_ic"])

        # 2) 경계조건 손실 (x = 0과 x = 1에서 u = 0)
        loss_bc = mse(model(xt_bc_left), torch.zeros_like(pts["t_bc"])) + mse(
            model(xt_bc_right), torch.zeros_like(pts["t_bc"])
        )

        # 3) 편미분 방정식 손실 (∂u/∂t - alpha * ∂²u/∂x² = 0)
        residual = problem.physics_residual(model, pts["x_pde"], pts["t_pde"])
        loss_pde = torch.mean(residual**2)

        # 총 손실 = IC + BC + PDE
        loss = loss_ic + loss_bc + loss_pde

        loss.backward()
        optimizer.step()

        history["total"].append(loss.item())
        history["ic"].append(loss_ic.item())
        history["bc"].append(loss_bc.item())
        history["pde"].append(loss_pde.item())

        if epoch % 1000 == 0 or epoch == 1:
            print(
                f"[Heat PINN] Epoch {epoch:5d}/{epochs} | "
                f"Total: {loss.item():.6f} | "
                f"IC: {loss_ic.item():.6f} | "
                f"BC: {loss_bc.item():.6f} | "
                f"PDE: {loss_pde.item():.6f}"
            )

    return model, history


def plot_results(
    problem: HeatEquation1D,
    model: MLP,
    save_path: Path,
) -> tuple[float, float]:
    """2D 시공간 히트맵(정답, PINN, 오차)과 시간별 온도 단면 그래프를 저장합니다."""
    nx, nt = 101, 101
    x = torch.linspace(0.0, 1.0, nx)
    t = torch.linspace(0.0, 1.0, nt)
    X, T = torch.meshgrid(x, t, indexing="ij")

    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    with torch.no_grad():
        u_exact = problem.exact_solution(x_flat, t_flat).reshape(nx, nt)
        u_pred = model(torch.cat([x_flat, t_flat], dim=1)).reshape(nx, nt)

    abs_err = torch.abs(u_exact - u_pred)
    rel_l2 = (torch.norm(u_exact - u_pred) / torch.norm(u_exact)).item()
    max_err = abs_err.max().item()

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), dpi=150)

    # 1) Exact solution heatmap
    im0 = axes[0, 0].imshow(
        u_exact.numpy(),
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 0].set_title("1. Exact Solution u(x, t)", fontsize=11.5, fontweight="bold")
    axes[0, 0].set_xlabel("Time t")
    axes[0, 0].set_ylabel("Position x")
    fig.colorbar(im0, ax=axes[0, 0])

    # 2) PINN prediction heatmap
    im1 = axes[0, 1].imshow(
        u_pred.numpy(),
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 1].set_title("2. PINN Prediction (Data-Free!)", fontsize=11.5, fontweight="bold")
    axes[0, 1].set_xlabel("Time t")
    axes[0, 1].set_ylabel("Position x")
    fig.colorbar(im1, ax=axes[0, 1])

    # 3) Absolute error heatmap
    im2 = axes[1, 0].imshow(
        abs_err.numpy(),
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[1, 0].set_title(
        f"3. Absolute Error |Exact - PINN| (Rel L2 = {rel_l2:.4%})",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Time t")
    axes[1, 0].set_ylabel("Position x")
    fig.colorbar(im2, ax=axes[1, 0])

    # 4) Time slices comparison
    ax_slice = axes[1, 1]
    colors = ["#ef4444", "#f59e0b", "#10b981", "#2563eb"]
    for t_val, color in zip([0.0, 0.25, 0.5, 1.0], colors):
        idx = int(round(t_val * (nt - 1)))
        ax_slice.plot(
            x.numpy(),
            u_exact[:, idx].numpy(),
            linestyle="--",
            color=color,
            linewidth=2.0,
            alpha=0.85,
            label=f"Exact (t={t_val:.2f})",
        )
        ax_slice.plot(
            x.numpy(),
            u_pred[:, idx].numpy(),
            linestyle="-",
            color=color,
            linewidth=1.5,
            alpha=0.95,
            marker="o",
            markevery=10,
            markersize=3.5,
        )

    ax_slice.set_title("4. Temperature Profiles Over Time (Dashed=Exact, Dots=PINN)", fontsize=10.5, fontweight="bold")
    ax_slice.set_xlabel("Position x")
    ax_slice.set_ylabel("Temperature u(x, t)")
    ax_slice.grid(True, alpha=0.3)
    ax_slice.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "1D Heat Equation Solved by PINN (∂u/∂t = α ∂²u/∂x², α = 0.1)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return rel_l2, max_err


def main() -> None:
    print("=" * 68)
    print("예제 2: 1차원 열전도 방정식 (1D Heat Equation, Data-Free PINN)")
    print("=" * 68)

    problem = HeatEquation1D(alpha=0.1)
    model, _ = train_heat_pinn(problem, epochs=4000, lr=1e-3)

    output_path = PROJECT_ROOT / "results" / "02_heat_equation_1d.png"
    rel_l2, max_err = plot_results(problem, model, output_path)

    print("\n" + "=" * 68)
    print(f"최종 상대 L2 오차 (Relative L2 Error): {rel_l2:.5%}")
    print(f"최대 절대 오차    (Max Abs Error)    : {max_err:.6f}")
    print(f"결과 그래프 저장 완료: {output_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
