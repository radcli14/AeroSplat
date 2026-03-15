"""
Flat-plate flow test: uniform inflow on the left, no-slip plate in the centre.
Runs the optimizer and produces a summary plot.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from aerosplat import *

# ── Problem definition ────────────────────────────────────────────────────────
INFLOW_SPEED = 10.0
N_SPLATS     = 6
N_ITERS      = 500

source_boundary = LineBoundary(point0=[0, -1], point1=[0,  1], velocity=[INFLOW_SPEED, 0])
plate_boundary  = LineBoundary(point0=[1, -0.5], point1=[1, 0.5], velocity=[0, 0])

problem = AeroSplatProblem(
    domain_x=[0, 2],
    domain_y=[-1, 1],
    boundaries=[source_boundary, plate_boundary],
)
print(f"Problem: domain {problem.domain_x} × {problem.domain_y}, "
      f"velocity scale = {problem.velocity_scale}")

# ── Initial solution ──────────────────────────────────────────────────────────
np.random.seed(42)
initial_solution = AeroSplatSolution(problem.domain, spawn=N_SPLATS)
print(f"Initial splats: {N_SPLATS}, params per splat: "
      f"{len(initial_solution.splats[0].as_normalized_array(problem.domain))}")

# ── Optimise ──────────────────────────────────────────────────────────────────
optimizer = AeroSplatOptimizer(problem, initial_solution,
                               points_per_boundary=15,
                               points_in_volume=75)

init_loss = sum(optimizer.history[0])
print(f"Initial loss: {init_loss:.4f}  "
      f"(boundary={optimizer.history[0][0]:.4f}, volume={optimizer.history[0][1]:.4f})")

for k in range(N_ITERS):
    optimizer.iterate()
    if (k + 1) % 100 == 0:
        total = sum(optimizer.history[-1])
        print(f"  iter {k+1:4d}  loss={total:.6f}")

final_loss = sum(optimizer.history[-1])
print(f"Final   loss: {final_loss:.4f}  "
      f"(boundary={optimizer.history[-1][0]:.4f}, volume={optimizer.history[-1][1]:.4f})")

# ── Evaluation grid ───────────────────────────────────────────────────────────
nx, ny = 81, 41
grid   = problem.point_grid(nx, ny)
xs     = np.array([grid[0, k, 0] for k in range(nx)])
ys     = np.array([grid[j, 0, 1] for j in range(ny)])

def flow_fields(sol):
    vf  = sol.velocity_on_grid(grid)
    mag = np.linalg.norm(vf, axis=-1)
    return vf, mag

init_vf,  init_mag  = flow_fields(initial_solution)
final_sol = optimizer.solutions[-1]
final_vf, final_mag = flow_fields(final_sol)

history = np.array(optimizer.history)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

CMAP   = "turbo"
LEVELS = 20

def draw_boundaries(ax):
    ax.axvline(x=0,  ymin=0, ymax=1, color="white", lw=1.5, ls="--", alpha=0.7, label="inflow")
    ax.plot([1, 1], [-0.5, 0.5], color="black", lw=3, label="plate")

# — Initial velocity magnitude —
ax0 = fig.add_subplot(gs[0, 0])
cf  = ax0.contourf(xs, ys, init_mag, levels=LEVELS, cmap=CMAP)
fig.colorbar(cf, ax=ax0, label="|v|")
draw_boundaries(ax0)
ax0.set_title("Initial |velocity|")
ax0.set_xlabel("x"); ax0.set_ylabel("y")

# — Final velocity magnitude —
ax1 = fig.add_subplot(gs[0, 1])
cf  = ax1.contourf(xs, ys, final_mag, levels=LEVELS, cmap=CMAP)
fig.colorbar(cf, ax=ax1, label="|v|")
draw_boundaries(ax1)
ax1.set_title("Final |velocity|")
ax1.set_xlabel("x")

# — Final streamlines —
ax2 = fig.add_subplot(gs[0, 2])
speed_norm = final_mag / (final_mag.max() + 1e-12)
ax2.streamplot(xs, ys, final_vf[:, :, 0], final_vf[:, :, 1],
               color=final_mag, cmap=CMAP, linewidth=1.0, density=1.5)
draw_boundaries(ax2)
ax2.set_xlim(problem.domain_x); ax2.set_ylim(problem.domain_y)
ax2.set_title("Final streamlines")
ax2.set_xlabel("x")

# — Loss history —
ax3 = fig.add_subplot(gs[1, 0])
ax3.semilogy(history[:, 0], label="boundary", alpha=0.8)
ax3.semilogy(history[:, 1], label="volume",   alpha=0.8)
ax3.semilogy(history.sum(axis=1), label="total", lw=2)
ax3.set_title("Loss history")
ax3.set_xlabel("iteration"); ax3.set_ylabel("loss (log)")
ax3.legend(fontsize=8)

# — Splat positions: initial vs final —
ax4 = fig.add_subplot(gs[1, 1])
init_pos  = np.array([s.position for s in initial_solution.splats])
final_pos = np.array([s.position for s in final_sol.splats])
ax4.scatter(init_pos[:,  0], init_pos[:,  1], marker="o", label="initial", zorder=3)
ax4.scatter(final_pos[:, 0], final_pos[:, 1], marker="s", label="final",   zorder=3)
for i, f in zip(init_pos, final_pos):
    ax4.annotate("", xy=f, xytext=i,
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
draw_boundaries(ax4)
ax4.set_xlim(-0.2, 2.2); ax4.set_ylim(-1.2, 1.2)
ax4.set_title("Splat positions")
ax4.set_xlabel("x"); ax4.set_ylabel("y")
ax4.legend(fontsize=8)

# — Boundary velocity error at final solution —
ax5 = fig.add_subplot(gs[1, 2])
t_vals = np.linspace(0, 1, 50)
for bnd, lbl, col in [(source_boundary, "inflow (x=0)", "C0"),
                      (plate_boundary,  "plate  (x=1)", "C1")]:
    pts   = np.array([bnd.point_at(t) for t in t_vals])
    vels  = np.array([final_sol.velocity_at(p) for p in pts])
    errs  = np.linalg.norm(vels - bnd.velocity[:2], axis=1)
    ax5.plot(t_vals, errs, label=lbl, color=col)
ax5.set_title("Boundary velocity error (final)")
ax5.set_xlabel("boundary parameter t"); ax5.set_ylabel("|v_computed − v_bc|")
ax5.legend(fontsize=8)

fig.suptitle(f"AeroSplat flat-plate test  |  {N_SPLATS} splats, {N_ITERS} iterations\n"
             f"loss {init_loss:.4f} → {final_loss:.4f}  "
             f"({100*(init_loss-final_loss)/init_loss:.1f}% reduction)",
             fontsize=11)

out_path = "/home/user/AeroSplat/flat_plate_result.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
