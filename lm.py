# Lagrange multiplier method with helicity conservation.
from firedrake import *
import csv
import os
import sys
import numpy as np
from config import CONFIG
from common import apply_jump_schedule, build_initial_condition, build_mesh_and_spaces

# ANSI colours used by the time-stepping prints.
RED   = "\033[91m%s\033[0m"
GREEN = "\033[92m%s\033[0m"
BLUE  = "\033[94m%s\033[0m"

# ============================================================
# Parameters
# ============================================================
output = CONFIG.output
ic = CONFIG.ic
is_e3 = CONFIG.is_e3
bc = CONFIG.bc
scheme_name = "lm"
output_dir = f"{scheme_name}-{ic.lower()}"

if bc == "closed":
    periodic = False
elif bc == "periodic":
    periodic = True
else:
    raise ValueError(f"unknown bc: {bc}")

os.makedirs(output_dir, exist_ok=True)
(Lx, Ly, Lz), (Nx, Ny, Nz) = CONFIG.domain
dirichlet_ids = CONFIG.dirichlet_ids

order = CONFIG.order
tau = Constant(CONFIG.tau)
t = Constant(0)
dt = Constant(CONFIG.dt_init)

T = CONFIG.T

dt_init = float(dt)

# delta_energy threshold at which the two-LM scheme hands off to single-LM
two_to_single_LM_de = CONFIG.two_to_single_LM_de
lm_handoff_rise_rtol = CONFIG.lm_handoff_rise_rtol
lm_handoff_rise_steps = CONFIG.lm_handoff_rise_steps

jump_after_steps = CONFIG.jump_after_steps
jump_dt = CONFIG.jump_dt
jump_tau = CONFIG.jump_tau

# ============================================================
# Mesh / function spaces
# ============================================================
mesh, spaces = build_mesh_and_spaces(CONFIG, include_real=True)
Vg, Vg_, Vc, Vd, Vn, VR = (
    spaces[name] for name in ("Vg", "Vg_", "Vc", "Vd", "Vn", "VR")
)

# ============================================================
# Initial condition
# ============================================================
B_init, guide_field, _B_b, k_sign = build_initial_condition(mesh, CONFIG)

# B_init already contains the E3 harmonic background on the periodic domain.
# Since that component cannot be represented by curl(A), retain it explicitly
# in the B = curl(A) + B_h relation and in H_G.
harmonic_field = guide_field
evolved_harmonic = as_vector([0.0, 0.0, 0.0])
if is_e3 and periodic:
    guide_field = as_vector([0.0, 0.0, 0.0])
    evolved_harmonic = harmonic_field

# ============================================================
# Mixed unknowns
#   two-LM:    [B, A, E, j, λ_e, λ_m]   energy + helicity LMs
#   single-LM: [B, A, E, j, λ_m]        helicity LM only
# ============================================================
Z = MixedFunctionSpace([Vd, Vc, Vc, Vc, VR, VR])
z      = Function(Z)
z_prev = Function(Z)
z_test = TestFunction(Z)
(B,  A,  E,  j,  lmbda_e,  lmbda_m)  = split(z)
(Bt, At, Et, jt, lmbda_et, lmbda_mt) = split(z_test)
(Bp, Ap, Ep, jp, lmbda_ep, lmbda_mp) = split(z_prev)

Z_s = MixedFunctionSpace([Vd, Vc, Vc, Vc, VR])
z_s      = Function(Z_s)
z_s_prev = Function(Z_s)
z_s_test = TestFunction(Z_s)
(B_s,  A_s,  E_s,  j_s,  lmbda_m_s)  = split(z_s)
(B_st, A_st, E_st, j_st, lmbda_m_st) = split(z_s_test)
(B_sp, A_sp, E_sp, j_sp, lmbda_m_sp) = split(z_s_prev)

def form_energy(B):
    B_total = B + guide_field
    return dot(B_total, B_total)

def form_dissipation(B, j):
    B_total = B + guide_field
    return 2 * tau * inner(cross(B_total, j), cross(B_total, j))

def form_helicity(A, B):
    """
    Time-discrete LM constraint integrand. The constraint
        (1/dt) * (form_helicity(A,B) - form_helicity(Ap,Bp), λ_m_test) * dx
    enforces conservation of the integrated helicity functional.

    closed:    A·B                              (helicity)
    periodic:  A·(B + harmonic)                 (generalized helicity)
    """
    if is_e3 and periodic:
        return dot(A, B + harmonic_field)
    elif periodic:
        harmonic = Function(Vd)
        harmonic.project(B - curl(A))
        return dot(A, B + harmonic)
    else:
        return dot(A, B)

# Two-LM weak form.  For periodic E3, B is complete and satisfies
# B = curl(A) + B_h; harmonic_field is zero for the other configurations.
B_total = B + guide_field
# Reduced derivative dH/dA.  For E3 periodic,
# H_G = ∫ A·(B_total + B_h) = ∫ A·(curl(A) + 2B_h).
helicity_gradient = 2 * (B + harmonic_field) if is_e3 and periodic else 2 * B
F = (
      inner(B - evolved_harmonic, Bt) * dx
    - inner(curl(A), Bt) * dx
    + inner((A - Ap)/dt, At) * dx
    + inner(E, At) * dx
    + lmbda_m * inner(helicity_gradient, At) * dx
    + 2 * lmbda_e * inner(B_total, curl(At)) * dx

    + inner(E, Et) * dx
    + tau * inner(cross(cross(j, B_total), B_total), Et) * dx

    + inner(j, jt) * dx
    - inner(B, curl(jt)) * dx

    # energy law:    d/dt ‖B‖² + 2τ ‖j×B‖² = 0
    + 1/dt * inner(form_energy(B) - form_energy(Bp), lmbda_et) * dx
    + inner(form_dissipation(B, j), lmbda_et) * dx
    # helicity law: dH/dt = 0
    + 1/dt * inner(form_helicity(A, B) - form_helicity(Ap, Bp), lmbda_mt) * dx
)

# Single-LM weak form
B_total_s = B_s + guide_field
helicity_gradient_s = 2 * (B_s + harmonic_field) if is_e3 and periodic else 2 * B_s
F_s = (
      inner(B_s - evolved_harmonic, B_st) * dx
    - inner(curl(A_s), B_st) * dx
    + inner((A_s - A_sp)/dt, A_st) * dx
    + inner(E_s, A_st) * dx
    + lmbda_m_s * inner(helicity_gradient_s, A_st) * dx

    + inner(E_s, E_st) * dx
    + tau * inner(cross(cross(j_s, B_total_s), B_total_s), E_st) * dx

    + inner(j_s, j_st) * dx
    - inner(B_s, curl(j_st)) * dx

    + 1/dt * inner(form_helicity(A_s, B_s) - form_helicity(A_sp, B_sp), lmbda_m_st) * dx
)

# ============================================================
# Boundary conditions: homogeneous Dirichlet on all physical components.
# ============================================================
bcs   = [DirichletBC(Z.sub(index),   0, subdomain)
         for index in range(len(Z)-2)   for subdomain in dirichlet_ids]
bcs_s = [DirichletBC(Z_s.sub(index), 0, subdomain)
         for index in range(len(Z_s)-1) for subdomain in dirichlet_ids]

lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_monitor": None,
    "ksp_monitor": None,
    "ksp_type": "preonly",
    "pc_type":  "lu",
    "pc_factor_mat_solver_type": "mumps",
}
sp = None  # Firedrake default

# ============================================================
# Setup: rename and initialise solution fields
# ============================================================
(B_, A_, E_, j_, lmbda_e_, lmbda_m_) = z.subfunctions
B_.rename("MagneticField")
E_.rename("ElectricField")
A_.rename("MagneticPotential")
j_.rename("Current")

# ============================================================
# Initial-condition projection: divergence-free B with prescribed
# normal trace.
# ============================================================
def project_initial_conditions(B_init):
    Zp = MixedFunctionSpace([Vd, Vn])
    zp = Function(Zp)
    (Bv, p) = split(zp)

    bcp = [DirichletBC(Zp.sub(0), 0, subdomain) for subdomain in dirichlet_ids]

    L = (
          0.5*inner(Bv, Bv)*dx
        - inner(B_init, Bv)*dx
        - inner(p, div(Bv))*dx
        )
    Fp = derivative(L, zp, TestFunction(Zp))
    spp = {
        "mat_type": "nest",
        "snes_type": "ksponly",
        "snes_monitor": None,
        "ksp_monitor": None,
        "ksp_max_it": 1000,
        "ksp_norm_type": "preconditioned",
        "ksp_type": "minres",
        "pc_type":  "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_pc_type": "cholesky",
        "fieldsplit_pc_factor_mat_solver_type": "mumps",
        "ksp_atol": 1.0e-5,
        "ksp_rtol": 1.0e-5,
        "ksp_minres_nutol": 1E-8,
        "ksp_convergence_test": "skip",
    }
    gamma = Constant(1E5)
    Up = 0.5*(inner(Bv, Bv) + inner(div(Bv) * gamma, div(Bv)) + inner(p * (1/gamma), p))*dx
    Jp = derivative(derivative(Up, zp), zp)
    solve(Fp == 0, zp, bcp, Jp=Jp, solver_parameters=spp,
          options_prefix="B_init_div_free_projection")
    return zp.subfunctions[0]


def build_linear_solver(a, L, u_sol, bcs, aP=None, solver_parameters=None, options_prefix=None):
    problem = LinearVariationalProblem(a, L, u_sol, bcs=bcs, aP=aP)
    solver  = LinearVariationalSolver(problem,
                                      solver_parameters=solver_parameters,
                                      options_prefix=options_prefix)
    return solver

def build_nonlinear_solver(F, z_sol, bcs, Jp=None, solver_parameters=None, options_prefix=None):
    problem = NonlinearVariationalProblem(F, z_sol, bcs, Jp=Jp)
    solver  = NonlinearVariationalSolver(problem,
                                         solver_parameters=solver_parameters,
                                         options_prefix=options_prefix)
    return solver

def potential_solver_direct(B_func):
    """
    Solve curl(A) = B_func with A ∈ H_0(curl).
    """
    Afunc = Function(Vc)
    v = TestFunction(Vc)
    F_curl = inner(curl(Afunc), curl(v)) * dx - inner(B_func, curl(v)) * dx

    sp_helicity = {
        "ksp_type": "gmres",
        "pc_type":  "ilu",
    }
    bcs_curl = [DirichletBC(Vc, 0, sub) for sub in dirichlet_ids]
    solver = build_nonlinear_solver(F_curl, Afunc, bcs_curl,
                                    solver_parameters=sp_helicity,
                                    options_prefix="solver_curlcurl")
    solver.solve()
    return Afunc

# Initialise with projected B and consistent A_f
proj_B0 = project_initial_conditions(B_init)
z_prev.sub(0).project(proj_B0)
z_prev.sub(1).project(potential_solver_direct(proj_B0))
z.assign(z_prev)

if output:
    pvd = VTKFile(f"{output_dir}/parker.pvd")
    pvd.write(*z.subfunctions, time=float(t))


def helicity_solver_setup():
    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    u_sol = Function(Vc)

    a = inner(curl(u), curl(v)) * dx
    L = inner(B, curl(v)) * dx                  # solve for A, target = B
    beta = Constant(0.1)
    Jp_curl = a + inner(beta * u, v) * dx
    bcs_curl = [DirichletBC(Vc, 0, sub) for sub in dirichlet_ids]
    sparams = {
        "snes_type": "ksponly",
        "ksp_type":  "minres",
        "ksp_max_it": 1000,
        "pc_type":   "cholesky",
        "ksp_norm_type": "preconditioned",
        "ksp_minres_nutol": 1E-8,
    }
    solver = build_linear_solver(a, L, u_sol, bcs_curl, Jp_curl, sparams,
                                 options_prefix="helicity")
    return solver

helicity_solver = helicity_solver_setup()

def riesz_map(functional):
    function = Function(functional.function_space().dual())
    with functional.dat.vec as x_, function.dat.vec as y_:
        helicity_solver.snes.ksp.pc.apply(x_, y_)
    return function

def compute_potential(B):
    helicity_solver.solve()
    problem = helicity_solver._problem
    if helicity_solver.snes.ksp.getResidualNorm() > 0.01:
        r     = assemble(problem.F, bcs=problem.bcs)
        rstar = r.riesz_representation(riesz_map=riesz_map, bcs=problem.bcs)
        c     = assemble(action(r, problem.u)) / assemble(action(r, rstar))
        ulft  = Function(Vc, name="u_lifted")
        ulft.assign(problem.u - c * rstar)
        A = ulft
    else:
        A = problem.u
    diff = norm(curl(A) - B, "L2")
    if mesh.comm.rank == 0:
        print(f"[compute_potential] ||curl(A) - B||_L2 = {diff:.8e}", flush=True)
    A_ = Function(Vc, name="MagneticPotential")
    A_.project(A)
    return A_


pb = NonlinearVariationalProblem(F, z, bcs=bcs)
time_stepper = NonlinearVariationalSolver(pb, solver_parameters=sp)

pb_s = NonlinearVariationalProblem(F_s, z_s, bcs=bcs_s)
time_stepper_s = NonlinearVariationalSolver(pb_s, solver_parameters=sp)


def compute_helicity(A_func, B_func):
    """
    Diagnostic helicity (matches what the LM enforces in form_helicity).
    """
    if is_e3 and periodic:
        return assemble(inner(A_func, B_func + harmonic_field) * dx)
    elif periodic:
        harmonic = Function(Vd)
        harmonic.project(B_func - curl(A_func))
        return assemble(inner(A_func, B_func + harmonic) * dx)
    else:
        return assemble(inner(A_func, B_func) * dx)


def compute_divB(B_func):
    return norm(div(B_func), "L2")


def compute_energy(B_func, A_func):
    if is_e3:
        return assemble(inner(B_func, B_func) * dx)
    elif periodic:
        harmonic = Function(Vd)
        harmonic.project(B_func - curl(A_func))
        return assemble(inner(B_func - harmonic, B_func - harmonic) * dx)
    else:
        return assemble(inner(B_func, B_func) * dx)


def compute_free_energy(B_func):
    perturbation = B_func - harmonic_field if is_e3 and periodic else B_func
    return assemble(inner(perturbation, perturbation) * dx)


def compute_background_energy():
    """Energy of the E3 harmonic field; zero when no E3 field is present."""
    if not is_e3:
        return 0.0
    return assemble(inner(harmonic_field, harmonic_field) * dx(domain=mesh))


def compute_lamb(j, B):
    eps = 1e-10
    B_total = B + guide_field
    lamb = Function(Vg_).interpolate(dot(j, B_total)/(dot(B_total, B_total) + eps))
    with lamb.dat.vec_ro as v:
        _, max_val = v.max()
        _, min_val = v.min()
    if abs(min_val) < eps:
        return eps
    else:
        return max_val/min_val


def compute_xi_max(j, B):
    eps = 1e-10
    B_total = B + guide_field
    xi = Function(Vg).interpolate(cross(j, B_total)/(dot(B_total, B_total) + eps))
    with xi.dat.vec_ro as v:
        _, max_val = v.max()
        _, min_val = v.min()
    if abs(min_val) < eps:
        return eps
    else:
        return max_val/min_val


# ============================================================
# Reproducibility: print + persist all run parameters
# ============================================================
def write_params(path, params, header=""):
    import datetime, subprocess
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git = "unknown"
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [f"# {header}", f"# time: {ts}", f"# git:  {git}", ""]
    width = max(len(k) for k in params) + 2
    for k, v in params.items():
        lines.append(f"{k:<{width}} {v}")
    text = "\n".join(lines) + "\n"
    if mesh.comm.rank == 0:
        print(text, flush=True)
        with open(path, "w") as f:
            f.write(text)

write_params(f"{output_dir}/param.txt", {
    "script":              "lm.py",
    "method":              "Lagrange multiplier (energy + helicity)",
    "ic":                  ic,
    "k_sign":              "n/a" if not is_e3 else k_sign,
    "bc":                  bc,
    "periodic":            periodic,
    "order":               order,
    "Lx,Ly,Lz":            f"{Lx},{Ly},{Lz}",
    "Nx,Ny,Nz":            f"{Nx},{Ny},{Nz}",
    "tau":                 float(tau),
    "dt_init":             dt_init,
    "two_to_single_LM_de": two_to_single_LM_de,
    "lm_handoff_rise_rtol": lm_handoff_rise_rtol,
    "lm_handoff_rise_steps": lm_handoff_rise_steps,
    "jump_after_steps":    jump_after_steps,
    "jump_dt":             jump_dt,
    "jump_tau":            jump_tau,
    "T":                   T,
    "Vc.dim":              Vc.dim(),
    "Vd.dim":              Vd.dim(),
    "Vn.dim":              Vn.dim(),
    "VR.dim":              VR.dim(),
    "Z.dim":               Z.dim(),
    "Z_s.dim":             Z_s.dim(),
    "solver_parameters":   "None (Firedrake default direct LU/MUMPS)",
    "output_dir":          f"{output_dir}/",
}, header="lm.py")

# ============================================================
# Time stepping
# ============================================================
data_filename = f"{output_dir}/data.csv"
fieldnames = ["t", "helicity", "energy", "free_energy",
              "background_energy", "divB", "lamb", "xi"]
helicity_print_label = "generalized_helicity" if periodic else "helicity"
if mesh.comm.rank == 0:
    with open(data_filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

helicity    = compute_helicity(z.sub(1), z.sub(0))
divB        = compute_divB(z.sub(0))
energy      = compute_energy(z.sub(0), z.sub(1))
free_energy = compute_free_energy(z.sub(0))
background_energy = compute_background_energy()
lamb        = compute_lamb(z.sub(3), z.sub(0))
xi          = compute_xi_max(z.sub(3), z.sub(0))
helicity_initial = helicity

if mesh.comm.rank == 0:
    print(f"Initial {helicity_print_label} = {helicity_initial:.10e}", flush=True)
    row = {
        "t": float(t),
        "helicity":    float(helicity),
        "energy":      float(energy),
        "free_energy": float(free_energy),
        "background_energy": float(background_energy),
        "divB":        float(divB),
        "lamb":        float(lamb),
        "xi":          float(xi),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
        printed_row = row.copy()
        if periodic:
            printed_row[helicity_print_label] = printed_row.pop("helicity")
        print(f"{printed_row}")

delta_energy = 1.0
previous_delta_energy = None
delta_energy_rise_count = 0
automatic_lm_handoff = False
timestep = 0
z_s_init = False
z_prev.assign(z)
while (float(t) < float(T) - 1.0e-9):
    if float(t) + float(dt) > float(T):
        dt.assign(float(T) - float(t))
    if float(dt) <= 1e-14:
        break

    # Which scheme is active this step. delta_energy is only updated in the
    # two-LM branch, so once it drops below the threshold we stay single-LM.
    two_lm = (delta_energy > two_to_single_LM_de) and not automatic_lm_handoff

    # One-time handoff: seed the single-LM state from the two-LM state.
    # λ_e (z.sub(4)) is dropped; λ_m moves from z.sub(5) into z_s.sub(4).
    if (not two_lm) and (not z_s_init):
        z_s_prev.sub(0).assign(z.sub(0))
        z_s_prev.sub(1).assign(z.sub(1))
        z_s_prev.sub(2).assign(z.sub(2))
        z_s_prev.sub(3).assign(z.sub(3))
        z_s_prev.sub(4).assign(z.sub(5))
        if mesh.comm.rank == 0:
            print("initializing for z_s is done")
        z_s_init = True

    if mesh.comm.rank == 0:
        print(RED % f"Solving for t = {float(t):.4f} (+dt={float(dt):g}), "
                    f"dofs = {Z.dim()}, ic = {ic}, bc = {bc}, T = {T}, "
                    f"scheme = {'two-LM' if two_lm else 'single-LM'}", flush=True)

    # ---- solve this step: single plain solve(), no retry ----
    if two_lm:
        z.assign(z_prev)
        solver = time_stepper
    else:
        z_s.assign(z_s_prev)
        solver = time_stepper_s
    solver.solve()
    iters = solver.snes.getIterationNumber()

    # ---- step accepted: advance time by the dt actually used in the solve ----
    dt_used = float(dt)
    t.assign(float(t) + dt_used)

    if two_lm:
        helicity     = compute_helicity(z.sub(1), z.sub(0))
        divB         = compute_divB(z.sub(0))
        energy       = compute_energy(z.sub(0), z.sub(1))
        free_energy  = compute_free_energy(z.sub(0))
        delta_energy = 1/dt_used * (compute_energy(z_prev.sub(0), z_prev.sub(1)) - energy)
        if previous_delta_energy is not None:
            rise_tolerance = lm_handoff_rise_rtol * max(abs(previous_delta_energy), 1e-14)
            if delta_energy > previous_delta_energy + rise_tolerance:
                delta_energy_rise_count += 1
            else:
                delta_energy_rise_count = 0

            if delta_energy_rise_count >= lm_handoff_rise_steps:
                automatic_lm_handoff = True
                if mesh.comm.rank == 0:
                    print(
                        "delta_energy has risen for "
                        f"{delta_energy_rise_count} consecutive steps; "
                        "switching permanently to single-LM on the next step",
                        flush=True,
                    )
        previous_delta_energy = delta_energy
        lamb         = compute_lamb(z.sub(3), z.sub(0))
        xi           = compute_xi_max(z.sub(3), z.sub(0))
        if mesh.comm.rank == 0:
            print(GREEN % f"{delta_energy}")
            print(BLUE  % f"lmbda_e = {norm(z.sub(4))}, lmbda_m = {norm(z.sub(5))}")
    else:
        helicity    = compute_helicity(z_s.sub(1), z_s.sub(0))
        divB        = compute_divB(z_s.sub(0))
        energy      = compute_energy(z_s.sub(0), z_s.sub(1))
        free_energy = compute_free_energy(z_s.sub(0))
        lamb        = compute_lamb(z_s.sub(3), z_s.sub(0))
        xi          = compute_xi_max(z_s.sub(3), z_s.sub(0))
        if mesh.comm.rank == 0:
            print(BLUE % f"lmbda_m = {norm(z_s.sub(4))}")
        # Sync z_s state into z so the output PVD keeps the same set of
        # functions across the two-LM → single-LM transition. λ_e (z.sub(4))
        # is no longer evolved, so leave it as-is; λ_m moves from z_s.sub(4)
        # into z.sub(5).
        for i in range(4):
            z.sub(i).assign(z_s.sub(i))
        z.sub(5).assign(z_s.sub(4))

    H_err = abs(helicity - helicity_initial)
    H_err_label = "|H_g - H_g0|" if periodic else "|H - H_0|"
    if mesh.comm.rank == 0:
        print(f"Solved t = {float(t):.4f}, dt = {dt_used:g}, Newton iters = {iters}", flush=True)
        row = {
            "t": float(t),
            "helicity":    float(helicity),
            "energy":      float(energy),
            "free_energy": float(free_energy),
            "background_energy": float(background_energy),
            "divB":        float(divB),
            "lamb":        float(lamb),
            "xi":          float(xi),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            printed_row = row.copy()
            if periodic:
                printed_row[helicity_print_label] = printed_row.pop("helicity")
            print(f"{printed_row}, {H_err_label} = {H_err:.4e}")

    if output:
        pvd.write(*z.subfunctions, time=float(t))
    z_prev.assign(z)
    z_s_prev.assign(z_s)
    timestep += 1

    # ---- choose dt for the NEXT step: fixed dt, then jump to a larger dt + drop tau ----
    apply_jump_schedule(timestep, dt, tau, CONFIG)
