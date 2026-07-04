# Lagrange multiplier method with helicity conservation.
from firedrake import *
import csv
import os
import sys
import numpy as np

# ANSI colours used by the time-stepping prints.
RED   = "\033[91m%s\033[0m"
GREEN = "\033[92m%s\033[0m"
BLUE  = "\033[94m%s\033[0m"

# ============================================================
# Parameters
# ============================================================
output = True
ic = os.environ.get("IC", "E3")  # hopf, E3, or E3-positive
is_e3 = ic in ("E3", "E3-positive")
bc = "closed"   # "closed" or "periodic"
scheme_name = "lm"
output_dir = f"{scheme_name}-{ic.lower()}"

if bc == "closed":
    periodic = False
elif bc == "periodic":
    periodic = True
else:
    raise ValueError(f"unknown bc: {bc}")

os.makedirs(output_dir, exist_ok=True)

if ic == "hopf":
    Lx, Ly, Lz = 8, 8, 20
    Nx, Ny, Nz = 8, 8, 10
elif is_e3:
    Lx, Ly, Lz = 8, 8, 48
    Nx, Ny, Nz = 4, 4, 24
else:
    raise ValueError(f"unknown initial condition: {ic}")

if periodic:
    dirichlet_ids = ("on_boundary",)
else:
    dirichlet_ids = ("on_boundary", "top", "bottom")

order = 1
tau = Constant(1)  
t = Constant(0)
if is_e3:
    dt = Constant(0.1)
else:
    dt = Constant(1)

T = 10000

dt_init = float(dt)

# delta_energy threshold at which the two-LM scheme hands off to single-LM
two_to_single_LM_de = 9e-5

jump_after_steps = 100
jump_dt          = 100 if is_e3 else 100.0
jump_tau         = 1

# ============================================================
# Mesh / function spaces
# ============================================================
base = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)
mesh = ExtrudedMesh(base, Lz, 1, periodic=periodic)
mesh.coordinates.dat.data[:, 0] -= Lx/2
mesh.coordinates.dat.data[:, 1] -= Ly/2
mesh.coordinates.dat.data[:, 2] -= Lz/2

Vg  = VectorFunctionSpace(mesh, "Q",   order)
Vg_ = FunctionSpace(mesh,        "Q",   order)
Vc  = FunctionSpace(mesh,        "NCE", order)
Vd  = FunctionSpace(mesh,        "NCF", order)
Vn  = FunctionSpace(mesh,        "DQ",  order-1)
VR  = FunctionSpace(mesh,        "R",   0)

# ============================================================
# Initial condition
# ============================================================
(X0, Y0, Z0) = x = SpatialCoordinate(mesh)

if ic == "hopf":
    w1 = 3
    w2 = 2
    s = 1
    deno = 1 + dot(x, x)
    coeff = 4*sqrt(s)/((pi * deno * deno * deno)*sqrt(w1**2+w2**2))
    B_init = as_vector([ coeff*2*(w2*Y0 - w1*X0*Z0),
                        -coeff*2*(w2*X0 + w1*Y0*Z0),
                         coeff*w1*(-1 + X0**2 + Y0**2 - Z0**2)])

elif is_e3:
    x_c = [1, -1, 1, -1, 1, -1]
    y_c = 0.0
    z_c = [-20, -12, -4, 4, 12, 20]
    a = sqrt(2)
    # strength of twist
    k = 5.0
    k_sign = [1, -1, 1, -1, 1, -1] if ic == "E3" else [1] * 6
    l = 2.0
    B_0 = 1.0

    B_x = 0.0
    B_y = 0.0
    B_z = B_0

    # background magnetic field
    B_b = as_vector([0.0, 0.0, B_0])

    for i in range(6):
        coeff = exp(
            -((X0 - x_c[i])**2 / (a**2))
            -((Y0 - y_c)**2 / (a**2))
            -((Z0 - z_c[i])**2 / (l**2))
        )
        B_x += coeff * ((2.0 * k * k_sign[i] * B_0 / a) * (-(Y0 - y_c)))
        B_y += coeff * ((2.0 * k * k_sign[i] * B_0 / a) * ((X0 - x_c[i])))

    B_init = as_vector([B_x, B_y, B_z]) - B_b

# Diagnostic guide field only.  The weak forms below intentionally continue
# to evolve the fluctuation B and preserve the original standard helicity.
guide_field = B_b if is_e3 else as_vector([0.0, 0.0, 0.0])

# ============================================================
# Mixed unknowns
#   two-LM:    [B, u, A, E, j, λ_e, λ_m]   energy + helicity LMs
#   single-LM: [B, u, A, E, j, λ_m]        helicity LM only
# ============================================================
Z = MixedFunctionSpace([Vd, Vd, Vc, Vc, Vc, VR, VR])
z      = Function(Z)
z_prev = Function(Z)
z_test = TestFunction(Z)
(B,  u,  A,  E,  j,  lmbda_e,  lmbda_m)  = split(z)
(Bt, ut, At, Et, jt, lmbda_et, lmbda_mt) = split(z_test)
(Bp, up, Ap, Ep, jp, lmbda_ep, lmbda_mp) = split(z_prev)

Z_s = MixedFunctionSpace([Vd, Vd, Vc, Vc, Vc, VR])
z_s      = Function(Z_s)
z_s_prev = Function(Z_s)
z_s_test = TestFunction(Z_s)
(B_s,  u_s,  A_s,  E_s,  j_s,  lmbda_m_s)  = split(z_s)
(B_st, u_st, A_st, E_st, j_st, lmbda_m_st) = split(z_s_test)
(B_sp, u_sp, A_sp, E_sp, j_sp, lmbda_m_sp) = split(z_s_prev)

def form_energy(B):
    return dot(B, B)

def form_dissipation(B, j):
    return 2 * tau * inner(cross(B, j), cross(B, j))

def form_helicity(A, B):
    """
    Time-discrete LM constraint integrand. The constraint
        (1/dt) * (form_helicity(A,B) - form_helicity(Ap,Bp), λ_m_test) * dx
    enforces conservation of the integrated helicity functional.

    closed:    A·B                              (standard helicity)
    periodic:  A·(B + harmonic)                 (generalised helicity)
    """
    if periodic:
        harmonic = Function(Vd)
        harmonic.project(B - curl(A))
        return dot(A, B + harmonic)
    else:
        return dot(A, B)

# Two-LM weak form — B equation is (B, Bt) - (curl A, Bt) = 0 so B = curl A.
F = (
      inner(B, Bt) * dx
    - inner(curl(A), Bt) * dx
    + inner((A - Ap)/dt, At) * dx
    + inner(E, At) * dx
    + 2 * lmbda_m * inner(B, At) * dx       # LM force: δH/δA = 2B
    + 2 * lmbda_e * inner(B, curl(At)) * dx # LM force: δE/δA = 2 curl B

    + inner(E, Et) * dx
    + inner(cross(u, B), Et) * dx

    + inner(u, ut) * dx
    - tau * inner(cross(j, B), ut) * dx

    + inner(j, jt) * dx
    - inner(B, curl(jt)) * dx

    # energy law:    d/dt ‖B‖² + 2τ ‖j×B‖² = 0
    + 1/dt * inner(form_energy(B) - form_energy(Bp), lmbda_et) * dx
    + inner(form_dissipation(B, j), lmbda_et) * dx
    # helicity law: dH/dt = 0
    + 1/dt * inner(form_helicity(A, B) - form_helicity(Ap, Bp), lmbda_mt) * dx
)

# Single-LM weak form
F_s = (
      inner(B_s, B_st) * dx
    - inner(curl(A_s), B_st) * dx
    + inner((A_s - A_sp)/dt, A_st) * dx
    + inner(E_s, A_st) * dx
    + 2 * lmbda_m_s * inner(B_s, A_st) * dx

    + inner(E_s, E_st) * dx
    + inner(cross(u_s, B_s), E_st) * dx

    + inner(u_s, u_st) * dx
    - tau * inner(cross(j_s, B_s), u_st) * dx

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
(B_, u_, A_, E_, j_, lmbda_e_, lmbda_m_) = z.subfunctions
B_.rename("MagneticField")
E_.rename("ElectricField")
u_.rename("Velocity")
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
z_prev.sub(2).project(potential_solver_direct(proj_B0))
z.assign(z_prev)

B_recover = Function(Vd, name="RecoveredMagneticField")
if output:
    pvd = VTKFile(f"{output_dir}/parker.pvd")
    pvd.write(*z.subfunctions, time=float(t))
    if is_e3:
        pvd1 = VTKFile(f"{output_dir}/recover.pvd")
        B_recover.project(z.sub(0) + B_b)
        pvd1.write(B_recover, time=float(t))


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
    if periodic:
        harmonic = Function(Vd)
        harmonic.project(B_func - curl(A_func))
        return assemble(inner(A_func, B_func + harmonic) * dx)
    else:
        return assemble(inner(A_func, B_func) * dx)


def compute_relative_helicity(A_func, B_func):
    """Relative-helicity diagnostic; this is not imposed by the LM scheme."""
    return assemble(inner(A_func, B_func + 2 * guide_field) * dx)


def compute_divB(B_func):
    return norm(div(B_func), "L2")


def compute_energy(B_func, A_func):
    if periodic:
        harmonic = Function(Vd)
        harmonic.project(B_func - curl(A_func))
        return assemble(inner(B_func - harmonic, B_func - harmonic) * dx)
    else:
        return assemble(inner(B_func, B_func) * dx)


def compute_free_energy(B_func):
    return assemble(inner(B_func, B_func) * dx)


def compute_lamb(j, B):
    eps = 1e-10
    lamb = Function(Vg_).interpolate(dot(j, B)/(dot(B, B) + eps))
    with lamb.dat.vec_ro as v:
        _, max_val = v.max()
        _, min_val = v.min()
    if abs(min_val) < eps:
        return eps
    else:
        return max_val/min_val


def compute_xi_max(j, B):
    eps = 1e-10
    xi = Function(Vg).interpolate(cross(j, B)/(dot(B, B) + eps))
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
fieldnames = ["t", "helicity", "relative_helicity", "energy", "free_energy", "divB", "lamb", "xi"]
if mesh.comm.rank == 0:
    with open(data_filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

helicity    = compute_helicity(z.sub(2), z.sub(0))
relative_helicity = compute_relative_helicity(z.sub(2), z.sub(0))
divB        = compute_divB(z.sub(0))
energy      = compute_energy(z.sub(0), z.sub(2))
free_energy = compute_free_energy(z.sub(0))
lamb        = compute_lamb(z.sub(4), z.sub(0))
xi          = compute_xi_max(z.sub(4), z.sub(0))
helicity_initial = helicity

if mesh.comm.rank == 0:
    label = "GeneralizedHelicity" if bc == "periodic" else "Helicity"
    print(f"Initial {label} = {helicity_initial:.10e}", flush=True)
    row = {
        "t": float(t),
        "helicity":    float(helicity),
        "relative_helicity": float(relative_helicity),
        "energy":      float(energy),
        "free_energy": float(free_energy),
        "divB":        float(divB),
        "lamb":        float(lamb),
        "xi":          float(xi),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
        print(f"{row}")

delta_energy = 1.0
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
    two_lm = delta_energy > two_to_single_LM_de

    # One-time handoff: seed the single-LM state from the two-LM state.
    # λ_e (z.sub(5)) is dropped; λ_m moves from z.sub(6) into z_s.sub(5).
    if (not two_lm) and (not z_s_init):
        z_s_prev.sub(0).assign(z.sub(0))
        z_s_prev.sub(1).assign(z.sub(1))
        z_s_prev.sub(2).assign(z.sub(2))
        z_s_prev.sub(3).assign(z.sub(3))
        z_s_prev.sub(4).assign(z.sub(4))
        z_s_prev.sub(5).assign(z.sub(6))
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
        helicity     = compute_helicity(z.sub(2), z.sub(0))
        relative_helicity = compute_relative_helicity(z.sub(2), z.sub(0))
        divB         = compute_divB(z.sub(0))
        energy       = compute_energy(z.sub(0), z.sub(2))
        free_energy  = compute_free_energy(z.sub(0))
        delta_energy = 1/dt_used * (compute_energy(z_prev.sub(0), z_prev.sub(2)) - energy)
        lamb         = compute_lamb(z.sub(4), z.sub(0))
        xi           = compute_xi_max(z.sub(4), z.sub(0))
        if mesh.comm.rank == 0:
            print(GREEN % f"{delta_energy}")
            print(BLUE  % f"lmbda_e = {norm(z.sub(5))}, lmbda_m = {norm(z.sub(6))}")
    else:
        helicity    = compute_helicity(z_s.sub(2), z_s.sub(0))
        relative_helicity = compute_relative_helicity(z_s.sub(2), z_s.sub(0))
        divB        = compute_divB(z_s.sub(0))
        energy      = compute_energy(z_s.sub(0), z_s.sub(2))
        free_energy = compute_free_energy(z_s.sub(0))
        lamb        = compute_lamb(z_s.sub(4), z_s.sub(0))
        xi          = compute_xi_max(z_s.sub(4), z_s.sub(0))
        if mesh.comm.rank == 0:
            print(BLUE % f"lmbda_m = {norm(z_s.sub(5))}")
        # Sync z_s state into z so the output PVD keeps the same set of
        # functions across the two-LM → single-LM transition. λ_e (z.sub(5))
        # is no longer evolved, so leave it as-is; λ_m moves from z_s.sub(5)
        # into z.sub(6).
        for i in range(5):
            z.sub(i).assign(z_s.sub(i))
        z.sub(6).assign(z_s.sub(5))

    H_err = abs(helicity - helicity_initial)
    H_err_label = "|H - H_0|"
    if mesh.comm.rank == 0:
        print(f"Solved t = {float(t):.4f}, dt = {dt_used:g}, Newton iters = {iters}", flush=True)
        row = {
            "t": float(t),
            "helicity":    float(helicity),
            "relative_helicity": float(relative_helicity),
            "energy":      float(energy),
            "free_energy": float(free_energy),
            "divB":        float(divB),
            "lamb":        float(lamb),
            "xi":          float(xi),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            print(f"{row}, {H_err_label} = {H_err:.4e}")

    if output:
        pvd.write(*z.subfunctions, time=float(t))
        if is_e3:
            B_recover.project(z.sub(0) + B_b)
            pvd1.write(B_recover, time=float(t))
    z_prev.assign(z)
    z_s_prev.assign(z_s)
    timestep += 1

    # ---- choose dt for the NEXT step: fixed dt, then jump to a larger dt + drop tau ----
    if timestep > jump_after_steps:
        dt.assign(jump_dt)
        tau.assign(jump_tau)
