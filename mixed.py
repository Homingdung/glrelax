# projection-based method

from firedrake import *
import csv
import os
import sys
# ============================================================
# Parameters
# ============================================================
output = True
ic = os.environ.get("IC", "E3-positive")  # hopf, E3, or E3-positive
is_e3 = ic in ("E3", "E3-positive")
bc = "closed"  # "closed" or "periodic"
scheme_name = "mixed"
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
jump_after_steps = 100
jump_dt          = 100 if is_e3 else 100.0
jump_tau         = 1

# ============================================================
# Mesh
# ============================================================
base = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)
mesh = ExtrudedMesh(base, Lz, 1, periodic=periodic)
mesh.coordinates.dat.data[:, 0] -= Lx/2
mesh.coordinates.dat.data[:, 1] -= Ly/2
mesh.coordinates.dat.data[:, 2] -= Lz/2

Vg = VectorFunctionSpace(mesh, "Q", order)
Vg_ = FunctionSpace(mesh, "Q", order)
Vc = FunctionSpace(mesh, "NCE", order)
Vd = FunctionSpace(mesh, "NCF", order)
Vn = FunctionSpace(mesh, "DQ", order-1)

# ============================================================
# Initial condition (defined early so it can drive B')
# ============================================================
(X0, Y0, Z0) = x = SpatialCoordinate(mesh)

if ic == "hopf":
    w1 = 3
    w2 = 2
    s = 1
    deno = 1 + dot(x, x)
    coeff = 4*sqrt(s)/((pi * deno * deno * deno)*sqrt(w1**2+w2**2))
    B_init = as_vector([coeff*2*(w2*Y0-w1*X0*Z0),
                        -coeff*2*(w2*X0+w1*Y0*Z0),
                        coeff*w1*(-1+X0**2+Y0**2-Z0**2)])

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

    # closed/periodic: subtract harmonic background
    B_init = as_vector([B_x, B_y, B_z]) - B_b
# ============================================================
# Mixed unknowns: [B, j, H, u, E]
# ============================================================
Z = MixedFunctionSpace([Vd, Vc, Vc, Vd, Vc])
z = Function(Z)
(B, j, H, u, E) = split(z)
(Bt, jt, Ht, ut, Et) = split(TestFunction(Z))

z_prev = Function(Z)
(Bp, jp, Hp, up, Ep) = split(z_prev)
B_avg = (B + Bp)/2
E_avg = E
H_avg = H
j_avg = j
u_avg = u
eps = 1e-5
F = (
      inner((B-Bp)/dt, Bt) * dx
    + inner(curl(E_avg), Bt) * dx
    - inner(B_avg, curl(jt)) * dx
    + inner(j_avg, jt) * dx
    - inner(cross(Et, H_avg), u) * dx
    + inner(E_avg, Et) * dx
    + inner(u_avg, ut) * dx
    - tau * inner(cross(j_avg, H_avg), ut) * dx
    + inner(H_avg, Ht) * dx
    - inner(B_avg, Ht) * dx
    )

# ============================================================
# Boundary conditions: homogeneous Dirichlet on all components.
# ============================================================
bcs = [DirichletBC(Z.sub(index), 0, subdomain)
       for index in range(len(Z))
       for subdomain in dirichlet_ids]

lu = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "snes_monitor": None,
    "ksp_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}
sp = lu

# ============================================================
# Initial conditions
# ============================================================
# (B_init was defined above so it could drive B'.)

(B_, j_, H_, u_, E_) = z.subfunctions
B_.rename("MagneticField")
E_.rename("ElectricField")
H_.rename("HCurlMagneticField")
j_.rename("Current")
u_.rename("Velocity")

# ============================================================
# Project initial conditions: ensure div(B) = 0 with correct boundary
# ============================================================
def project_initial_conditions(B_init):
    Zp = MixedFunctionSpace([Vd, Vn])
    zp = Function(Zp)
    (Bvar, p) = split(zp)
    
    bcp = [DirichletBC(Zp.sub(0), 0, subdomain) for subdomain in dirichlet_ids]
    
    L_proj = (
          0.5*inner(Bvar, Bvar)*dx
        - inner(B_init, Bvar)*dx
        - inner(p, div(Bvar))*dx
        )
    Fp = derivative(L_proj, zp, TestFunction(Zp))
    spp = {
        "mat_type": "nest",
        "snes_type": "ksponly",
        "snes_monitor": None,
        "ksp_monitor": None,
        "ksp_max_it": 1000,
        "ksp_norm_type": "preconditioned",
        "ksp_type": "minres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_pc_type": "cholesky",
        "fieldsplit_pc_factor_mat_solver_type": "mumps",
        "ksp_atol": 1.0e-5,
        "ksp_rtol": 1.0e-5,
        "ksp_minres_nutol": 1E-8,
        "ksp_convergence_test": "skip",
    }
    gamma = Constant(1E5)
    Up = 0.5*(inner(Bvar, Bvar) + inner(div(Bvar) * gamma, div(Bvar)) + inner(p * (1/gamma), p))*dx
    Jp = derivative(derivative(Up, zp), zp)
    solve(Fp == 0, zp, bcp, Jp=Jp, solver_parameters=spp,
          options_prefix="B_init_div_free_projection")
    return zp.subfunctions[0]

B_.assign(project_initial_conditions(B_init))
z_prev.assign(z)

B_recover = Function(Vd, name="RecoveredMagneticField")

if output:
    pvd = VTKFile(f"{output_dir}/parker.pvd")
    pvd.write(*z.subfunctions, time=float(t))
    if is_e3 and bc == "closed":
        pvd1 = VTKFile(f"{output_dir}/recover.pvd")
        B_recover.project(z.sub(0) + B_b)
        pvd1.write(B_recover, time=float(t))

# ============================================================
# Solver builders
# ============================================================
def build_linear_solver(a, L, u_sol, bcs, aP=None, solver_parameters=None, options_prefix=None):
    problem = LinearVariationalProblem(a, L, u_sol, bcs=bcs, aP=aP)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=solver_parameters,
                                     options_prefix=options_prefix)
    return solver

def build_nonlinear_solver(F, z, bcs, Jp=None, solver_parameters=None, options_prefix=None):
    problem = NonlinearVariationalProblem(F, z, bcs, Jp=Jp)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=solver_parameters,
                                        options_prefix=options_prefix)
    return solver

# ============================================================
# Helicity / relative helicity solver
# ============================================================
def helicity_solver():
    """
    Solve for magnetic potential A satisfying ∇×A = B (with appropriate gauge),
    A ∈ H_0(curl).
    """
    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    u_sol = Function(Vc)

    a = inner(curl(u), curl(v)) * dx
    L = inner(B_, curl(v)) * dx
    beta = Constant(0.1)
    Jp_curl = a + inner(beta * u, v) * dx
    
    # Homogeneous tangential BC: A_f ∈ H_0(curl)
    bcs_curl = [DirichletBC(Vc, 0, subdomain) for subdomain in dirichlet_ids]
    
    sparams = {
        "snes_type": "ksponly",
        "ksp_type": "minres",
        "ksp_max_it": 1000,
        "ksp_convergence_test": "skip",
        "pc_type": "lu",
        "ksp_norm_type": "preconditioned",
        "ksp_minres_nutol": 1E-8,
    }
    solver = build_linear_solver(a, L, u_sol, bcs_curl, Jp_curl, sparams,
                                 options_prefix="helicity")
    return solver

helicity_solver = helicity_solver()

def riesz_map(functional):
    function = Function(functional.function_space().dual())
    with functional.dat.vec as x_, function.dat.vec as y_:
        helicity_solver.snes.ksp.pc.apply(x_, y_)
    return function

def compute_helicity_energy(B):
    """
    Compute helicity and energy depending on boundary condition.

    Returns: (helicity, ||curl(A) - B||, residual_field, energy)

    - closed:    H = (A, B),                E = (B, B)
    - periodic:  generalized helicity
    """
    helicity_solver.solve()
    problem = helicity_solver._problem

    if helicity_solver.snes.ksp.getResidualNorm() > 0.01:
        r = assemble(problem.F, bcs=problem.bcs)
        rstar = r.riesz_representation(riesz_map=riesz_map, bcs=problem.bcs)
        rstar.rename("RHS")
        c = assemble(action(r, problem.u)) / assemble(action(r, rstar))
        ulft = Function(Vc, name="u_lifted")
        ulft.assign(problem.u - c * rstar)
        A = ulft
    else:
        A = problem.u

    diff = norm(curl(A) - B, "L2")

    if mesh.comm.rank == 0:
        print(f"magnetic potential: ||curl(A) - B||_L2 = {diff:.8e}", flush=True)

    A_out = Function(Vc, name="MagneticPotential")
    A_out.project(A)

    curl_A = Function(Vd, name="CurlA")
    curl_A.project(curl(A))

    diff_field = Function(Vd, name="CurlAMinusB")
    diff_field.project(B - curl_A)

    if bc == "closed":
        # Standard helicity
        return (assemble(inner(A, B) * dx),
                diff,
                diff_field,
                assemble(inner(B, B) * dx))
    else:
        # periodic: generalized helicity
        return (assemble(inner(A, B + diff_field) * dx),
                diff,
                diff_field,
                assemble(inner(B - diff_field, B - diff_field) * dx))

def compute_divB(B):
    return norm(div(B), "L2")

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
# Time stepping
# ============================================================
time_stepper = build_nonlinear_solver(F, z, bcs, solver_parameters=sp,
                                      options_prefix="time_stepper")

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
    "script":          "mixed.py",
    "method":          "projection-based (mixed)",
    "ic":              ic,
    "k_sign":          "n/a" if not is_e3 else k_sign,
    "bc":              bc,
    "periodic":        periodic,
    "order":           order,
    "Lx,Ly,Lz":        f"{Lx},{Ly},{Lz}",
    "Nx,Ny,Nz":        f"{Nx},{Ny},{Nz}",
    "tau":             float(tau),
    "dt_init":         dt_init,
    "jump_after_steps":    jump_after_steps,
    "jump_dt":             jump_dt,
    "jump_tau":            jump_tau,
    "T":               T,
    "Vc.dim":          Vc.dim(),
    "Vd.dim":          Vd.dim(),
    "Vn.dim":          Vn.dim(),
    "Z.dim":           Z.dim(),
    "snes_type":       sp["snes_type"],
    "ksp_type":        sp["ksp_type"],
    "pc_type":         sp["pc_type"],
    "mat_solver":      sp["pc_factor_mat_solver_type"],
    "output_dir":      f"{output_dir}/",
}, header="mixed.py")

data_filename = f"{output_dir}/data.csv"
fieldnames = ["t", "helicity", "energy", "divB", "lamb", "xi"]
if mesh.comm.rank == 0:
    with open(data_filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Initial values
helicity, diff, diff_field, energy = compute_helicity_energy(z.sub(0))
divB = compute_divB(z.sub(0))
lamb = compute_lamb(z.sub(1), z.sub(0))
xi = compute_xi_max(z.sub(1), z.sub(0))

# Store initial helicity for monitoring conservation
helicity_initial = helicity

if mesh.comm.rank == 0:
    label = "GeneralizedHelicity" if bc == "periodic" else "Helicity"
    print(f"Initial {label} = {helicity_initial:.10e}", flush=True)
    
    row = {
        "t": float(t),
        "helicity": float(helicity),
        "energy": float(energy),
        "divB": float(divB),
        "lamb": float(lamb),
        "xi": float(xi),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
        print(f"{row}")

timestep = 0
z_prev.assign(z)
while (float(t) < float(T) - 1.0e-9):
    # don't overshoot the final time
    if float(t) + float(dt) > float(T):
        dt.assign(float(T) - float(t))
    if float(dt) <= 1e-14:
        break

    # ---- solve this step: single plain solve(), no retry ----
    z.assign(z_prev)
    time_stepper.solve()
    iters = time_stepper.snes.getIterationNumber()

    # ---- step accepted: advance time by the dt actually used in the solve ----
    dt_used = float(dt)
    t.assign(float(t) + dt_used)

    helicity, diff, diff_field, energy = compute_helicity_energy(z.sub(0))
    divB = compute_divB(z.sub(0))
    lamb = compute_lamb(z.sub(1), z.sub(0))
    xi = compute_xi_max(z.sub(1), z.sub(0))

    H_err = abs(helicity - helicity_initial)
    H_err_label = "|H - H_0|"

    if mesh.comm.rank == 0:
        print(f"Solved t = {float(t):.4f}, dt = {dt_used:g}, Newton iters = {iters}, "
              f"ic = {ic}, bc = {bc}, T = {T}", flush=True)
        row = {
            "t": float(t),
            "helicity": float(helicity),
            "energy": float(energy),
            "divB": float(divB),
            "lamb": float(lamb),
            "xi": float(xi),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            print(f"{row}, {H_err_label} = {H_err:.4e}")

    if output:
        pvd.write(*z.subfunctions, time=float(t))
        if is_e3 and bc == "closed":
            B_b = as_vector([0, 0, 1])
            B_recover.project(z.sub(0) + B_b)
            pvd1.write(B_recover, time=float(t))

    z_prev.assign(z)
    timestep += 1

    # ---- choose dt for the NEXT step: fixed dt, then jump to a larger dt + drop tau ----
    if timestep > jump_after_steps:
        dt.assign(jump_dt)
        tau.assign(jump_tau)
