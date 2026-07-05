# non-conservative scheme 
# also used in M.He, P.E.Farrell, K.Hu, B.D.Andrews 2025 SISC.
from firedrake import *
import csv
import os
import sys
from config import CONFIG
from common import apply_jump_schedule, build_initial_condition, build_mesh_and_spaces

# parameters 
output = CONFIG.output
ic = CONFIG.ic
is_e3 = CONFIG.is_e3
bc = CONFIG.bc
scheme_name = "hdiv"
output_dir = f"{scheme_name}-{ic.lower()}"

if bc == "closed":
    periodic = False

elif bc == "periodic":
    periodic = True # no top and bottom label
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
jump_after_steps = CONFIG.jump_after_steps
jump_dt = CONFIG.jump_dt
jump_tau = CONFIG.jump_tau

mesh, spaces = build_mesh_and_spaces(CONFIG)
Vg, Vg_, Vc, Vd, Vn = (spaces[name] for name in ("Vg", "Vg_", "Vc", "Vd", "Vn"))
B_init, guide_field, _B_b, k_sign = build_initial_condition(mesh, CONFIG)

# E3 is evolved as the complete magnetic field on the z-periodic domain.
# Keep the harmonic field separately for the generalized-helicity diagnostic.
harmonic_field = guide_field
if is_e3 and periodic:
    B_init = B_init + harmonic_field
    guide_field = as_vector([0.0, 0.0, 0.0])

# Mixed unknowns: [B, j, u, E]
Z = MixedFunctionSpace([Vd, Vc, Vc])
z = Function(Z)
(B , j,  E) = split(z)
(Bt, jt, Et) = split(TestFunction(Z))

z_prev = Function(Z)
(Bp, jp, Ep) = split(z_prev)
B_avg = (B + Bp)/2
B_total_avg = B_avg + guide_field
E_avg = E
j_avg = j
eps = 1e-5
F = (
      inner((B-Bp)/dt, Bt) * dx
    + inner(curl(E_avg), Bt) * dx
    # j 
    + inner(j_avg, jt) * dx
    - inner(B_avg, curl(jt)) * dx
    # E
    + inner(E_avg, Et) * dx
    #+ tau * inner(cross(cross(j_avg, B_avg), B_avg)/(dot(Bp, Bp) + eps), Et) * dx
    + tau * inner(cross(cross(j_avg, B_total_avg), B_total_avg), Et) * dx
)

bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)) for subdomain in dirichlet_ids]

lu = {
	"mat_type": "aij",
	"snes_type": "newtonls",
    "snes_monitor": None,
    "ksp_monitor": None,
    "ksp_type":"preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type":"mumps"
}
sp = lu
       

(B_, j_, E_) = z.subfunctions
B_.rename("MagneticField")
E_.rename("ElectricField")
j_.rename("Current")

def project_initial_conditions(B_init):
    # Need to project the initial conditions
    # such that div(B) = 0 and B·n = 0
    Zp = MixedFunctionSpace([Vd, Vn])
    zp = Function(Zp)
    (B, p) = split(zp)
    bcp = [DirichletBC(Zp.sub(0), 0, subdomain) for subdomain in dirichlet_ids]
    # Write Lagrangian
    L = (
          0.5*inner(B, B)*dx
        - inner(B_init, B)*dx
        - inner(p, div(B))*dx
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
    Up = 0.5*(inner(B, B) + inner(div(B) * gamma, div(B)) + inner(p * (1/gamma), p))*dx
    Jp = derivative(derivative(Up, zp), zp)
    solve(Fp == 0, zp, bcp, Jp=Jp, solver_parameters=spp,
            options_prefix="B_init_div_free_projection")
    return zp.subfunctions[0]

B_.assign(project_initial_conditions(B_init))
z_prev.assign(z)

if output:
    pvd = VTKFile(f"{output_dir}/parker.pvd")
    pvd.write(*z.subfunctions, time=float(t))

def build_linear_solver(a, L, u_sol, bcs, aP=None, solver_parameters = None, options_prefix=None):
    problem = LinearVariationalProblem(a, L, u_sol, bcs=bcs, aP=aP)
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=solver_parameters,
                                     options_prefix=options_prefix)
    return solver

def build_nonlinear_solver(F, z, bcs, Jp=None, solver_parameters = None, options_prefix=None):
    problem = NonlinearVariationalProblem(F, z, bcs, Jp=Jp)
    solver = NonlinearVariationalSolver(problem,
                solver_parameters=solver_parameters,
                options_prefix=options_prefix)
    return solver


def helicity_solver():
    # Spaces for magnetic potential computation
    # If using periodic boundary conditions, we need to modify
    # this to account for the harmonic form [0, 0, 1]^T
    # using Yang's solver

    u = TrialFunction(Vc)
    v = TestFunction(Vc)
    u_sol = Function(Vc)

    # weak form of curl-curl problem 
    a = inner(curl(u), curl(v)) * dx
    L = inner(B_, curl(v)) * dx
    beta = Constant(0.1)
    Jp_curl = a + inner(beta * u, v) * dx
    bcs_curl = [DirichletBC(Vc, 0, subdomain) for subdomain in dirichlet_ids]
    rtol = 1E-8
    preconditioner = True
    if preconditioner:
        pc_type = "lu"
    else:
        pc_type = "none"
    sparams = {
        "snes_type": "ksponly",
        # "ksp_type": "lsqr",
        "ksp_type": "minres",
        "ksp_max_it": 1000,
        "ksp_convergence_test": "skip",
        #"ksp_monitor": None,
        "pc_type": pc_type,
        "ksp_norm_type": "preconditioned",
        "ksp_minres_nutol": 1E-8,
        }

    solver = build_linear_solver(a, L, u_sol, bcs_curl, Jp_curl, sparams, options_prefix="helicity")
    return solver


helicity_solver = helicity_solver()

def riesz_map(functional):
    function = Function(functional.function_space().dual())
    with functional.dat.vec as x, function.dat.vec as y:
        helicity_solver.snes.ksp.pc.apply(x, y)
    return function


def compute_helicity_energy(B):
    helicity_solver.solve()
    problem = helicity_solver._problem
    if helicity_solver.snes.ksp.getResidualNorm() > 0.01:
        # lifting strategy
        r = assemble(problem.F, bcs=problem.bcs)
        rstar = r.riesz_representation(riesz_map=riesz_map, bcs=problem.bcs)
        rstar.rename("RHS")
        # lft = uh - inner(r, uh)/inner(r, rstar) * rstar
        c = assemble(action(r, problem.u)) / assemble(action(r, rstar))
        ulft = Function(Vc, name="u_lifted")
        ulft.assign(problem.u - c * rstar)
        A = ulft
    else:
        A = problem.u
    potential_field = B - harmonic_field if is_e3 and periodic else B
    diff = norm(curl(A) - potential_field, "L2")
    if mesh.comm.rank == 0:
        print(f"magnetic potential residual = {diff:.8e}", flush=True)
    A_ = Function(Vc, name="MagneticPotential")
    A_.project(A)
    curlA = Function(Vd, name="CurlA")
    curlA.project(curl(A))
    diff_ = Function(Vd, name="CurlAMinusB")
    diff_.project(potential_field-curlA)
    # VTKFile(f"{output_dir}/magnetic_potential.pvd").write(curlA, diff_, A_)
    if is_e3 and periodic:
        # B is the complete field and curl(A) is its zero-harmonic part.
        return (assemble(inner(A, B + harmonic_field)*dx), diff, diff_,
                assemble(inner(B, B) * dx))
    elif is_e3:
        return (assemble(inner(A, B + 2 * guide_field)*dx), diff, diff_,
                assemble(inner(B + guide_field, B + guide_field) * dx))
    elif bc=="closed":
        return assemble(inner(A, B)*dx), diff, diff_, assemble(inner(B, B) * dx)
    else: 
        return assemble(inner(A, B + diff_)*dx), diff, diff_, assemble(inner(B - diff_, B-diff_) * dx)
       
def compute_Bn(B):
    n = FacetNormal(mesh)
    return assemble(inner(dot(B, n), dot(B, n))*ds_v)

def compute_divB(B):
    return norm(div(B), "L2")

def compute_free_energy(B):
    """Energy of the part above the fixed E3 harmonic field."""
    perturbation = B - harmonic_field if is_e3 and periodic else B
    return assemble(inner(perturbation, perturbation) * dx)

def compute_background_energy():
    """Energy of the E3 harmonic field; zero when no E3 field is present."""
    if not is_e3:
        return 0.0
    return assemble(inner(harmonic_field, harmonic_field) * dx(domain=mesh))

# monitor of (non)linear force-free field
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

# monitor of force-free
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

time_stepper = build_nonlinear_solver(F, z, bcs, solver_parameters=sp, options_prefix="time_stepper")

# Reproducibility: print + persist all run parameters
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
    "script":          "hdiv.py",
    "method":          "non-conservative HDiv",
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
}, header="hdiv.py")

data_filename = f"{output_dir}/data.csv"
fieldnames = ["t", "helicity", "energy", "free_energy", "background_energy",
              "divB", "lamb", "xi"]
helicity_print_label = "helicity_g" if periodic else "helicity"
if mesh.comm.rank == 0:
    with open(data_filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

helicity, diff, diff_, energy = compute_helicity_energy(z.sub(0))
free_energy = compute_free_energy(z.sub(0))
background_energy = compute_background_energy()
divB = compute_divB(z.sub(0))
lamb = compute_lamb(z.sub(1), z.sub(0))
xi = compute_xi_max(z.sub(1), z.sub(0))

if mesh.comm.rank == 0:
    row = {
        "t": float(t),
        "helicity": float(helicity),
        "energy": float(energy),
        "free_energy": float(free_energy),
        "background_energy": float(background_energy),
        "divB": float(divB),
        "lamb": float(lamb),
        "xi": float(xi),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
        printed_row = row.copy()
        if periodic:
            printed_row[helicity_print_label] = printed_row.pop("helicity")
        print(f"{printed_row}")


timestep = 0
z_prev.assign(z)
while (float(t) < float(T) - 1.0e-9):
    if float(t) + float(dt) > float(T):
        dt.assign(float(T) - float(t))
    if float(dt) <= 1e-14:
        break

    if mesh.comm.rank == 0:
        print(f"Solving for t = {float(t):.4f} (+dt={float(dt):g}), "
              f"dofs = {Z.dim()}, initial condition = {ic}, T={T}, bc={bc}", flush=True)

    z.assign(z_prev)
    time_stepper.solve()

    dt_used = float(dt)
    t.assign(float(t) + dt_used)

    # monitor
    helicity, diff, diff_, energy= compute_helicity_energy(z.sub(0))
    free_energy = compute_free_energy(z.sub(0))
    divB = compute_divB(z.sub(0))
    lamb = compute_lamb(z.sub(1), z.sub(0))
    xi = compute_xi_max(z.sub(1), z.sub(0))
    
    if mesh.comm.rank == 0:
        print(f"Solved t = {float(t):.4f}, dt = {dt_used:g}", flush=True)
        row = {
            "t": float(t),
            "helicity": float(helicity),
            "energy": float(energy),
            "free_energy": float(free_energy),
            "background_energy": float(background_energy),
            "divB": float(divB),
            "lamb": float(lamb),
            "xi": float(xi),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            printed_row = row.copy()
            if periodic:
                printed_row[helicity_print_label] = printed_row.pop("helicity")
            print(f"{printed_row}")

    if output:
        #if timestep % 10 == 0:
        pvd.write(*z.subfunctions,time=float(t))
    timestep += 1
    z_prev.assign(z)

    # ---- choose dt for the NEXT step: fixed dt, then jump to a larger dt + drop tau ----
    apply_jump_schedule(timestep, dt, tau, CONFIG)
