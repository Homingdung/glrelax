"""Shared Firedrake infrastructure and diagnostics."""

import datetime
import os
import subprocess

from firedrake import *


def build_mesh_and_spaces(config, include_real=False):
    (Lx, Ly, Lz), (Nx, Ny, Nz) = config.domain
    base = RectangleMesh(Nx, Ny, Lx, Ly, quadrilateral=True)
    mesh = ExtrudedMesh(base, Lz, 1, periodic=config.periodic)
    mesh.coordinates.dat.data[:, 0] -= Lx / 2
    mesh.coordinates.dat.data[:, 1] -= Ly / 2
    mesh.coordinates.dat.data[:, 2] -= Lz / 2
    spaces = {
        "Vg": VectorFunctionSpace(mesh, "Q", config.order),
        "Vg_": FunctionSpace(mesh, "Q", config.order),
        "Vc": FunctionSpace(mesh, "NCE", config.order),
        "Vd": FunctionSpace(mesh, "NCF", config.order),
        "Vn": FunctionSpace(mesh, "DQ", config.order - 1),
    }
    if include_real:
        spaces["VR"] = FunctionSpace(mesh, "R", 0)
    return mesh, spaces


def build_initial_condition(mesh, config):
    X, Y, Z = SpatialCoordinate(mesh)
    if config.ic == "hopf":
        w1, w2, s = 3, 2, 1
        deno = 1 + dot(as_vector([X, Y, Z]), as_vector([X, Y, Z]))
        coeff = 4 * sqrt(s) / ((pi * deno**3) * sqrt(w1**2 + w2**2))
        B_init = as_vector([
            coeff * 2 * (w2 * Y - w1 * X * Z),
            -coeff * 2 * (w2 * X + w1 * Y * Z),
            coeff * w1 * (-1 + X**2 + Y**2 - Z**2),
        ])
        zero = as_vector([0.0, 0.0, 0.0])
        return B_init, zero, zero, "n/a"

    x_c = [1, -1, 1, -1, 1, -1]
    z_c = [-20, -12, -4, 4, 12, 20]
    k_sign = [1, -1, 1, -1, 1, -1] if config.ic == "E3" else [1] * 6
    a, k, length, B0 = sqrt(2), 5.0, 2.0, 1.0
    Bx, By = 0.0, 0.0
    for i in range(6):
        coeff = exp(-((X - x_c[i])**2 / a**2) - (Y**2 / a**2)
                    - ((Z - z_c[i])**2 / length**2))
        Bx += coeff * ((2 * k * k_sign[i] * B0 / a) * (-Y))
        By += coeff * ((2 * k * k_sign[i] * B0 / a) * (X - x_c[i]))
    guide = as_vector([0.0, 0.0, B0])
    return as_vector([Bx, By, B0]) - guide, guide, guide, k_sign


def project_initial_conditions(B_init, Vd, Vn, dirichlet_ids):
    Zp = MixedFunctionSpace([Vd, Vn])
    zp = Function(Zp)
    B, p = split(zp)
    bcs = [DirichletBC(Zp.sub(0), 0, marker) for marker in dirichlet_ids]
    lagrangian = 0.5 * inner(B, B) * dx - inner(B_init, B) * dx - inner(p, div(B)) * dx
    residual = derivative(lagrangian, zp, TestFunction(Zp))
    parameters = {
        "mat_type": "nest", "snes_type": "ksponly", "snes_monitor": None,
        "ksp_monitor": None, "ksp_max_it": 1000,
        "ksp_norm_type": "preconditioned", "ksp_type": "minres",
        "pc_type": "fieldsplit", "pc_fieldsplit_type": "additive",
        "fieldsplit_pc_type": "cholesky",
        "fieldsplit_pc_factor_mat_solver_type": "mumps",
        "ksp_atol": 1e-5, "ksp_rtol": 1e-5,
        "ksp_minres_nutol": 1e-8, "ksp_convergence_test": "skip",
    }
    gamma = Constant(1e5)
    energy = 0.5 * (inner(B, B) + inner(gamma * div(B), div(B))
                    + inner(p / gamma, p)) * dx
    Jp = derivative(derivative(energy, zp), zp)
    solve(residual == 0, zp, bcs, Jp=Jp, solver_parameters=parameters,
          options_prefix="B_init_div_free_projection")
    return zp.subfunctions[0]


def build_linear_solver(a, L, solution, bcs, aP=None, solver_parameters=None,
                        options_prefix=None):
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs, aP=aP)
    return LinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                   options_prefix=options_prefix)


def build_nonlinear_solver(F, solution, bcs, Jp=None, solver_parameters=None,
                           options_prefix=None):
    problem = NonlinearVariationalProblem(F, solution, bcs, Jp=Jp)
    return NonlinearVariationalSolver(problem, solver_parameters=solver_parameters,
                                      options_prefix=options_prefix)


def compute_divB(B):
    return norm(div(B), "L2")


def compute_free_energy(B):
    return assemble(inner(B, B) * dx)


def compute_lamb(j, B, guide_field, scalar_space):
    eps = 1e-10
    total = B + guide_field
    value = Function(scalar_space).interpolate(dot(j, total) / (dot(total, total) + eps))
    with value.dat.vec_ro as vec:
        _, maximum = vec.max()
        _, minimum = vec.min()
    return eps if abs(minimum) < eps else maximum / minimum


def compute_xi_max(j, B, guide_field, vector_space):
    eps = 1e-10
    total = B + guide_field
    value = Function(vector_space).interpolate(cross(j, total) / (dot(total, total) + eps))
    with value.dat.vec_ro as vec:
        _, maximum = vec.max()
        _, minimum = vec.min()
    return eps if abs(minimum) < eps else maximum / minimum


def apply_jump_schedule(timestep, dt, tau, config):
    """Set values for the next solve, preserving the existing > semantics."""
    if timestep > config.jump_after_steps:
        dt.assign(config.jump_dt)
        tau.assign(config.jump_tau)


def write_params(path, params, mesh, header=""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      text=True).strip()
    except Exception:
        git = "unknown"
    lines = [f"# {header}", f"# time: {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"# git:  {git}", ""]
    width = max(len(key) for key in params) + 2
    lines.extend(f"{key:<{width}} {value}" for key, value in params.items())
    text = "\n".join(lines) + "\n"
    if mesh.comm.rank == 0:
        print(text, flush=True)
        with open(path, "w") as stream:
            stream.write(text)
