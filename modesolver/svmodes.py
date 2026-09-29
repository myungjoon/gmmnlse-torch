import numpy as np
from scipy.sparse import eye as speye, coo_array
from scipy.sparse.linalg import eigs, LinearOperator


def svmodes(wavelength, guess, nmodes, dx, dy, eps, boundary, field):
    """
    Calculate modes of a dielectric waveguide using the semivectorial
    finite difference method.

    Parameters
    ----------
    wavelength : float
        Optical wavelength
    guess : float
        Effective index guess (eigenvalue search center)
    nmodes : int
        Number of modes to calculate
    dx : float or array
        Horizontal grid spacing (scalar or 1D array)
    dy : float or array
        Vertical grid spacing (scalar or 1D array)
    eps : ndarray (nx, ny)
        Permittivity mesh (= n^2(x,y))
    boundary : str
        4-character boundary condition string (N/S/E/W).
        Each character: 'A' antisymmetric, 'S' symmetric, '0' zero.
    field : str
        'EX', 'EY', or 'scalar'

    Returns
    -------
    phi : ndarray (nx, ny, nmodes)
        Normalized mode fields
    neff : ndarray (nmodes,)
        Effective indices
    """

    boundary = boundary.upper()
    nx, ny = eps.shape

    # Pad eps on all sides by one grid point
    eps = np.column_stack([eps[:, 0], eps, eps[:, -1]])
    eps = np.vstack([eps[0, :], eps, eps[-1, :]])

    k = 2 * np.pi / wavelength

    # Build dx array
    if np.isscalar(dx):
        dx = dx * np.ones(nx + 2)
    else:
        dx = np.asarray(dx, dtype=float).ravel()
        dx = np.concatenate([[dx[0]], dx, [dx[-1]]])

    # Build dy array
    if np.isscalar(dy):
        dy = dy * np.ones(ny + 2)
    else:
        dy = np.asarray(dy, dtype=float).ravel()
        dy = np.concatenate([[dy[0]], dy, [dy[-1]]])

    # Grid metric arrays (shape: nx*ny)
    ones_x = np.ones(nx)
    ones_y = np.ones(ny)

    n  = np.outer(ones_x, (dy[2:ny+2] + dy[1:ny+1]) / 2).ravel()
    s  = np.outer(ones_x, (dy[0:ny]   + dy[1:ny+1]) / 2).ravel()
    e  = np.outer((dx[2:nx+2] + dx[1:nx+1]) / 2, ones_y).ravel()
    w  = np.outer((dx[0:nx]   + dx[1:nx+1]) / 2, ones_y).ravel()
    p  = np.outer(dx[1:nx+1], ones_y).ravel()
    q  = np.outer(ones_x, dy[1:ny+1]).ravel()

    # Permittivity at neighbouring nodes
    en = eps[1:nx+1, 2:ny+2].ravel()
    es = eps[1:nx+1, 0:ny  ].ravel()
    ee = eps[2:nx+2, 1:ny+1].ravel()
    ew = eps[0:nx,   1:ny+1].ravel()
    ep = eps[1:nx+1, 1:ny+1].ravel()

    # FD coefficients
    field_lower = field.lower()

    if field_lower == 'ex':
        an  = 2.0 / n / (n + s)
        as_ = 2.0 / s / (n + s)
        denom_e = ((p*(ep-ee) + 2*e*ee) * (p**2*(ep-ew) + 4*w**2*ew) +
                   (p*(ep-ew) + 2*w*ew) * (p**2*(ep-ee) + 4*e**2*ee))
        ae  = 8 * (p*(ep-ew) + 2*w*ew) * ee / denom_e
        aw  = 8 * (p*(ep-ee) + 2*e*ee) * ew / denom_e
        ap  = ep * k**2 - an - as_ - ae * ep/ee - aw * ep/ew

    elif field_lower == 'ey':
        denom_n = ((q*(ep-en) + 2*n*en) * (q**2*(ep-es) + 4*s**2*es) +
                   (q*(ep-es) + 2*s*es) * (q**2*(ep-en) + 4*n**2*en))
        an  = 8 * (q*(ep-es) + 2*s*es) * en / denom_n
        as_ = 8 * (q*(ep-en) + 2*n*en) * es / denom_n
        ae  = 2.0 / e / (e + w)
        aw  = 2.0 / w / (e + w)
        ap  = ep * k**2 - an * ep/en - as_ * ep/es - ae - aw

    else:  # scalar
        an  = 2.0 / n / (n + s)
        as_ = 2.0 / s / (n + s)
        ae  = 2.0 / e / (e + w)
        aw  = 2.0 / w / (e + w)
        ap  = ep * k**2 - an - as_ - ae - aw

    # Node index array
    ii = np.arange(nx * ny).reshape(nx, ny)

    # Apply boundary conditions (modify ap in-place)
    # North
    ib = ii[:, ny-1]
    if boundary[0] == 'S':
        ap[ib] += an[ib]
    elif boundary[0] == 'A':
        ap[ib] -= an[ib]

    # South
    ib = ii[:, 0]
    if boundary[1] == 'S':
        ap[ib] += as_[ib]
    elif boundary[1] == 'A':
        ap[ib] -= as_[ib]

    # East
    ib = ii[nx-1, :]
    if boundary[2] == 'S':
        ap[ib] += ae[ib]
    elif boundary[2] == 'A':
        ap[ib] -= ae[ib]

    # West
    ib = ii[0, :]
    if boundary[3] == 'S':
        ap[ib] += aw[ib]
    elif boundary[3] == 'A':
        ap[ib] -= aw[ib]

    # Index arrays for off-diagonal entries
    iall = ii.ravel()
    in_  = ii[:, 1:ny ].ravel()   # north neighbours (interior)
    is_  = ii[:, 0:ny-1].ravel()  # south neighbours (interior)
    ie_  = ii[1:nx, :  ].ravel()  # east  neighbours (interior)
    iw_  = ii[0:nx-1, :].ravel()  # west  neighbours (interior)

    # Build sparse matrix via COO (memory-efficient)
    row  = np.concatenate([iall, iw_,  ie_,  is_,  in_ ])
    col  = np.concatenate([iall, ie_,  iw_,  in_,  is_ ])
    data = np.concatenate([ap[iall], ae[iw_], aw[ie_], an[is_], as_[in_]])

    N = nx * ny
    A = coo_array((data, (row, col)), shape=(N, N)).tocsc()

    # Free large intermediate arrays
    del an, as_, ae, aw, ap, en, es, ee, ew, ep
    del n, s, e, w, p, q
    del iall, in_, is_, ie_, iw_, row, col, data

    # Shift for shift-invert mode
    shift = (2 * np.pi * guess / wavelength) ** 2

    # Build OPinv using PyAMG
    OPinv = _make_amg_opinv(A, shift)

    # Solve eigenvalue problem
    d, v = eigs(A, k=nmodes, sigma=shift, which='LM',
                OPinv=OPinv, tol=1e-10, ncv=max(2*nmodes+1, 20))

    neff = wavelength * np.sqrt(d) / (2 * np.pi)

    # Sort by descending real(neff)
    idx  = np.argsort(-neff.real)
    neff = neff[idx]
    v    = v[:, idx]

    # Reshape and normalise mode fields
    phi = np.zeros((nx, ny, nmodes), dtype=complex)
    for m in range(nmodes):
        mode = v[:, m].reshape(nx, ny)
        phi[:, :, m] = mode / np.max(np.abs(mode))

    # Return real arrays if imaginary parts are negligible
    if np.all(np.abs(neff.imag) < 1e-10 * np.abs(neff.real)):
        neff = neff.real
    if np.all(np.abs(phi.imag) < 1e-10 * np.abs(phi.real).max()):
        phi = phi.real

    return phi, neff


def _make_amg_opinv(A, shift):
    """
    Build a LinearOperator that approximates (A - shift*I)^{-1}
    using an Algebraic Multigrid (PyAMG) solver as preconditioner
    inside a GMRES iteration.

    Falls back to spilu -> gmres if PyAMG is not installed,
    and finally to a direct splu if the matrix is small enough.
    """
    from scipy.sparse import eye as speye

    M = (A - shift * speye(A.shape[0], format='csc')).tocsr()

    # ----------------------------------------------------------------
    # Option 1: PyAMG  (best for large grids)
    # ----------------------------------------------------------------
    try:
        import pyamg
        # Ensure int32 indices (required by pyamg's C kernels)
        M_real = M.real.tocsr()
        M_real.indptr  = M_real.indptr.astype(np.int32)
        M_real.indices = M_real.indices.astype(np.int32)
        ml = pyamg.smoothed_aggregation_solver(
            M_real,
            strength='symmetric',
            max_coarse=500,
        )
        print(f"[svmodes] PyAMG solver ready  "
              f"(levels={len(ml.levels)}, "
              f"operator complexity={ml.operator_complexity():.2f})")

        def matvec_amg(x):
            x_r = x.real
            x_i = x.imag
            sol_r = ml.solve(x_r, tol=1e-12, accel='gmres', maxiter=30)
            sol_i = ml.solve(x_i, tol=1e-12, accel='gmres', maxiter=30)
            return sol_r + 1j * sol_i

        return LinearOperator(A.shape, matvec=matvec_amg, dtype=complex)

    except ImportError:
        print("[svmodes] PyAMG not found — falling back to spilu+gmres")

    # ----------------------------------------------------------------
    # Option 2: spilu + gmres  (medium grids)
    # ----------------------------------------------------------------
    try:
        from scipy.sparse.linalg import spilu, gmres as _gmres

        M_csc = M.tocsc()
        ilu   = spilu(M_csc, fill_factor=20,
                      drop_tol=1e-6,
                      permc_spec='MMD_AT_PLUS_A')
        P     = LinearOperator(A.shape, ilu.solve)
        print("[svmodes] spilu preconditioner ready  (fill_factor=20)")

        def matvec_ilu(x):
            sol, info = _gmres(M_csc, x, M=P, atol=1e-12, maxiter=500)
            if info != 0:
                print(f"[svmodes] Warning: gmres did not converge (info={info})")
            return sol

        return LinearOperator(A.shape, matvec=matvec_ilu, dtype=complex)

    except MemoryError:
        print("[svmodes] spilu ran out of memory — falling back to direct splu")

    # ----------------------------------------------------------------
    # Option 3: splu  (small grids / last resort)
    # ----------------------------------------------------------------
    from scipy.sparse.linalg import splu
    print("[svmodes] Using direct splu solver")
    lu = splu(M.tocsc(), permc_spec='MMD_AT_PLUS_A')
    return LinearOperator(A.shape, lu.solve, dtype=complex)