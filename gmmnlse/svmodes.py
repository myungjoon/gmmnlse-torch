import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs


def svmodes(lambda_, guess, nmodes, dx, dy, eps, boundary, field):
    """Calculate waveguide modes using semivectorial finite difference method."""

    boundary = boundary.upper()

    nx, ny = eps.shape

    # Now we pad eps on all sides by one grid point
    eps = np.column_stack([eps[:, 0], eps, eps[:, -1]])
    eps = np.vstack([eps[0, :], eps, eps[-1, :]])

    # Compute free-space wavevector
    k = 2 * np.pi / lambda_

    # Handle dx and dy (make them arrays if they're scalars)
    if np.isscalar(dx):
        dx = dx * np.ones(nx + 2)  # uniform grid
    else:
        dx = dx.flatten()  # convert to 1D array
        dx = np.concatenate([[dx[0]], dx, [dx[-1]]])  # pad dx on top and bottom

    if np.isscalar(dy):
        dy = dy * np.ones(ny + 2)  # uniform grid
    else:
        dy = dy.flatten()  # convert to 1D array
        dy = np.concatenate([[dy[0]], dy, [dy[-1]]])  # pad dy on top and bottom

    # Create grid metric arrays
    n = np.outer(np.ones(nx), (dy[2:ny+2] + dy[1:ny+1]) / 2).flatten()
    s = np.outer(np.ones(nx), (dy[0:ny] + dy[1:ny+1]) / 2).flatten()
    e = np.outer((dx[2:nx+2] + dx[1:nx+1]) / 2, np.ones(ny)).flatten()
    w = np.outer((dx[0:nx] + dx[1:nx+1]) / 2, np.ones(ny)).flatten()
    p = np.outer(dx[1:nx+1], np.ones(ny)).flatten()
    q = np.outer(np.ones(nx), dy[1:ny+1]).flatten()

    # Extract epsilon values at various grid points
    en = eps[1:nx+1, 2:ny+2].flatten()
    es = eps[1:nx+1, 0:ny].flatten()
    ee = eps[2:nx+2, 1:ny+1].flatten()
    ew = eps[0:nx, 1:ny+1].flatten()
    ep = eps[1:nx+1, 1:ny+1].flatten()

    # Calculate finite difference coefficients based on field type
    field_lower = field.lower()

    if field_lower == 'ex':
        an = 2.0 / n / (n + s)
        as_ = 2.0 / s / (n + s)
        ae = 8 * (p * (ep - ew) + 2.0 * w * ew) * ee / \
             ((p * (ep - ee) + 2.0 * e * ee) * (p**2 * (ep - ew) + 4.0 * w**2 * ew) +
              (p * (ep - ew) + 2.0 * w * ew) * (p**2 * (ep - ee) + 4.0 * e**2 * ee))
        aw = 8 * (p * (ep - ee) + 2.0 * e * ee) * ew / \
             ((p * (ep - ee) + 2.0 * e * ee) * (p**2 * (ep - ew) + 4.0 * w**2 * ew) +
              (p * (ep - ew) + 2.0 * w * ew) * (p**2 * (ep - ee) + 4.0 * e**2 * ee))
        ap = ep * k**2 - an - as_ - ae * ep / ee - aw * ep / ew

    elif field_lower == 'ey':
        an = 8 * (q * (ep - es) + 2.0 * s * es) * en / \
             ((q * (ep - en) + 2.0 * n * en) * (q**2 * (ep - es) + 4.0 * s**2 * es) +
              (q * (ep - es) + 2.0 * s * es) * (q**2 * (ep - en) + 4.0 * n**2 * en))
        as_ = 8 * (q * (ep - en) + 2.0 * n * en) * es / \
              ((q * (ep - en) + 2.0 * n * en) * (q**2 * (ep - es) + 4.0 * s**2 * es) +
               (q * (ep - es) + 2.0 * s * es) * (q**2 * (ep - en) + 4.0 * n**2 * en))
        ae = 2.0 / e / (e + w)
        aw = 2.0 / w / (e + w)
        ap = ep * k**2 - an * ep / en - as_ * ep / es - ae - aw

    elif field_lower == 'scalar':
        an = 2.0 / n / (n + s)
        as_ = 2.0 / s / (n + s)
        ae = 2.0 / e / (e + w)
        aw = 2.0 / w / (e + w)
        ap = ep * k**2 - an - as_ - ae - aw

    else:
        raise ValueError(f"field must be 'EX', 'EY', or 'scalar', got '{field}'")

    # Create index array for boundary conditions
    ii = np.arange(nx * ny).reshape(nx, ny)

    # Modify matrix elements to account for boundary conditions

    # North boundary
    ib = ii[:, -1]
    if boundary[0] == 'S':
        ap[ib] = ap[ib] + an[ib]
    elif boundary[0] == 'A':
        ap[ib] = ap[ib] - an[ib]

    # South boundary
    ib = ii[:, 0]
    if boundary[1] == 'S':
        ap[ib] = ap[ib] + as_[ib]
    elif boundary[1] == 'A':
        ap[ib] = ap[ib] - as_[ib]

    # East boundary
    ib = ii[-1, :]
    if boundary[2] == 'S':
        ap[ib] = ap[ib] + ae[ib]
    elif boundary[2] == 'A':
        ap[ib] = ap[ib] - ae[ib]

    # West boundary
    ib = ii[0, :]
    if boundary[3] == 'S':
        ap[ib] = ap[ib] + aw[ib]
    elif boundary[3] == 'A':
        ap[ib] = ap[ib] - aw[ib]

    # Build sparse matrix indices
    iall = ii.flatten()
    in_ = ii[:, 1:].flatten()
    is_ = ii[:, :-1].flatten()
    ie = ii[1:, :].flatten()
    iw = ii[:-1, :].flatten()

    # Build the sparse matrix A
    # MATLAB: A = sparse([iall,iw,ie,is,in], [iall,ie,iw,in,is], [ap(iall),ae(iw),aw(ie),an(is),as(in)])
    row_indices = np.concatenate([iall, iw, ie, is_, in_])
    col_indices = np.concatenate([iall, ie, iw, in_, is_])
    data = np.concatenate([ap[iall], ae[iw], aw[ie], an[is_], as_[in_]])

    A = csr_matrix((data, (row_indices, col_indices)), shape=(nx * ny, nx * ny))

    # Make it symmetric
    A = (A + A.T) / 2

    # Solve eigenvalue problem
    shift = (2 * np.pi * guess / lambda_)**2

    # Use scipy's sparse eigenvalue solver
    try:
        d, v = eigs(A, k=nmodes, sigma=shift, which='LM', tol=1e-20)
    except Exception as e:
        print(f"Warning: eigenvalue solver failed with error: {e}")
        print("Attempting with default tolerance...")
        d, v = eigs(A, k=nmodes, sigma=shift, which='LM')

    # Calculate effective indices
    neff = lambda_ * np.sqrt(d) / (2 * np.pi)

    # Reshape eigenvectors into mode shapes
    phi = np.zeros((nx, ny, nmodes), dtype=complex)

    for k in range(nmodes):
        temp = v[:, k].reshape(nx, ny)
        temp = temp / np.max(np.abs(temp))
        phi[:, :, k] = temp

    # Return real-valued results if input was real
    if np.all(np.isreal(neff)):
        neff = np.real(neff)
    if np.all(np.isreal(phi)):
        phi = np.real(phi)

    return phi, neff
