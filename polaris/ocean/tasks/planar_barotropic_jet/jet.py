import argparse

import numpy as np
import xarray
from scipy.integrate import quadrature

from polaris.ocean.tasks.planar_barotropic_jet.msh import load_mesh
from polaris.ocean.tasks.planar_barotropic_jet.operators import trsk_mats
from polaris.ocean.tasks.planar_barotropic_jet.stb import strtobool

# from scipy.sparse.linalg import gcrotmk


def ypos_to_ylat(y, ymin, ymax):
    return (np.pi / 2) * (1 / (ymax - ymin)) * (2 * y - ymax - ymin)


def xpos_to_xlon(x, xmin, xmax):
    return (2 * np.pi / (xmax - xmin)) * (x - xmin)


def ujet(lat, lat0, lat1, uamp):
    """
    Helper to integrate PSI from U = -1/R * d/d.lat(PSI) via
    quadrature...

    U(lat) = A * exp(1/((lat-lat0) * (lat-lat1)))

    """

    vals = -1 * uamp * np.exp(1.0E+0 / ((lat - lat0) *
                                        (lat - lat1)))

    vals[lat < lat0] = 0.0
    vals[lat > lat1] = 0.0

    return vals


def h_balance(lat, lat0, lat1, uamp, f, g):
    return -1 * (-1 / g) * f * ujet(lat, lat0, lat1, uamp)


def init(name, save, rsph=6371220.0, pert=False):
    """
    INIT: Form SWE initial conditions for the barotropic jet
    case.

    Adds initial conditions to the MPAS mesh file NAME.nc,
    with the output IC's written to SAVE.nc.

    If PERT=TRUE, adds a perturbation to the layer thickness

    """
    # Authors: Darren Engwirda, Sara Calandrini
    # Edits for plane: Jeremy Lilly

# ------------------------------------ load an MPAS mesh file

    print("Loading the mesh file...")

    mesh = load_mesh(name)

# ------------------------------------ build TRSK matrix op's

    print("Forming coefficients...")

    trsk = trsk_mats(mesh)

# ------------------------------------ build a streamfunction

    print("Computing streamfunction...")

# -- J. Galewsky, R.K. Scott & L.M. Polvani (2004) An initial
# -- value problem for testing numerical models of the global
# -- shallow-water equations, Tellus A: Dynamic Meteorology &
# -- Oceanography, 56:5, 429-440

# -- this routine returns a scaled version of the Galewsky et
# -- al flow in cases where RSPH != 6371220m, with {u, h, g}
# -- multiplied by RSPH / 6371220m. This preserves (vortical)
# -- dynamics across arbitrarily sized spheres.

    erot = 7.292E-05  # Earth's omega
    grav = 9.80616  # gravity

    xmin = np.min(mesh.vert.xpos)
    xmax = np.max(mesh.vert.xpos)
    ymin = np.min(mesh.vert.ypos)
    ymax = np.max(mesh.vert.ypos)

    lat0 = np.pi / 7.0  # jet lat width
    lat1 = np.pi / 2.0 - lat0
    # jet_center = 0.7  # relative center of jet
    # jet_width = 0.3  # relative width of jet
    # shift_up = jet_center + jet_width / 2
    # shift_down = jet_center - jet_width / 2
    # y0 = (1 - shift_down) * ymin + shift_down * ymax  # bottom of jet
    # y1 = (1 - shift_up) * ymin + shift_up * ymax  # top of jet

    umax = 80.0  # jet max speed m/s
    umid = umax * (ymax - ymin) / np.pi  # scale(ish) to mesh
    hbar = 10000.0  # mean layer thickness

    uamp = umid / np.exp(-4. / (lat1 - lat0) ** 2)

# -- build a streamfunction at mesh vertices using quadrature

    vpsi = np.zeros(
        mesh.vert.size, dtype=np.float64)
    cpsi = np.zeros(
        mesh.cell.size, dtype=np.float64)

    ylat_vert = ypos_to_ylat(mesh.vert.ypos, ymin, ymax)
    for vert in range(mesh.vert.size):
        lat = ylat_vert[vert]
        if (lat >= lat0 and lat < lat1):
            vpsi[vert], _ = quadrature(
                ujet, lat0, lat, miniter=8,
                args=(lat0, lat1, uamp))

    vpsi[ylat_vert >= lat1] = np.min(vpsi)
    # vpsi[ylat_vert[:] >= lat1] = np.max(vpsi)

    print("--> done: vert!")

    ylat_cell = ypos_to_ylat(mesh.cell.ypos, ymin, ymax)
    for cell in range(mesh.cell.size):
        lat = ylat_cell[cell]
        if (lat >= lat0 and lat < lat1):
            cpsi[cell], _ = quadrature(
                ujet, lat0, lat, miniter=8,
                args=(lat0, lat1, uamp))

    cpsi[ylat_cell >= lat1] = np.min(cpsi)
    # cpsi[ylat_cell[:] >= lat1] = np.max(cpsi)

    print("--> done: cell!")

# -- form velocity on edges from streamfunction: ensures u is
# -- div-free in a discrete sense.

# -- this comes from taking div(*) of the momentum equations,
# -- see: H. Weller, J. Thuburn, C.J. Cotter (2012):
# -- Computational Modes and Grid Imprinting on Five Quasi-
# -- Uniform Spherical C-Grids, M.Wea.Rev. 140(8): 2734-2755.

    print("Computing velocity field...")

    ylat_edge = ypos_to_ylat(mesh.edge.ypos, ymin, ymax)

    unrm = trsk.edge_grad_perp * vpsi * -1.00
    # repair at boundary
    unrm[ylat_edge > lat1] = 0
    unrm[ylat_edge < lat0] = 0

    uprp = trsk.edge_grad_norm * cpsi * -1.00
    # repair at boundary
    uprp[ylat_edge > lat1] = 0
    uprp[ylat_edge < lat0] = 0

    udiv = trsk.cell_flux_sums * unrm

    print("--> max(abs(unrm)):", np.max(unrm))
    print("--> sum(div(unrm)):", np.sum(udiv))

# -- solve -g * del^2 h = div f * u_perp for layer thickness,
# -- leads to a h which is in discrete balance

    print("Computing flow thickness...")

    frot = 2.0 * erot * np.sin(np.pi / 4)
    # frot = 2.0 * erot * np.sin(ylat_edge)

    # vrhs = trsk.cell_flux_sums * (frot * uprp)
    # vrhs = vrhs * -1.00 / grav

    # vrhs = vrhs - np.mean(vrhs)  # INT rhs dA must be 0.0
    # vrhs = vrhs - np.mean(vrhs)

    # ttic = time.time()
    # hdel, info = gcrotmk(trsk.cell_del2_sums, vrhs,
    #                      rtol=1.E-8, atol=1.E-8, m=50, k=25)
    # ttoc = time.time()

    # if (info != +0):
    #     raise Exception("Did not converge!")

    hdel = np.zeros(mesh.cell.size)
    for cell in range(mesh.cell.size):
        lat = ylat_cell[cell]
        if (lat >= lat0 and lat < lat1):
            hdel[cell], _ = quadrature(
                h_balance, lat0, lat, miniter=8,
                args=(lat0, lat1, uamp, frot, grav))

    hdel[ylat_cell >= lat1] = np.min(hdel)

    herr = hbar - hdel
    # add offset for mean hh
    hdel = hdel + (np.sum(mesh.cell.area * herr) /
                   np.sum(mesh.cell.area * 1.00))

    # test repairs for planar balance
    # h_min_lat = ylat_cell[np.argmin(hdel)]
    # h_max_lat = ylat_cell[np.argmax(hdel)]
    # hdel[ylat_cell < h_max_lat] = np.max(hdel)
    # hdel[ylat_cell > h_min_lat] = np.min(hdel)

# -- optional: add perturbation to the thickness distribution

    xlon_cell = xpos_to_xlon(mesh.cell.xpos, xmin, xmax)
    lat2 = np.pi / 4.  # perturbation constants
    lon2 = np.pi / 1.

    hmul = 120.0
    eta1 = 1. / 3.
    eta2 = 1. / 15.

    hadd = (hmul * np.cos(ylat_cell) *
            np.exp(-((xlon_cell - lon2) / eta1) ** 2) *
            np.exp(-((lat2 - ylat_cell) / eta2) ** 2))

    hdel = hdel + float(pert) * hadd

# -- inject mesh with IC.'s and write output MPAS netCDF file

    print("Output written to:", save)

    vmag = np.sqrt(unrm ** 2 + uprp ** 2)
    vvel = (
        vmag[mesh.vert.edge[:, 0] - 1] +
        vmag[mesh.vert.edge[:, 1] - 1] +
        vmag[mesh.vert.edge[:, 2] - 1]
    ) / 3.00

    init = xarray.open_dataset(name)
    # init.attrs.update({"sphere_radius": mesh.rsph})
    # init.attrs.update({"config_gravity": grav})
    # init["xCell"] = (("nCells"), mesh.cell.xpos)
    # init["yCell"] = (("nCells"), mesh.cell.ypos)
    # init["zCell"] = (("nCells"), mesh.cell.zpos)
    # init["areaCell"] = (("nCells"), mesh.cell.area)

    # init["xEdge"] = (("nEdges"), mesh.edge.xpos)
    # init["yEdge"] = (("nEdges"), mesh.edge.ypos)
    # init["zEdge"] = (("nEdges"), mesh.edge.zpos)
    # init["dvEdge"] = (("nEdges"), mesh.edge.vlen)
    # init["dcEdge"] = (("nEdges"), mesh.edge.clen)

    # init["xVertex"] = (("nVertices"), mesh.vert.xpos)
    # init["yVertex"] = (("nVertices"), mesh.vert.ypos)
    # init["zVertex"] = (("nVertices"), mesh.vert.zpos)
    # init["areaTriangle"] = (("nVertices"), mesh.vert.area)
    # init["kiteAreasOnVertex"] = (
    #     ("nVertices", "vertexDegree"), mesh.vert.kite)

    init["h"] = (
        ("Time", "nCells"),
        np.reshape(hdel, (1, mesh.cell.size)))
    init["u"] = (
        ("Time", "nEdges"),
        np.reshape(unrm, (1, mesh.edge.size)))

    init["streamfunction"] = (("nVertices"), vpsi)
    init["velocityTotals"] = (("nVertices"), vvel)
    # init["vorticity"] = (
    #     ("nVertices"),
    #     (trsk.dual_curl_sums * unrm) / mesh.vert.area)

    # using cos for agreement at periodic boundary in y
    # init["fCell"] = (("nCells"),
    #                  2.00E+00 * erot * np.sin(ylat_cell))
    # init["fEdge"] = (("nEdges"),
    #                  2.00E+00 * erot * np.sin(ylat_edge))
    # init["fVertex"] = (("nVertices"),
    #                    2.00E+00 * erot * np.sin(ylat_vert))
    init["fCell"] = (("nCells"),
                     2.00E+00 * erot *
                     np.sin(np.pi / 4) * np.ones(mesh.cell.size))
    init["fEdge"] = (("nEdges"),
                     2.00E+00 * erot *
                     np.sin(np.pi / 4) * np.ones(mesh.edge.size))
    init["fVertex"] = (("nVertices"),
                       2.00E+00 * erot *
                       np.sin(np.pi / 4) * np.ones(mesh.vert.size))

    init.to_netcdf(save, format="NETCDF4")

    return


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--mesh-file", dest="mesh_file", type=str,
        required=True, help="Path to user mesh file.")

    parser.add_argument(
        "--init-file", dest="init_file", type=str,
        required=True, help="IC's filename to write.")

    parser.add_argument(
        "--with-pert", dest="with_pert",
        type=lambda x: bool(strtobool(str(x.strip()))),
        required=True, help="True to add h perturbation.")

    parser.add_argument(
        "--radius", dest="radius", type=float,
        required=True, help="Value of sphere_radius [m].")

    args = parser.parse_args()

    init(name=args.mesh_file,
         save=args.init_file,
         rsph=args.radius,
         pert=args.with_pert)