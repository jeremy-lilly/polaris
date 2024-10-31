import cmocean  # noqa: F401
import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh
from scipy.integrate import quad

from polaris import Step
from polaris.mesh.planar import compute_planar_hex_nx_ny
from polaris.ocean.vertical import init_vertical_coord
# from polaris.ocean.viz import compute_transect, plot_transect
from polaris.viz import plot_horiz_field


def ujet(y, y0, y1, u_amp):
    if y <= y0 or y >= y1:
        return 0
    return u_amp * np.exp(1 / ((y - y0) * (y1 - y)))


def balance(y, a, f, y0, y1, u_amp):
    u = ujet(y, y0, y1, u_amp)
    return a * u * (f + (np.tan(y) / a) * u)


class Init(Step):
    """
    A step for creating a mesh and initial condition for planar
    barotropic jet tasks

    Attributes
    ----------
    resolution : float
        The resolution of the task in km
    """
    def __init__(self, component, resolution, indir):
        """
        Create the step

        Parameters
        ----------
        component : polaris.Component
            The component the step belongs to

        resolution : float
            The resolution of the task in km

        indir : str
            the directory the step is in, to which ``name`` will be appended
        """
        super().__init__(component=component, name='init', indir=indir)
        self.resolution = resolution

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info']:
            self.add_output_file(file)
        self.add_output_file('initial_state.nc',
                             validate_vars=['layerThickness'])

    def run(self):
        """
        Run this step of the task
        """
        config = self.config
        logger = self.logger

        section = config['planar_barotropic_jet']
        resolution = self.resolution

        # lx = section.getfloat('lx')
        # ly = section.getfloat('ly')
        earth_radius = 6.37122e6  # meters
        lx = 2 * np.pi * earth_radius / 1000  # km
        ly = lx / 2  # km

        g = 9.80616  # gravity

        # these could be hard-coded as functions of specific supported
        # resolutions but it is preferable to make them algorithmic like here
        # for greater flexibility
        nx, ny = compute_planar_hex_nx_ny(lx, ly, resolution)
        dc = 1e3 * resolution

        ds_mesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                       nonperiodic_x=False,
                                       nonperiodic_y=True)
        write_netcdf(ds_mesh, 'base_mesh.nc')

        ds_mesh = cull(ds_mesh, logger=logger)
        ds_mesh = convert(ds_mesh, graphInfoFileName='culled_graph.info',
                          logger=logger)
        write_netcdf(ds_mesh, 'culled_mesh.nc')

        section = config['planar_barotropic_jet']
        coriolis_parameter = section.getfloat('coriolis_parameter')

        ds = ds_mesh.copy()

        # nCells = ds.sizes['nCells']
        # x_cell = ds.xCell.values
        y_cell = ds.yCell.values

        nEdges = ds.sizes['nEdges']
        # x_edge = ds.xEdge.values
        # y_edge = ds.yEdge.values

        # nVertices = ds.sizes['nVertices']
        # x_vert = ds.xVertex.values
        y_vert = ds.yVertex.values

        verticesOnEdge = ds.verticesOnEdge.values - 1
        dvEdge = ds.dvEdge.values

        bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

        ds['bottomDepth'] = bottom_depth * xr.ones_like(ds.xCell)
        bottom_depth = ds['bottomDepth'].values

        ds['ssh'] = xr.zeros_like(ds.xCell)

        init_vertical_coord(config, ds)

        # x_min = x_vert.min()
        # x_max = x_vert.max()
        y_min = y_vert.min()
        y_max = y_vert.max()

        y0 = 0.25 * (y_min + y_max)  # bottom of jet
        y1 = 0.75 * (y_min + y_max)  # top of jet

        u_max = 80.0  # max jet velocity, m/s
        u_amp = u_max / np.exp(-4 / (y1 - y0)**2)

        # build a stream function at vertices
        # note that the flow only depends on y,
        # so the stream function only depends on y
        #     psi(y) = int_a^y u(y') dy'
        psi_vert = xr.zeros_like(ds.xVertex)
        for iVert, y in enumerate(y_vert):
            if y > y0 and y < y1:
                psi_vert[iVert], _ = quad(ujet, y_min, y,
                                          args=(y0, y1, u_amp))

        # it can be shown that
        #     \vec{u} = - \grad^{\perp} psi
        # so we get the normal velocity by taking
        # the discrete gradient perpendicular to
        # dual cell edges
        normal_velocity = xr.zeros_like(ds_mesh.xEdge)
        for iEdge in range(nEdges):
            vert1 = verticesOnEdge[iEdge, 0]
            vert2 = verticesOnEdge[iEdge, 1]
            if vert1 != -1 and vert2 != -1:
                normal_velocity[iEdge] = - ((psi_vert[vert2] -
                                             psi_vert[vert1]) /
                                            dvEdge[iEdge])

        normal_velocity, _ = xr.broadcast(normal_velocity, ds.refBottomDepth)
        normal_velocity = normal_velocity.transpose('nEdges', 'nVertLevels')
        normal_velocity = normal_velocity.expand_dims(dim='Time', axis=0)

        layer_thickness = xr.zeros_like(ds.xCell)
        for iCell, y in enumerate(y_cell):
            # this integral (3) from Galewsky doesn't make on the plane
            # we would need to treat the plane as an unrolled sphere
            # or figure out an appropriate change of variables?
            # but does this then imply geometry that Engin can't use?
            # this needs more thought
            integral, _ = quad(balance, y_min, y,
                               args=(earth_radius,
                                     coriolis_parameter,
                                     y0, y1, u_amp))
            layer_thickness[iCell] = bottom_depth[iCell] - integral / g

        # still need to add the thickness perturbation

        layer_thickness, _ = xr.broadcast(layer_thickness, ds.refBottomDepth)
        layer_thickness = layer_thickness.transpose('nCells', 'nVertLevels')
        layer_thickness = layer_thickness.expand_dims(dim='Time', axis=0)

        ds['normalVelocity'] = normal_velocity
        ds['layerThickness'] = layer_thickness
        ds['fCell'] = coriolis_parameter * xr.ones_like(ds.xCell)
        ds['fEdge'] = coriolis_parameter * xr.ones_like(ds_mesh.xEdge)
        ds['fVertex'] = coriolis_parameter * xr.ones_like(ds_mesh.xVertex)

        ds.attrs['nx'] = nx
        ds.attrs['ny'] = ny
        ds.attrs['dc'] = dc

        write_netcdf(ds, 'initial_state.nc')

        cell_mask = ds.maxLevelCell >= 1

        plot_horiz_field(ds, ds_mesh, 'normalVelocity',
                         'initial_normal_velocity.png', cmap='cmo.balance',
                         show_patch_edges=True, cell_mask=cell_mask)
