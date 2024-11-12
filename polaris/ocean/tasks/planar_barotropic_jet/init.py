import cmocean  # noqa: F401
import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from polaris import Step
from polaris.mesh.planar import compute_planar_hex_nx_ny
from polaris.ocean.tasks.planar_barotropic_jet.jet import init as jet_init
from polaris.ocean.vertical import init_vertical_coord
from polaris.viz import plot_horiz_field


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

        # domain is roughly the size of an 'unrolled' earth
        rsph = 6.37122e6  # meters
        lx = 2 * np.pi * rsph / 1000  # km
        ly = lx / 2  # km

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

        ds = ds_mesh.copy()

        bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')
        ds['bottomDepth'] = bottom_depth * xr.ones_like(ds.xCell)
        ds['ssh'] = xr.zeros_like(ds.xCell)

        init_vertical_coord(config, ds)

        temperature = section.getfloat('temperature')
        salinity = section.getfloat('salinity')

        temperature_array = temperature * xr.ones_like(ds.xCell)
        temperature_array, _ = xr.broadcast(temperature_array, ds.refZMid)
        ds['temperature'] = temperature_array.expand_dims(dim='Time', axis=0)
        ds['salinity'] = salinity * xr.ones_like(ds.temperature)

        print('~~~~Calculating ICs for btr jet...')
        jet_init(name='culled_mesh.nc',
                 save='jet_ic.nc',
                 pert=True)
        jet_ds = xr.open_dataset('jet_ic.nc')
        print('~~~~Done')

        normal_velocity, _ = xr.broadcast(jet_ds.u, ds.refBottomDepth)
        layer_thickness, _ = xr.broadcast(jet_ds.h, ds.refBottomDepth)

        ds['normalVelocity'] = normal_velocity
        ds['layerThickness'] = layer_thickness
        ds['ssh'] = jet_ds.h - ds['bottomDepth']
        ds['fCell'] = jet_ds.fCell
        ds['fEdge'] = jet_ds.fEdge
        ds['fVertex'] = jet_ds.fVertex

        ds.attrs['nx'] = nx
        ds.attrs['ny'] = ny
        ds.attrs['dc'] = dc

        write_netcdf(ds, 'initial_state.nc')

        cell_mask = ds.maxLevelCell >= 1

        plot_horiz_field(ds, ds_mesh, 'normalVelocity',
                         'initial_normal_velocity.png', cmap='cmo.balance',
                         show_patch_edges=True, cell_mask=cell_mask)
