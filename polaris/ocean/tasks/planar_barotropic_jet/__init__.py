from polaris.config import PolarisConfigParser
from polaris.ocean.resolution import resolution_to_subdir
from polaris.ocean.tasks.planar_barotropic_jet.default import Default
from polaris.ocean.tasks.planar_barotropic_jet.init import Init


def add_planar_barotropic_jet_tasks(component):
    """
    Add tasks for different planar barotopic jet tests to the ocean component

    component : polaris.ocean.Ocean
        the ocean component that the tasks will be added to
    """
    for resolution in [240., 120., 60.]:
        resdir = resolution_to_subdir(resolution)
        resdir = f'planar/planar_barotropic_jet/{resdir}'

        config_filename = 'planar_barotropic_jet.cfg'
        config = PolarisConfigParser(filepath=f'{resdir}/{config_filename}')
        config.add_from_package('polaris.ocean.tasks.planar_barotropic_jet',
                                'planar_barotropic_jet.cfg')

        init = Init(component=component, resolution=resolution, indir=resdir)
        init.set_shared_config(config, link=config_filename)

        default = Default(component=component, resolution=resolution,
                          indir=resdir, init=init)
        default.set_shared_config(config, link=config_filename)
        component.add_task(default)
