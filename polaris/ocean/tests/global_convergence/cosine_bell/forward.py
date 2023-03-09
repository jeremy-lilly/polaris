import time

from polaris.ocean.model import OceanModelStep


class Forward(OceanModelStep):
    """
    A step for performing forward ocean component runs as part of the cosine
    bell test case

    Attributes
    ----------
    resolution : int
        The resolution of the (uniform) mesh in km

    mesh_name : str
        The name of the mesh
    """

    def __init__(self, test_case, resolution, mesh_name):
        """
        Create a new step

        Parameters
        ----------
        test_case : polaris.ocean.tests.global_convergence.cosine_bell.CosineBell  # noqa: E501
            The test case this step belongs to

        resolution : int
            The resolution of the (uniform) mesh in km

        mesh_name : str
            The name of the mesh
        """
        super().__init__(test_case=test_case,
                         name=f'{mesh_name}_forward',
                         subdir=f'{mesh_name}/forward',
                         openmp_threads=1)

        self.resolution = resolution
        self.mesh_name = mesh_name

        # make sure output is double precision
        self.add_yaml_file('polaris.ocean.config', 'output.yaml')

        self.add_yaml_file(
            'polaris.ocean.tests.global_convergence.cosine_bell',
            'forward.yaml')

        self.add_input_file(filename='init.nc',
                            target='../init/initial_state.nc')
        self.add_input_file(filename='graph.info',
                            target='../mesh/graph.info')

        self.add_output_file(filename='output.nc')

    def setup(self):
        """
        Set namelist options base on config options
        """
        super().setup()
        dt = self.get_dt()
        self.add_model_config_options({'config_dt': dt})
        self._get_resources()

    def constrain_resources(self, available_cores):
        """
        Update resources at runtime from config options
        """
        self._get_resources()
        super().constrain_resources(available_cores)

    def runtime_setup(self):
        """
        Update the resources and time step in case the user has update config
        options
        """
        super().runtime_setup()

        # update dt in case the user has changed dt_per_km
        dt = self.get_dt()
        self.update_model_config_at_runtime(options={'config_dt': dt})

    def get_dt(self):
        """
        Get the time step

        Returns
        -------
        dt : str
            the time step in HH:MM:SS
        """
        config = self.config
        # dt is proportional to resolution: default 30 seconds per km
        dt_per_km = config.getint('cosine_bell', 'dt_per_km')

        dt = dt_per_km * self.resolution
        # https://stackoverflow.com/a/1384565/7728169
        dt = time.strftime('%H:%M:%S', time.gmtime(dt))

        return dt

    def _get_resources(self):
        mesh_name = self.mesh_name
        config = self.config
        self.ntasks = config.getint('cosine_bell', f'{mesh_name}_ntasks')
        self.min_tasks = config.getint('cosine_bell', f'{mesh_name}_min_tasks')
