from polaris import Task
from polaris.ocean.tasks.planar_barotropic_jet.forward import Forward


class Default(Task):
    """
    The default planar barotropic jet test case simply creates the mesh and
    initial condition, then performs a short forward run on 4 cores.
    """

    def __init__(self, component, resolution, indir, init):
        """
        Create the test case

        Parameters
        ----------
        component : polaris.ocean.Ocean
            The ocean component that this task belongs to

        resolution : float
            The resolution of the test case in km

        indir : str
            The directory the task is in, to which ``name`` will be appended

        init : polaris.ocean.tasks.planar_barotropic_jet.init.Init
            A shared step for creating the initial state
        """
        super().__init__(component=component, name='default', indir=indir)

        self.add_step(init, symlink='init')

        self.add_step(
            Forward(component=component, indir=self.subdir, ntasks=None,
                    min_tasks=None, openmp_threads=1, resolution=resolution,
                    graph_target=f'{init.path}/culled_graph.info'))
