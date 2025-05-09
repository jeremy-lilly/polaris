import numpy as np
import pyproj
import xarray as xr
from geometric_features import FeatureCollection
from mpas_tools.cime.constants import constants
from mpas_tools.mesh.creation.signed_distance import (
    signed_distance_from_geojson,
)

from polaris.mesh import QuasiUniformSphericalMeshStep
from polaris.tasks.ocean.isomip_plus.mesh.xy import add_isomip_plus_xy
from polaris.tasks.ocean.isomip_plus.projection import get_projections


class SphericalMesh(QuasiUniformSphericalMeshStep):
    """
    A step for creating an ISOMIP+ mesh that is a small region on a sphere
    """

    def setup(self):
        """
        Add input files
        """
        self.add_output_file('base_mesh.nc')

        super().setup()

    def build_cell_width_lat_lon(self):
        """
        Create cell width array for this mesh on a regular latitude-longitude
        grid

        Returns
        -------
        cellWidth : numpy.array
            m x n array of cell width in km

        lon : numpy.array
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.array
            longitude in degrees (length m and between -90 and 90)
        """
        section = self.config['isomip_plus_mesh']

        dlon = 0.1
        dlat = dlon
        earth_radius = constants['SHR_CONST_REARTH']
        nlon = int(360.0 / dlon) + 1
        nlat = int(180.0 / dlat) + 1
        lon = np.linspace(-180.0, 180.0, nlon)
        lat = np.linspace(-90.0, 90.0, nlat)

        # this is the width of cells (in km) on the globe outside the domain of
        # interest, set to a coarse value to speed things up
        background_width = 100.0

        lx = section.getfloat('lx')
        ly = section.getfloat('ly')
        buffer = section.getfloat('buffer')
        fc = _make_feature(lx, ly, buffer)
        fc.to_geojson('isomip_plus_high_res.geojson')

        signed_distance = signed_distance_from_geojson(
            fc, lon, lat, earth_radius, max_length=0.25
        )

        # this is a distance (in m) over which the resolution coarsens outside
        # the domain of interest plus buffer
        trans_width = 1000e3
        weights = np.maximum(
            0.0, np.minimum(1.0, signed_distance / trans_width)
        )

        cell_width = (
            self.cell_width * (1 - weights) + background_width * weights
        )

        return cell_width, lon, lat

    def run(self):
        """
        Run this step of the test case
        """
        super().run()

        ds = xr.open_dataset('base_mesh_without_xy.nc')
        add_isomip_plus_xy(ds)
        ds.to_netcdf('base_mesh.nc')


def _make_feature(lx, ly, buffer):
    # a box with a buffer of 80 km surrounding the are of interest
    # (0 <= x <= 800) and (0 <= y <= 80)
    bounds = 1e3 * np.array((-buffer, lx + buffer, -buffer, ly + buffer))
    projection, lat_lon_projection = get_projections()
    transformer = pyproj.Transformer.from_proj(projection, lat_lon_projection)

    x = [bounds[0], bounds[1], bounds[1], bounds[0], bounds[0]]
    y = [bounds[2], bounds[2], bounds[3], bounds[3], bounds[2]]
    lon, lat = transformer.transform(x, y)

    coordinates = [[[lon[index], lat[index]] for index in range(len(lon))]]

    features = [
        {
            'type': 'Feature',
            'properties': {
                'name': 'ISOMIP+ high res region',
                'component': 'ocean',
                'object': 'region',
                'author': 'Polaris',
            },
            'geometry': {'type': 'Polygon', 'coordinates': coordinates},
        }
    ]

    fc = FeatureCollection(features=features)

    return fc
