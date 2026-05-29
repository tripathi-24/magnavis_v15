'''
Created on Jan 19, 2025

@author: Admin
'''
import sys
import vtk
from vtk.util import numpy_support
import numpy
from pygeomag import GeoMag
geo_mag = GeoMag(coefficients_file="wmm/WMMHR_2025.COF", high_resolution=True)


map_meta = {
                'start_lat': 5.0,
                'end_lat': 40.0,
                'start_lon': 65.0,
                'end_lon': 100.0 
            }

map_extent = -200, 200, -200, 200,-200, 200

def get_map_xy_from_latlon(lat, lon):
    lat_min = map_meta['start_lat']
    lat_max = map_meta['end_lat']
    lon_min = map_meta['start_lon']
    lon_max = map_meta['end_lon']
    x_min, x_max, y_min, y_max, z_, z__ = map_extent
    x = x_min + (x_max - x_min)*(lon - lon_min)/(lon_max - lon_min)
    y = y_min + (y_max - y_min)*(lat - lat_min)/(lat_max - lat_min)
    return (x,y)

def get_map_lat_lon_from_xy(x, y):
    lat_min = map_meta['start_lat']
    lat_max = map_meta['end_lat']
    lon_min = map_meta['start_lon']
    lon_max = map_meta['end_lon']
    x_min, x_max, y_min, y_max, z_, z__ = map_extent
    lon = lon_min + (lon_max - lon_min)*(x - x_min)/(x_max - x_min)
    lat = lat_min + (lat_max - y_min)*(y - y_min)/(y_max - y_min)
    return (lat, lon)
    
def load_vtk_file(path: str):
    # reader = vtk.vtkPolyDataReader()
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(path)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    return reader

# reader = load_vtk_file('cube_vista.vtk')
# reader = load_vtk_file('GFG.msh')
reader = load_vtk_file('salome_cube_world_res_10.vtk')
 
data = reader.GetOutput()

###
# create the desired vtkPoltData object from points and faces
# if data.__class__.__name__ == 'vtkUnstructuredGrid':
#     polydata = vtk.vtkPolyData()
#     cells = data.GetCells()
#     polydata.SetPoints(points)
#     polydata.SetPolys(cells)
###

npts = data.GetNumberOfPoints()

def scalar_field_function(x, y, z):
    return 10*x*x + 5*y*y + z

def magnetic_field_function(x, y, z):
    lat, lon = get_map_lat_lon_from_xy(x, y)
    result = geo_mag.calculate(glat=lat, glon=lon, alt=z, time=2025.25)
    return float(result.f) # f = total intensity
    # d = declination (magnetic variation)
    
# field_scalars = []
#
# for _ in range(npts):
#     _x, _y, _z = data.GetPoint(_)
#     field_mag = scalar_field_function(_x, _y, _z)
#     field_scalars.append(field_mag)
#
# arr = numpy.array(field_scalars)
# vtkarr = numpy_support.numpy_to_vtk( arr.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
# vtkarr.SetName("AddedScalarField")


field_scalars = []

local_scalars = []
k_ = 0

for _ in range(npts):
    _x, _y, _z = data.GetPoint(_)
    # field_mag = scalar_field_function(_x, _y, _z)
    field_mag = magnetic_field_function(_x, _y, _z)
    local_scalars.append(field_mag)
    k_ = int(_ //1000)
    if _ %1000 == 0:
        print('added', _, 'out of', npts)
        field_scalars.append(local_scalars)
        # print('local_scalars', local_scalars)
        local_scalars = []
        # print(field_scalars[:5]
if local_scalars:
    field_scalars.append(local_scalars)
    

full_arr = []
for ar in field_scalars:
    full_arr.extend(ar) # allocate new
    # ar.clear() # delete old
    print('size, kb:', sys.getsizeof(full_arr) / 1024)
arr = numpy.array(full_arr)
print('len full_arr', len(full_arr), 'len arr', len(arr))
vtkarr = numpy_support.numpy_to_vtk( arr.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
vtkarr.SetName("WMM2025_MagneticField")
print('noof component', vtkarr.GetNumberOfComponents())

data.GetPointData().AddArray(vtkarr)
# vtkarr = vtk.vtkFloatArray() # the VTK array
# vtkarr.SetNumberOfComponents(1)
# imdata.GetPointData().SetScalars(vtkarr) ?


#Write in file
if data.__class__.__name__ == 'vtkUnstructuredGrid':
    w = vtk.vtkUnstructuredGridWriter()
else:
    w = vtk.vtkPolyDataWriter()
w.SetInputDataObject(data)
# w.SetFileName("cube_vista_with_scalarfield.vtk")
w.SetFileName("salome_cube_world_res_10_with_magneticfield.vtk")
w.Write()

if __name__ == '__main__':
    pass