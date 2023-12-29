from plane2d import Plane2d
from shapely import affinity, wkt
from shapely.geometry import Polygon
import numpy as np


from debugvisualizer.debugvisualizer import Plotter

np.random.seed(0)

def get_random_polygon() -> Polygon:
    """Creates and returns convex polygon randomly

    Returns:
        Polygon: Random polygon
    """
    
    vertices_count = np.random.randint(3, 10)

    points = np.random.rand(vertices_count, 2)

    scale_x = np.random.uniform(-10, 10)
    scale_y = np.random.uniform(-10, 10)

    points[:, 0] *= scale_x
    points[:, 1] *= scale_y

    return Polygon(points).convex_hull


if __name__ == "__main__":

    polygon = affinity.translate(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), *np.array([1, 1, 0]))
    plane2d = Plane2d(np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), normalize=True)

    plane_to_convert = Plane2d(np.array([-3, -7]), np.array([-1, 0]), np.array([0, -1]), normalize=True)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((-4 -8, -5 -8, -5 -9, -4 -9, -4 -8))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty


    plane_to_convert = Plane2d(np.array([5, 2]), np.array([1, 0]), np.array([0, -1]), normalize=True)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((6 1, 7 1, 7 0, 6 0, 6 1))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
    

    plane_to_convert = Plane2d(np.array([3, 6]), np.array([-1, 0]), np.array([0, 1]), normalize=True)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((2 7, 1.0000000000000002 7, 1.0000000000000002 8, 2 8, 2 7))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
    

    plane_to_convert = Plane2d(np.array([-6, -10]), np.array([-1, -1]), np.array([-1, 1]), normalize=True)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((-7.414213562373095 -10, -8.121320343559642 -10.707106781186548, -8.82842712474619 -10, -8.121320343559642 -9.292893218813452, -7.414213562373095 -10))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
    

    plane_to_convert = Plane2d(np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), normalize=True).get_rotated_plane(degree=45)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((0.0000000000000001 1.414213562373095, -0.7071067811865474 2.121320343559643, 0.0000000000000002 2.82842712474619, 0.7071067811865477 2.1213203435596424, 0.0000000000000001 1.414213562373095))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
    

    plane_to_convert = Plane2d(np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), normalize=True).get_rotated_plane(degree=33)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((-0.2940315329303978 1.383309602960451, -1.1327021008758222 1.9279486379754776, -0.5880630658607957 2.766619205920902, 0.2506075020846287 2.2219801709058755, -0.2940315329303978 1.383309602960451))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
    

    plane_to_convert = Plane2d(np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), normalize=True).get_rotated_plane(degree=173)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((1.1144154950464695 -0.8706768082361744, 2.1069616466877914 -0.7488074648310269, 2.228830990092939 -1.7413536164723489, 1.236284838451617 -1.8632229598774965, 1.1144154950464695 -0.8706768082361744))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
    

    plane_to_convert = Plane2d(np.array([-5, 13]), np.array([1, 0]), np.array([0, -1]), normalize=True).get_rotated_plane(degree=45)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((-5 11.585786437626904, -4.292893218813452 10.878679656440358, -5 10.17157287525381, -5.707106781186548 10.878679656440358, -5 11.585786437626904))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
    

    plane_to_convert = Plane2d(np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), normalize=True).get_rotated_plane(degree=45)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((1.414213562373095 -0.0000000000000003, 2.1213203435596424 -0.707106781186548, 2.82842712474619 -0.0000000000000007, 2.121320343559643 0.707106781186547, 1.414213562373095 -0.0000000000000003))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty


    plane_to_convert = Plane2d(np.array([10, 10]), np.array([1, 0]), np.array([0, 1]), normalize=True).get_rotated_plane(degree=118)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((10.413476030073037 8.647580844355183, 9.944004467287145 7.764633251496255, 10.826952060146072 7.2951616887103645, 11.296423622931963 8.178109281569292, 10.413476030073037 8.647580844355183))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty

    plane_original = plane2d.get_rotated_plane(45)
    plane_original.origin = np.array([-1, 5])
    plane_original.x_axis *= -1
    polygon = get_random_polygon()
    plane_to_convert = Plane2d(np.array([3, -7]), np.array([1, 0]), np.array([0, 1]), normalize=True).get_rotated_plane(degree=88)
    converted_geometry = plane_original.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((-0.5117602128615018 -3.6809365809282513, -0.7358235595010902 -5.76039395808481, -0.0413507673471366 -6.6115542951326525, 0.5401480664892073 -5.974998336640203, 0.9661191302429573 -5.482625991000526, 0.4557560584828839 -4.472534857048391, -0.5117602128615018 -3.6809365809282513))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
    
    plane_original = plane2d.get_rotated_plane(-13)
    plane_original.origin = np.array([1, 3])
    plane_original.y_axis *= -1
    polygon = get_random_polygon()
    
    plane_to_convert = Plane2d(np.array([-5, -8]), np.array([1, 0]), np.array([0, 1]), normalize=True).get_rotated_plane(degree=-55)
    converted_geometry = plane_original.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((-13.254843720024425 -0.4787358225352332, -11.914446850338816 -1.2465599592295984, -9.743143257334802 -2.3263047704692426, -13.254843720024425 -0.4787358225352332))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
    
    plane_original = plane2d.get_rotated_plane(270)
    plane_original.origin = np.array([10, 5])
    plane_original.x_axis *= -1
    plane_original.y_axis *= -1
    polygon = get_random_polygon()
    
    plane_to_convert = Plane2d(np.array([5, 2]), np.array([1, 0]), np.array([0, 1]), normalize=True).get_rotated_plane(degree=155)
    converted_geometry = plane_original.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((-5.3531602291787586 5.1631020913620524, -3.7666758127090016 6.469730707325887, -2.782717034528268 5.438440100263563, -5.3531602291787586 5.1631020913620524))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty