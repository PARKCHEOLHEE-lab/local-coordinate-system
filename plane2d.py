from __future__ import annotations

import numpy as np
from typing import Tuple
from matplotlib.textpath import TextPath
from shapely.geometry import GeometryCollection
from shapely.geometry import Polygon, LineString
from shapely import ops, affinity, wkt

from debugvisualizer.debugvisualizer import Plotter


class AxesVisualizer:
    def __init__(self):
        pass
    
    def _get_polygonized_text(self, text: str, font_size: float = 1.0, position: Tuple[float] = (0, 0)):
        tp = TextPath(position, text, size=font_size)

        vertices = tp.to_polygons()

        polygons = [Polygon(vertex) for vertex in vertices if len(vertex) > 2]
        
        if len(polygons) > 1:
            return ops.unary_union(polygons)

        else:
            return polygons[0]
        
    def _get_axes_geometries(self, font_size: float, x_degree: float, y_degree: float):
        origin_text = self._get_polygonized_text(text="O", font_size=font_size)
        origin_text = origin_text - origin_text.buffer(-font_size / 10)
        
        x_axis_text = self._get_polygonized_text(text="X", font_size=font_size)
        y_axis_text = self._get_polygonized_text(text="Y", font_size=font_size)
        
        rotated_text_x = affinity.rotate(x_axis_text, x_degree, use_radians=False)
        rotated_text_y = affinity.rotate(y_axis_text, -y_degree, use_radians=False)

        translated_text_o = affinity.translate(origin_text, *self.origin)
        translated_text_x = affinity.translate(rotated_text_x, *(self.origin + self.x_axis))
        translated_text_y = affinity.translate(rotated_text_y, *(self.origin + self.y_axis))
        
        return (
            translated_text_o,
            translated_text_x,
            translated_text_y,
            LineString([self.origin, self.origin + self.x_axis * 0.9]),
            LineString([self.origin, self.origin + self.y_axis * 0.9]),
        )
    

class Plane2d(AxesVisualizer):
    def __init__(self, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, font_size: float = 0.1, normalize: bool = True):
        self.origin = origin
        self.x_axis = x_axis / np.linalg.norm(x_axis) if normalize else x_axis
        self.y_axis = y_axis / np.linalg.norm(y_axis) if normalize else y_axis
        self.font_size = font_size
        
        self.x_degree, self.y_degree = self._get_axes_degrees()
        self.axes_geometries = self._get_axes_geometries(
            font_size=font_size, x_degree=self.x_degree, y_degree=self.y_degree
        )

    def _get_axes_degrees(self) -> float:
        angle_x_axis = np.arctan2(self.x_axis[1], self.x_axis[0])
        angle_y_axis = np.arctan2(self.y_axis[0], self.y_axis[1])

        return np.degrees(angle_x_axis), np.degrees(angle_y_axis)
    
    def _get_local_coords_matrix(self):
        return np.array(
            [
                [self.x_axis[0], self.y_axis[0], self.origin[0]],
                [self.x_axis[1], self.y_axis[1], self.origin[1]],
                [0, 0, 1],
            ]
        )
        
    def get_converted_geometry_by_plane(self, plane_to_convert: Plane2d, geometry_to_convert: GeometryCollection) -> GeometryCollection:
        
        matrix_to_map_global_coords = np.linalg.inv(self._get_local_coords_matrix())
        matrix_to_map_local_coords = matrix_to_map_global_coords @ plane_to_convert._get_local_coords_matrix()
        matrix_to_map_local_coords_flattened = [
            matrix_to_map_local_coords[0][0], 
            matrix_to_map_local_coords[0][1],
            matrix_to_map_local_coords[1][0], 
            matrix_to_map_local_coords[1][1],
            matrix_to_map_local_coords[0][2],
            matrix_to_map_local_coords[1][2],
        ]
    
        return affinity.affine_transform(geometry_to_convert, matrix_to_map_local_coords_flattened)
    
    def get_rotated_plane(self, degree: float):
        cos_angle = np.cos(np.radians(degree))
        sin_angle = np.sin(np.radians(degree))
        
        rotation_matrix = np.array(
            [
                [cos_angle, sin_angle, 0],
                [-sin_angle, cos_angle, 0],
                [0, 0, 1]
            ]
        )
        
        rotated_x_axis = np.dot(rotation_matrix, np.array([*self.x_axis, 0]))[:2]
        rotated_y_axis = np.dot(rotation_matrix, np.array([*self.y_axis, 0]))[:2]
        
        return Plane2d(origin=self.origin, x_axis=rotated_x_axis, y_axis=rotated_y_axis)
    

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
