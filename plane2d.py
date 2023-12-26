from __future__ import annotations

import numpy as np
from typing import Tuple
from matplotlib.textpath import TextPath
from shapely.geometry import GeometryCollection
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from shapely import ops, affinity, wkt
from debugvisualizer.debugvisualizer import Plotter


class AxesVisualizer:
    def __init__(self):
        pass
    
    def _get_text_polygon(self, text: str, font_size: float = 1.0, position: Tuple[float] = (0, 0)):
        tp = TextPath(position, text, size=font_size)

        vertices = tp.to_polygons()

        polygons = [Polygon(vertex) for vertex in vertices if len(vertex) > 2]
        
        if len(polygons) > 1:
            return ops.unary_union(polygons)

        else:
            return polygons[0]
        
    def _get_axes_geometries(self, font_size: float, x_degree: float, y_degree: float):
        origin_text = self._get_text_polygon(text="O", font_size=font_size)
        origin_text = origin_text - origin_text.buffer(-font_size / 10)
        
        x_axis_text = self._get_text_polygon(text="X", font_size=font_size)
        y_axis_text = self._get_text_polygon(text="Y", font_size=font_size)
        
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
        self.x_axis = x_axis
        self.y_axis = y_axis
        
        self.font_size = font_size
        self.normalize = normalize
        
        if self.normalize:
            self.x_axis = self.x_axis / np.linalg.norm(self.x_axis)
            self.y_axis = self.y_axis / np.linalg.norm(self.y_axis)
        
        self.x_degree, self.y_degree = self._calculate_axes_angles()
        self.axes_geometries = self._get_axes_geometries(
            font_size=font_size, x_degree=self.x_degree, y_degree=self.y_degree
        )

    def _calculate_axes_angles(self) -> float:
        angle_x_axis = np.arctan2(self.x_axis[1], self.x_axis[0])
        angle_y_axis = np.arctan2(self.y_axis[0], self.y_axis[1])

        return np.degrees(angle_x_axis), np.degrees(angle_y_axis)
        
    def _get_translation_matrix(self, dx: float, dy: float):
        return np.array(
            [
                [1, 0, dx],
                [0, 1, dy],
                [0, 0, 1]
            ]
        )
        
    def _get_rotation_matrix(self, degree: float):
        cos_angle = np.cos(np.radians(degree))
        sin_angle = np.sin(np.radians(degree))
        
        return np.array(
            [
                [cos_angle, sin_angle, 0],
                [-sin_angle, cos_angle, 0],
                [0, 0, 1]
            ]
        )

    def _get_reflection_matrix(self, rotation_matrix: np.ndarray, plane_to_convert: Plane2d):
        
        y_reflection_needed = np.dot(
            np.dot(rotation_matrix, np.array([*plane_to_convert.y_axis, 0])), np.array([*self.y_axis, 0])
        ) < 0
        
        x_reflection_needed = np.dot(
            np.dot(rotation_matrix, np.array([*plane_to_convert.x_axis, 0])), np.array([*self.x_axis, 0])
        ) < 0

        x = -1 if x_reflection_needed else 1 
        y = -1 if y_reflection_needed else 1 
        
        return np.array(
            [
                [x, 0, 0],
                [0, y, 0],
                [0, 0, 1]
            ]
        )

    def _get_transformation_matrix(self, plane_to_convert: Plane2d):
        
        rotation_matrix = self._get_rotation_matrix(plane_to_convert.x_degree)
        translation_matrix = self._get_translation_matrix(*plane_to_convert.origin)        
        reflection_matrix = self._get_reflection_matrix(rotation_matrix, plane_to_convert)
        
        transformation_matrix = translation_matrix @ reflection_matrix @ rotation_matrix
        
        return transformation_matrix
        
    def get_converted_geometry_by_plane(self, plane_to_convert: Plane2d, geometry_to_convert: GeometryCollection) -> GeometryCollection:
        
        transformation_matrix = self._get_transformation_matrix(plane_to_convert)
        transformation_matrix_flattened = [
            transformation_matrix[0][0],  # a 
            transformation_matrix[0][1],  # b
            transformation_matrix[1][0],  # d 
            transformation_matrix[1][1],  # e
            transformation_matrix[0][2],  # xoff
            transformation_matrix[1][2],  # yoff
        ]
    
        return affinity.affine_transform(geometry_to_convert, transformation_matrix_flattened)
    
    def get_rotated_plane(self, degree: float):
        rotation_matrix = self._get_rotation_matrix(degree)
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
    

    plane_to_convert = Plane2d(np.array([0, 0]), np.array([1, 0]), np.array([0, -1]), normalize=True).get_rotated_plane(degree=45)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((0 -1.4142135623730951, 0.7071067811865476 -2.121320343559643, 0 -2.8284271247461903, -0.7071067811865476 -2.121320343559643, 0 -1.4142135623730951))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
    

    plane_to_convert = Plane2d(np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), normalize=True).get_rotated_plane(degree=45)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((1.414213562373095 -0.0000000000000003, 2.1213203435596424 -0.707106781186548, 2.82842712474619 -0.0000000000000007, 2.121320343559643 0.707106781186547, 1.414213562373095 -0.0000000000000003))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty


    plane_to_convert = Plane2d(np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), normalize=True).get_rotated_plane(degree=118)
    converted_geometry = plane2d.get_converted_geometry_by_plane(plane_to_convert, polygon)
    expected_geometry = wkt.loads('POLYGON ((0.4134760300730362 -1.3524191556448177, -0.0559955327128545 -2.2353667485037447, 0.8269520601460725 -2.7048383112896355, 1.2964236229319632 -1.8218907184307085, 0.4134760300730362 -1.3524191556448177))')
    assert (expected_geometry - converted_geometry.buffer(1e-5)).is_empty
