from __future__ import annotations

import numpy as np
from typing import Tuple
from matplotlib.textpath import TextPath
from shapely.geometry import GeometryCollection
from shapely.geometry import Polygon, LineString, Point
from shapely import ops, affinity
from debugvisualizer.debugvisualizer import Plotter


class Plane2D:
    def __init__(self, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, font_size: float = 0.1, normalize: bool = True):
        self.origin = origin
        self.x_axis = x_axis
        self.y_axis = y_axis
        
        self.normalize = normalize
        if self.normalize:
            self.x_axis = self.x_axis / np.linalg.norm(self.x_axis)
            self.y_axis = self.y_axis / np.linalg.norm(self.y_axis)
        
        self.x_degree, self.y_degree = self._calculate_axes_angles()
        self.axes_geometries = self._get_axes_geometries(font_size=font_size)
        
    def _get_text_polygon(self, text: str, font_size: float = 1.0, position: Tuple[float] = (0, 0)):
        tp = TextPath(position, text, size=font_size)

        vertices = tp.to_polygons()

        polygons = [Polygon(vertex) for vertex in vertices if len(vertex) > 2]
        
        if len(polygons) > 1:
            return ops.unary_union(polygons)

        else:
            return polygons[0]

    def _calculate_axes_angles(self) -> float:
        angle_x_axis = np.arctan2(self.x_axis[1], self.x_axis[0])
        angle_y_axis = np.arctan2(self.y_axis[0], self.y_axis[1])

        return np.degrees(angle_x_axis), np.degrees(angle_y_axis)
        
    def _get_axes_geometries(self, font_size: float):
        origin_text = self._get_text_polygon(text="O", font_size=font_size)
        origin_text = origin_text - origin_text.buffer(-font_size / 10)
        
        x_axis_text = self._get_text_polygon(text="X", font_size=font_size)
        y_axis_text = self._get_text_polygon(text="Y", font_size=font_size)
        
        rotated_text_x = affinity.rotate(x_axis_text, self.x_degree, use_radians=False)
        rotated_text_y = affinity.rotate(y_axis_text, -self.y_degree, use_radians=False)

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
        
    def _get_translation_matrix(self, dx: float, dy: float):
        return np.array(
            [
                [1, 0, dx],
                [0, 1, dy],
                [0, 0, 1]
            ]
        )
        
    def _get_rotation_matrix(self, degree: float):
        cos = np.cos(np.radians(degree))
        sin = np.sin(np.radians(degree))
        
        return np.array(
            [
                [cos, -sin, 0],
                [sin,  cos, 0],
                [0,  0, 1]
            ]
        )

    def _get_inversion_matrix(self, plane_to_convert: Plane2D):
        is_x_inverted = np.dot(plane_to_convert.x_axis, self.x_axis) < 0
        is_y_inverted = np.dot(plane_to_convert.y_axis, self.y_axis) < 0
        
        if is_x_inverted and not is_y_inverted:
            return np.array(
                [
                    [-1, 0, 0],
                    [ 0, 1, 0],
                    [ 0, 0, 1]
                ]
            )

        elif not is_x_inverted and is_y_inverted:
            return np.array(
                [
                    [ 1, 0, 0],
                    [ 0,-1, 0],
                    [ 0, 0, 1]
                ]
            )

        else:
            return np.array(
                [
                    [ 1, 0, 0],
                    [ 0, 1, 0],
                    [ 0, 0, 1]
                ]
            )
            
    def _get_transformation_matrix(self, plane_to_convert: Plane2D):
        
        rotation_matrix = self._get_rotation_matrix(plane_to_convert.x_degree)
        translation_matrix = self._get_translation_matrix(*plane_to_convert.origin)
        inversion_matrix = self._get_inversion_matrix(plane_to_convert)
        
        return translation_matrix @ inversion_matrix @ rotation_matrix
        
    def convert_plane_to_plane(self, plane_to_convert: Plane2D, geometry_to_convert: GeometryCollection) -> GeometryCollection:
        
        transformation_matrix = self._get_transformation_matrix(plane_to_convert)
        
        # Convert geometry to homogeneous coordinates
        if isinstance(geometry_to_convert, Point):
            coords = np.array([geometry_to_convert.x, geometry_to_convert.y, 1])

        elif isinstance(geometry_to_convert, LineString):
            coords = np.array([list(point) + [1] for point in geometry_to_convert.coords])
            
        elif isinstance(geometry_to_convert, Polygon):
            coords = np.array([list(point) + [1] for point in geometry_to_convert.boundary.coords])

        # Apply the transformation
        transformed_coords = np.dot(transformation_matrix, coords.T)

        # Convert back from homogeneous coordinates
        if isinstance(geometry_to_convert, Point):
            converted_geometry = Point(transformed_coords[:2])
        elif isinstance(geometry_to_convert, LineString):
            converted_geometry =  LineString(transformed_coords[:2].T)
        elif isinstance(geometry_to_convert, Polygon):
            converted_geometry = Polygon(transformed_coords[:2].T)
        
        return converted_geometry
        
if __name__ == "__main__":

    point = Point(1, 1)
    line = LineString([(0, 0), (2, 2)])
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    plane_a = Plane2D(np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), normalize=True)

    plane_to_convert = Plane2D(np.array([0, 0]), np.array([-1, 0]), np.array([0, -1]), normalize=True)
    a = plane_a.convert_plane_to_plane(plane_to_convert, point)
    # assert a == Point(-3, -3)
    # b = plane_a.convert_plane_to_plane(plane_to_convert, line)
    # c = plane_a.convert_plane_to_plane(plane_to_convert, polygon)

    plane_to_convert = Plane2D(np.array([0, 0]), np.array([1, 0]), np.array([0, -1]), normalize=True)
    a = plane_a.convert_plane_to_plane(plane_to_convert, point)
    # assert a == Point(-1, -3)
    # b = plane_a.convert_plane_to_plane(plane_to_convert, line)
    # c = plane_a.convert_plane_to_plane(plane_to_convert, polygon)

    plane_to_convert = Plane2D(np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), normalize=True)
    a = plane_a.convert_plane_to_plane(plane_to_convert, point)
    # assert a == Point(-3, -1)
    # b = plane_a.convert_plane_to_plane(plane_to_convert, line)
    # c = plane_a.convert_plane_to_plane(plane_to_convert, polygon)

    # plane_to_convert = Plane2D(np.array([-1, -3]), np.array([1, 1]), np.array([1, -1]), normalize=True)
    # a = plane_a.convert_plane_to_plane(plane_to_convert, point)
    # b = plane_a.convert_plane_to_plane(plane_to_convert, line)
    # c = plane_a.convert_plane_to_plane(plane_to_convert, polygon)

    # plane_to_convert = Plane2D(np.array([-1, -3]), np.array([1, 0]), np.array([0, -1]))
    # a = plane_a.convert_plane_to_plane(plane_to_convert, point)
    # b = plane_a.convert_plane_to_plane(plane_to_convert, line)
    # c = plane_a.convert_plane_to_plane(plane_to_convert, polygon)

    # plane_to_convert = Plane2D(np.array([0, 0]), np.array([-1, 0]), np.array([0, 1]), normalize=True)
    # a = plane_a.convert_plane_to_plane(plane_to_convert, point)
    # b = plane_a.convert_plane_to_plane(plane_to_convert, line)
    # c = plane_a.convert_plane_to_plane(plane_to_convert, polygon)

    # plane_to_convert = Plane2D(np.array([-1, -1]), np.array([-1, -1]), np.array([-1, 1]), normalize=True)
    # a = plane_a.convert_plane_to_plane(plane_to_convert, point)
    # b = plane_a.convert_plane_to_plane(plane_to_convert, line)
    # c = plane_a.convert_plane_to_plane(plane_to_convert, polygon)

    # plane_to_convert = Plane2D(np.array([-1, -1]), np.array([-1, -1]), np.array([1, -1]), normalize=True)
    # a = plane_a.convert_plane_to_plane(plane_to_convert, point)
    # b = plane_a.convert_plane_to_plane(plane_to_convert, line)
    # c = plane_a.convert_plane_to_plane(plane_to_convert, polygon)
    
    