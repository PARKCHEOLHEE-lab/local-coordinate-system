from __future__ import annotations

import numpy as np
from typing import Tuple
from matplotlib.textpath import TextPath
from shapely.geometry import GeometryCollection
from shapely.geometry import Polygon, LineString
from shapely import ops, affinity

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
    
    def _get_axes_geometries(
        self, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, font_size: float, x_degree: float, y_degree: float
    ):
        origin_text = self._get_polygonized_text(text="O", font_size=font_size)
        origin_text = origin_text - origin_text.buffer(-font_size / 10)
        
        x_axis_text = self._get_polygonized_text(text="X", font_size=font_size)
        y_axis_text = self._get_polygonized_text(text="Y", font_size=font_size)
        
        rotated_text_x = affinity.rotate(x_axis_text, x_degree, use_radians=False)
        rotated_text_y = affinity.rotate(y_axis_text, -y_degree, use_radians=False)

        translated_text_o = affinity.translate(origin_text, *origin)
        translated_text_x = affinity.translate(rotated_text_x, *(origin + x_axis))
        translated_text_y = affinity.translate(rotated_text_y, *(origin + y_axis))
        
        return (
            translated_text_o,
            translated_text_x,
            translated_text_y,
            LineString([origin, origin + x_axis * 0.9]),
            LineString([origin, origin + y_axis * 0.9]),
        )
    

class Plane2d(AxesVisualizer):
    def __init__(self, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, font_size: float = 0.1, normalize: bool = True):
        self._origin = origin
        self._x_axis = x_axis / np.linalg.norm(x_axis) if normalize else x_axis
        self._y_axis = y_axis / np.linalg.norm(y_axis) if normalize else y_axis

        self.font_size = font_size
        
    @property
    def origin(self):
        return self._origin

    @property
    def x_axis(self):
        return self._x_axis

    @property
    def y_axis(self):
        return self._y_axis
        
    @property
    def axes_geometries(self):
        return self._get_axes_geometries(
            origin=self.origin, 
            x_axis=self.x_axis, 
            y_axis=self.y_axis, 
            font_size=self.font_size, 
            x_degree=self.x_degree, 
            y_degree=self.y_degree
        )

    @property
    def x_degree(self):
        return np.degrees(np.arctan2(self.x_axis[1], self.x_axis[0]))

    @property
    def y_degree(self):
        return np.degrees(np.arctan2(self.y_axis[0], self.y_axis[1]))
    
    @origin.setter
    def origin(self, value: np.ndarray):
        self._origin = value
    
    @x_axis.setter
    def x_axis(self, value: np.ndarray):
        self._x_axis = value
    
    @y_axis.setter
    def y_axis(self, value: np.ndarray):
        self._y_axis = value
    
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
        matrix_to_map_local_coords = plane_to_convert._get_local_coords_matrix() @ matrix_to_map_global_coords
    
        return affinity.affine_transform(
            geom=geometry_to_convert,
            matrix=[
                matrix_to_map_local_coords[0][0], 
                matrix_to_map_local_coords[0][1],
                matrix_to_map_local_coords[1][0], 
                matrix_to_map_local_coords[1][1],
                matrix_to_map_local_coords[0][2],
                matrix_to_map_local_coords[1][2],
            ]
        )
    
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