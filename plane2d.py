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
        self.font_size = 0.1
    
    def _get_polygonized_text(self, text: str, position: Tuple[float] = (0, 0)) -> Polygon:
        """Converts a given text string into a polygon representation of Shapely

        Args:
            text (str): The text to be polygonized
            font_size (float, optional): The size of the font for the text. Default is 1.0.
            position (Tuple[float, float], optional): The (x, y) position where the text starts. Default is (0, 0).

        Returns:
            Polygon: Polygonized text
        """

        tp = TextPath(position, text, size=self.font_size)

        vertices = tp.to_polygons()

        polygons = [Polygon(vertex) for vertex in vertices if len(vertex) > 2]
        
        if len(polygons) > 1:
            return ops.unary_union(polygons)

        else:
            return polygons[0]
    
    def _get_axes_geometries(
        self, 
        origin: np.ndarray, 
        x_axis: np.ndarray, 
        y_axis: np.ndarray, 
        x_degree: float, 
        y_degree: float
    ) -> GeometryCollection:
        """Generates geometries for the axes including labels and lines

        Args:
            origin (np.ndarray): The origin point of the axes
            x_axis (np.ndarray): The vector representing the x-axis
            y_axis (np.ndarray): The vector representing the y-axis
            x_degree (float): The rotation degree for the x-axis label
            y_degree (float): The rotation degree for the y-axis label

        Returns:
            GeometryCollection: Geometries visualizing the coordinates system
        """
        
        origin_text = self._get_polygonized_text(text="O")
        origin_text = origin_text - origin_text.buffer(-self.font_size / 10)
        
        x_axis_text = self._get_polygonized_text(text="X")
        y_axis_text = self._get_polygonized_text(text="Y")
        
        rotated_text_x = affinity.rotate(x_axis_text, x_degree, use_radians=False)
        rotated_text_y = affinity.rotate(y_axis_text, -y_degree, use_radians=False)

        translated_text_o = affinity.translate(origin_text, *origin)
        translated_text_x = affinity.translate(rotated_text_x, *(origin + x_axis))
        translated_text_y = affinity.translate(rotated_text_y, *(origin + y_axis))
        
        return GeometryCollection(
            [
                translated_text_o,
                translated_text_x,
                translated_text_y,
                LineString([origin, origin + x_axis * 0.9]),
                LineString([origin, origin + y_axis * 0.9]),
            ]
        )
    

class Plane2d(AxesVisualizer):
    def __init__(self, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, normalize: bool = True):
        self._origin = origin
        self._x_axis = x_axis / np.linalg.norm(x_axis) if normalize else x_axis
        self._y_axis = y_axis / np.linalg.norm(y_axis) if normalize else y_axis
        
        AxesVisualizer.__init__(self)
        
    @property
    def origin(self) -> np.ndarray:
        return self._origin

    @property
    def x_axis(self) -> np.ndarray:
        return self._x_axis

    @property
    def y_axis(self) -> np.ndarray:
        return self._y_axis

    @property
    def x_degree(self) -> float:
        return np.degrees(np.arctan2(self.x_axis[1], self.x_axis[0]))

    @property
    def y_degree(self) -> float:
        return np.degrees(np.arctan2(self.y_axis[0], self.y_axis[1]))
        
    @property
    def axes_geometries(self) -> GeometryCollection:
        return self._get_axes_geometries(
            origin=self.origin, 
            x_axis=self.x_axis, 
            y_axis=self.y_axis, 
            x_degree=self.x_degree, 
            y_degree=self.y_degree
        )
    
    @origin.setter
    def origin(self, value: np.ndarray) -> None:
        self._origin = value
    
    @x_axis.setter
    def x_axis(self, value: np.ndarray) -> None:
        self._x_axis = value
    
    @y_axis.setter
    def y_axis(self, value: np.ndarray) -> None:
        self._y_axis = value
    
    def _get_local_coords_matrix(self) -> np.ndarray:
        """Returns a local coordinate matrix expressed in homogeneous coordinates

        Returns:
            np.ndarray: local coordinates of self
        """
        
        return np.array(
            [
                [self.x_axis[0], self.y_axis[0], self.origin[0]],
                [self.x_axis[1], self.y_axis[1], self.origin[1]],
                [0, 0, 1],
            ]
        )
        
    def get_converted_geometry_by_plane(self, plane_to_convert: Plane2d, geometry_to_convert: GeometryCollection) -> GeometryCollection:
        """Converts a given geometry from one plane to another

        Args:
            plane_to_convert (Plane2d): Target plane to convert the coordinate system
            geometry_to_convert (GeometryCollection): Geometry of Shapely to convert

        Returns:
            GeometryCollection: Converted geometry
        """
        
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
    
    def get_rotated_plane(self, degree: float) -> Plane2d:
        """Returns a new Plane2d instance that is rotated by a given degree.

        Args:
            degree (float): Degree to rotate

        Returns:
            Plane2d: Rotated plane
        """
        
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