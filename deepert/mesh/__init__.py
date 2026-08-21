"""Mesh primitives and meshio interoperability."""

from deepert.mesh.core import (
    Mesh,
    build_quadratic_triangle_mesh,
    cell_areas_2d,
    extract_boundary_edges,
    locate_points_in_quadrilaterals,
    locate_points_in_triangles,
    refine_triangle_mesh,
    triangle_areas,
)

__all__ = [
    "Mesh",
    "build_quadratic_triangle_mesh",
    "cell_areas_2d",
    "extract_boundary_edges",
    "locate_points_in_quadrilaterals",
    "locate_points_in_triangles",
    "refine_triangle_mesh",
    "triangle_areas",
]
