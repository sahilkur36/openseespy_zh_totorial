#!/usr/bin/env python
# coding: utf-8
# 例子来源于https://opstool.readthedocs.io/en/stable/examples/post/forma11c/test_forma11c.html

import gmsh
import math
import json
from pathlib import Path

# Initialize gmsh
gmsh.initialize()

gmsh.model.add("forma11c_gmsh")

# Read the profile coordinates
file_id = open(Path(__file__).parent / "forma11c_profile.json", "r")
coords = json.load(file_id)
file_id.close()

# Set a default element size
el_size = 1.0

# Add profile points
v_profile = []
for coord in coords:
    v = gmsh.model.occ.addPoint(coord[0], coord[1], coord[2], el_size)
    v_profile.append(v)

# Add spline going through profile points
l1 = gmsh.model.occ.addBSpline(v_profile)
# Create copies and rotate
l2 = gmsh.model.occ.copy([(1, l1)])
l3 = gmsh.model.occ.copy([(1, l1)])
l4 = gmsh.model.occ.copy([(1, l1)])

# Rotate the copy
gmsh.model.occ.rotate(l2, 0, 0, 0, 0, 0, 1, math.pi / 2)
gmsh.model.occ.rotate(l3, 0, 0, 0, 0, 0, 1, math.pi)
gmsh.model.occ.rotate(l4, 0, 0, 0, 0, 0, 1, 3 * math.pi / 2)

# Sweep the lines
surf1 = gmsh.model.occ.revolve([(1, l1)], 0, 0, 0, 0, 0, 1, math.pi / 2)
surf2 = gmsh.model.occ.revolve(l2, 0, 0, 0, 0, 0, 1, math.pi / 2)
surf3 = gmsh.model.occ.revolve(l3, 0, 0, 0, 0, 0, 1, math.pi / 2)
surf4 = gmsh.model.occ.revolve(l4, 0, 0, 0, 0, 0, 1, math.pi / 2)

# Join the surfaces
surf5 = gmsh.model.occ.fragment(surf1, surf2)
surf6 = gmsh.model.occ.fragment(surf3, surf4)
surf7 = gmsh.model.occ.fragment(surf5[0], surf6[0])

gmsh.model.occ.remove_all_duplicates()
gmsh.model.occ.synchronize()

num_nodes_circ = 15
for curve in gmsh.model.occ.getEntities(1):
    gmsh.model.mesh.setTransfiniteCurve(curve[1], num_nodes_circ)

num_nodes_vert = 32
vertical_curves = [7, 10, 13, 17]
for curve in vertical_curves:
    gmsh.model.mesh.setTransfiniteCurve(curve, num_nodes_vert)

for surf in gmsh.model.occ.getEntities(2):
    gmsh.model.mesh.setTransfiniteSurface(surf[1])

gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
gmsh.option.setNumber("Mesh.Recombine3DLevel", 2)
gmsh.option.setNumber("Mesh.ElementOrder", 1)

# Important:
# Note that we use names to distinguish groups, so please do not overlook this!
# We use the "Boundary" group to include 4 lines
gmsh.model.addPhysicalGroup(dim=1, tags=[6, 9, 12, 15], tag=1, name="Boundary")

# Generate mesh
gmsh.model.mesh.generate(dim=2)

gmsh.option.setNumber("Mesh.SaveAll", 1)
mesh_file = Path(__file__).parent / "tower.msh"
gmsh.write(str(mesh_file))

gmsh.fltk.run()