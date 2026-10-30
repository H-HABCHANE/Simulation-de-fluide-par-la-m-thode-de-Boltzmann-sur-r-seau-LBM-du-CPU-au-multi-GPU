// Mesh of the cylinder case of the LBM solver, for OpenFOAM, in units of the diameter (D = 1).
// LBM case: 1000 x 320 cells, D = 40 cells, cylinder centre at (200, 161) cells, i.e. x = 5 D
// and y = 4 D + 1 cell = 4.025 D (the one-cell offset that triggers vortex shedding), domain
// 25 D x 8 D, periodic in y. Here: smooth circular wall, unstructured triangles refined around
// the cylinder and in the wake, extruded by one cell (2D case), top and bottom periodic.
Lx = 25; Ly = 8; xc = 5; yc = 4.025; R = 0.5; dz = 0.1;
lc_far = 0.25;   // far field
lc_wake = 0.06;  // wake region
lc_cyl = 0.026;  // on the cylinder: about 120 cells around the circumference

Point(1) = {0, 0, 0, lc_far};
Point(2) = {Lx, 0, 0, lc_far};
Point(3) = {Lx, Ly, 0, lc_far};
Point(4) = {0, Ly, 0, lc_far};
Point(5) = {xc, yc, 0, lc_cyl};
Point(6) = {xc + R, yc, 0, lc_cyl};
Point(7) = {xc, yc + R, 0, lc_cyl};
Point(8) = {xc - R, yc, 0, lc_cyl};
Point(9) = {xc, yc - R, 0, lc_cyl};

Line(1) = {1, 2};  // bottom
Line(2) = {2, 3};  // outlet
Line(3) = {4, 3};  // top, same orientation as the bottom for the periodic constraint
Line(4) = {1, 4};  // inlet
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

Curve Loop(1) = {1, 2, -3, -4};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};

// Same nodes on the top and bottom edges, so that OpenFOAM can couple them (cyclic patches).
Periodic Curve {3} = {1} Translate {0, Ly, 0};

// Mesh size: fine on the cylinder, medium in the wake, coarse elsewhere.
Field[1] = Distance;
Field[1].CurvesList = {5, 6, 7, 8};
Field[1].Sampling = 200;
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc_cyl;
Field[2].SizeMax = lc_far;
Field[2].DistMin = 0.15;
Field[2].DistMax = 3;
Field[3] = Box;
Field[3].VIn = lc_wake;
Field[3].VOut = lc_far;
Field[3].XMin = 4.3;
Field[3].XMax = 21;
Field[3].YMin = 2.3;
Field[3].YMax = 5.7;
Field[3].Thickness = 1.5;
Field[4] = Min;
Field[4].FieldsList = {2, 3};
Background Field = 4;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

// One cell in z. The lateral surfaces come in the order of the boundary curves of surface 1:
// bottom, outlet, top, inlet, then the 4 arcs of the cylinder.
out[] = Extrude {0, 0, dz} { Surface{1}; Layers{1}; Recombine; };
Physical Surface("frontAndBack_raw") = {1, out[0]};
Physical Surface("bottom_raw") = {out[2]};
Physical Surface("outlet") = {out[3]};
Physical Surface("top_raw") = {out[4]};
Physical Surface("inlet") = {out[5]};
Physical Surface("cylinder_raw") = {out[6], out[7], out[8], out[9]};
Physical Volume("fluid") = {out[1]};
