from pyplasm import *

# Outer cylinder:
Rout = 0.10
Hout = 0.30
cyl_outer = CYLINDER(Rout, Hout)

# Inner cylinder
Rin = 0.08
bt = 0.005
cyl_inner = CYLINDER(Rin, Hout)
cyl_inner = T(cyl_inner, 0, 0, bt)

# Subtract inner cylinder from outer:
out = DIFF(cyl_outer, cyl_inner)

# Testing material properties:
alpha = 1
r = 1
g = 0
b = 0
ambient = [r*0.4,g*0.4,b*0.4,alpha]
diffuse = [r*0.6,g*0.6,b*0.6,alpha]
specular = [0,0,0,alpha]
emission = [0,0,0,alpha]
shine = 100
out = MATERIAL(ambient + diffuse + specular + emission + [shine])(out)

# Show the result:
VIEW(out)
