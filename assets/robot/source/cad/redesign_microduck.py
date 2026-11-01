"""Run inside FreeCAD to rebuild the reference-inspired MicroDuck concept (mm).

Appearance concept only: joint travel, shell clearance, wall thickness and
hardware fit require engineering before fabrication. The training URDF is separate.
"""
import math
import os
import FreeCAD as App
import FreeCADGui as Gui
import Part

ROOT = os.path.dirname(os.path.abspath(__file__))
V = App.Vector
doc = App.newDocument('MicroDuck_Futuristic')
graphite = (0.22, 0.25, 0.28)
dark = (0.055, 0.068, 0.08)
black = (0.025, 0.035, 0.045)
metal = (0.62, 0.66, 0.68)
yellow = (1.0, 0.72, 0.055)
purple = (0.36, 0.26, 0.48)
cyan = (0.13, 0.81, 0.9)
groups = {}

def add(name, shape, color, group):
    if group not in groups:
        groups[group] = doc.addObject('App::DocumentObjectGroup', group)
    obj = doc.addObject('PartDesign::Feature', name)
    obj.Shape = shape
    obj.ViewObject.ShapeColor = color
    obj.ViewObject.LineColor = color
    obj.ViewObject.DisplayMode = 'Flat Lines'
    obj.ViewObject.LineWidth = 1.0
    obj.ViewObject.AngularDeflection = 8
    obj.ViewObject.Deviation = 0.08
    obj.ViewObject.ShapeMaterial.Shininess = 65 if color == metal else 25
    groups[group].addObject(obj)
    return obj

def rounded(x, y, z, dx, dy, dz, r):
    shape = Part.makeBox(dx, dy, dz, V(x, y, z))
    return shape.makeFillet(r, shape.Edges)

def cyl(center, radius, length, direction=(0, 1, 0)):
    return Part.makeCylinder(radius, length, V(*center), V(*direction))

def beam(a, b, width, depth, radius=1.4):
    a, b = V(*a), V(*b)
    shape = rounded(-width/2, -depth/2, 0, width, depth, (b-a).Length, radius)
    shape.Placement = App.Placement(a, App.Rotation(V(0, 0, 1), b-a))
    return shape

def cable(name, points, radius, group):
    curve = Part.BSplineCurve()
    curve.interpolate([V(*p) for p in points])
    path = Part.Wire(curve.toShape())
    tangent = curve.tangent(curve.FirstParameter)[0]
    section = Part.Wire([Part.makeCircle(radius, V(*points[0]), tangent)])
    add(name, path.makePipeShell([section], True, False), black, group)

def bolt(name, center, axis, group, radius=1.35):
    ax = V(*axis)
    body = cyl(center, radius, 0.8, axis)
    slot = Part.makeBox(radius*1.15, 0.4, 0.55, V(-radius*.575, -.2, .4))
    slot.Placement = App.Placement(V(*center), App.Rotation(V(0,0,1), ax)).multiply(slot.Placement)
    add(name, body.cut(slot), metal, group)

# Broad arched head, with a single forward-facing camera and recessed visor.
def arch(x, depth, width, bottom, shoulder, top):
    a, b = V(x,-width/2,bottom), V(x,width/2,bottom)
    c, d = V(x,width/2,shoulder), V(x,-width/2,shoulder)
    edges = [Part.makeLine(a,b), Part.makeLine(b,c),
             Part.Arc(c,V(x,0,top),d).toShape(),Part.makeLine(d,a)]
    return Part.Face(Part.Wire(edges)).extrude(V(depth,0,0))

shell = arch(-30, 56, 86, 177, 199, 224)
shell = shell.makeFillet(2.4, shell.Edges)
shell = shell.cut(arch(-31, 3.8, 77, 182, 198, 219))
add('Head_arched_housing', shell, graphite, 'Head')
add('Recessed_sensor_visor', arch(-27.5, 1.0, 77, 182, 198, 219), black, 'Head')
add('Visor_faceplate', arch(-28.0, .8, 71, 185, 198, 216), dark, 'Head')
add('Lower_head_gasket', rounded(-29,-42,175.4,54,84,3.2,1.3), black, 'Head')
add('Yellow_beak_upper', rounded(-38,-44,171,66,88,6.5,2.8), yellow, 'Head')
add('Beak_mouth_recess', rounded(-38.3,-37,172.1,2.0,74,2.7,.8), black, 'Head')
add('Lower_beak_violet_insert', rounded(-35,-39,169,54,78,2.2,.8), purple, 'Head')
for name, x, r, depth, color in [
    ('Camera_machined_bezel',-31.5,13,3.5,metal),
    ('Camera_yellow_ring',-32.4,11.7,2.8,yellow),
    ('Camera_black_barrel',-33.2,8.5,3.0,black),
    ('Camera_optical_glass',-33.5,6.4,1.0,(.055,.17,.23)),
    ('Camera_inner_aperture',-33.7,3.5,.6,black)]:
    add(name,cyl((x,0,199),r,depth,(1,0,0)),color,'Head')
add('Lens_reflection',Part.makeSphere(1.1,V(-34.0,-2.5,202)),(.65,.88,.94),'Head')
for y in [-25,25]:
    add('Status_light',cyl((-28.8,y,191),1.65,1.2,(1,0,0)),cyan,'Head')
for side in [-1,1]:
    y = side*43
    add('Head_side_yoke',beam((3,y,157),(0,y,190),7,3,1.2),yellow,'Head')
    add('Head_pivot',cyl((0,y-side*2,189),5,5,(0,side,0)),dark,'Head')
    bolt('Head_pivot_screw',(0,y+side*3,189),(0,side,0),'Head',1.9)
for j in range(5):
    add('Head_rear_cooling_slot',rounded(25.8,-19+j*8,187,1,3.2,14,.45),black,'Head')

# Compact chest and an exposed two-segment neck.
add('Torso_lower_chassis',rounded(-21,-30,101,43,60,10,3),dark,'Torso')
add('Torso_graphite_shell',rounded(-25,-35,109,50,70,22,6),graphite,'Torso')
add('Torso_top_inset',rounded(-18,-24,128,36,48,4,1.5),black,'Torso')
add('Torso_front_trim',rounded(-25.4,-26,113,2.2,52,3.3,.9),yellow,'Torso')
add('Torso_status_bar',rounded(-26,-13,120,1.2,26,1.4,.45),cyan,'Torso')
add('Neck_lower_servo',rounded(-8,-10,132,18,20,21,2),dark,'Neck')
add('Neck_upper_servo',rounded(1,-10,150,18,20,20,2),dark,'Neck')
for y in [-12,10]:
    add('Neck_linkage',beam((0,y,138),(9,y,164),7,2,0.65),metal,'Neck')
    for x,z in [(0,138),(9,164)]:
        add('Neck_horn',cyl((x,y,z),4,2),metal,'Neck')
        bolt('Neck_fastener',(x,y-0.85 if y<0 else y+2,z),(0,-1 if y<0 else 1,0),'Neck')
for y in [-8,-5,6]:
    cable('Neck_wiring',[(12,y,130),(22,y,143),(22,y,158),(14,y,172)],.7,'Neck')

# Six visually articulated leg actuators; rearward knee stance.
for side, prefix in [(-1,'Left'),(1,'Right')]:
    group = prefix+'_Leg'
    y = side*34
    hip, knee, ankle = (-1,y,102),(16,y,65),(-5,y,21)
    add(prefix+'_hip_roll_servo',rounded(-12,y-11,92,24,22,16,2),dark,group)
    add(prefix+'_hip_pitch_servo',beam((-1,y,84),(-1,y,104),21,23,2),black,group)
    add(prefix+'_thigh_motor',beam((4,y,80),knee,20,20,2),dark,group)
    add(prefix+'_thigh_inner_rail',beam(hip,knee,10,12,1.3),metal,group)
    add(prefix+'_shin_motor',beam((-5,y,17),(3,y,38),18,20,1.8),dark,group)
    for offset in [-9,7]:
        add(prefix+'_shin_link',beam((16,y+offset,65),(-5,y+offset,24),7,2,0.7),metal,group)
    # Sculpted triangular side cover with rounded corners, outside each thigh.
    yy = y+side*12.5
    pts = [V(-10,yy,101),V(28,yy,83),V(18,yy,60),V(-10,yy,101)]
    cover = Part.Face(Part.makePolygon(pts)).extrude(V(0,side*3,0))
    cover = cover.makeFillet(1.25,cover.Edges)
    add(prefix+'_thigh_armor',cover,graphite,group)
    for x,z,r in [(-1,100,8),(16,65,7),(-5,22,6)]:
        add(prefix+'_joint_housing',cyl((x,y-side*11,z),r,22,(0,side,0)),black,group)
        add(prefix+'_joint_alloy_ring',cyl((x,y+side*11,z),r-1,1.6,(0,side,0)),metal,group)
        add(prefix+'_joint_hub',cyl((x,y+side*12.6,z),r-2.3,1,(0,side,0)),yellow if z==22 else dark,group)
        bolt(prefix+'_axle_bolt',(x,y+side*13.7,z),(0,side,0),group,1.7)
    for x,z in [(-5,95),(18,76)]:
        bolt(prefix+'_armor_screw',(x,yy+side*3,z),(0,side,0),group)
    for z in [41,47,53]:
        add(prefix+'_link_hole',cyl((8+(z-47)*.46,y+side*9.1,z),1.55,.6,(0,side,0)),black,group)
    cable(prefix+'_servo_cable',[(9,y-side*9,99),(24,y-side*10,88),(28,y-side*10,69),(16,y-side*10,56),(1,y-side*10,33)],.85,group)
    fg = prefix+'_Foot'
    add(prefix+'_rubber_sole',rounded(-31,y-17,0,52,34,6,2.8),purple,fg)
    add(prefix+'_sole_gasket',rounded(-31,y-17,5.5,52,34,1.7,.65),black,fg)
    add(prefix+'_yellow_foot_deck',rounded(-31,y-17,7,52,34,5,2.2),yellow,fg)
    for off in [-12,9]:
        plate = Part.Face(Part.makePolygon([V(-17,y+off,11),V(7,y+off,11),V(-5,y+off,29),V(-17,y+off,11)])).extrude(V(0,3,0))
        add(prefix+'_ankle_bracket',plate,yellow,fg)
    for offset in [-10,0,10]:
        add(prefix+'_toe_grip',rounded(-31.1,y+offset-1.2,2,1.3,2.4,2,.5),dark,fg)
    for x in [-24,14]:
        for off in [-11,11]:
            bolt(prefix+'_deck_fastener',(x,y+off,12),(0,0,1),fg,1.1)

doc.addObject('App::DocumentObjectGroup','Design_Notes')
notes = doc.getObject('Design_Notes')
notes.addProperty('App::PropertyString','Status')
notes.Status = 'Appearance concept; not a validated mechanical assembly. Dimensions in mm.'
notes.addProperty('App::PropertyString','Reference')
notes.Reference = 'Microduck photo references: graphite shell, yellow beak, exposed servos, violet soles.'
doc.recompute()
for obj in doc.Objects:
    if obj.isDerivedFrom('Part::Feature'):
        obj.ViewObject.DisplayMode = 'Shaded'
Gui.activeDocument().activeView().setCameraOrientation(
    App.Rotation(V(1,-1,0), V(0,0,1), V(-1,-1,0.5), 'ZXY').Q)
Gui.activeDocument().activeView().fitAll()
doc.recompute()
doc.saveAs(os.path.join(ROOT,'MicroDuck_Futuristic.FCStd'))
print('Created',len([o for o in doc.Objects if o.isDerivedFrom('Part::Feature')]),'CAD parts')
