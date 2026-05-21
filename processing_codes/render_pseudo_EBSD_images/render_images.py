from ovito.io import import_file
from ovito.data import *
from ovito.modifiers import *
from ovito.vis import Viewport

from orix import plot
from orix.quaternion.symmetry import Oh
from orix.vector import Vector3d

import numpy as np

# The dump output file name
dump_filename = "dump_hybrid_rough.xyz"

def global_z_in_crystal_frame(q):
    """sample z axis in crystal coordinate system"""
    x, y, z, w = q
    vx = 2 * (x*z - y*w)
    vy = 2 * (y*z + x*w)
    vz = 1 - 2 * (x*x + y*y)
    return np.array([vx, vy, vz])
    
def direction2rgb(v):
    """convert vector v (Vector3d in orix) into RGB value"""
    color_key = plot.DirectionColorKeyTSL(symmetry = Oh)
    temp = color_key.direction2color(v)
    return temp[0]
    
def coloring_modifier(frame: int, data: DataCollection):
    data.tables['grains']
    grain_count = data.attributes['GrainSegmentation.grain_count']
    particle_count = data.particles.count
    
    RGB = np.zeros((grain_count, 3))
    for i in range(grain_count):
        quat = data.tables['grains']['Orientation'][i]
        z_in_crystal = global_z_in_crystal_frame(quat)
        v = Vector3d(z_in_crystal)
        RGB[i] = direction2rgb(v)
        
    new_color = np.full((particle_count, 3), 0.95)
    for i in range(particle_count):
        grain_no = data.particles['Grain'][i] - 1 #note that grain identifier starts from 1
        if data.tables['grains']['Structure Type'][grain_no]!=PolyhedralTemplateMatchingModifier.Type.HCP:
            new_color[i] = RGB[grain_no]
    
    # Output:
    data.particles_.create_property('Color', data=new_color)
    
pipeline = import_file(dump_filename)
pipeline.add_to_scene()

pipeline.modifiers.append(PolyhedralTemplateMatchingModifier(output_orientation=True))
pipeline.modifiers.append(GrainSegmentationModifier(handle_stacking_faults=False))
pipeline.modifiers.append(coloring_modifier)

vp = Viewport()
vp.type = Viewport.Type.Top
vp.camera_pos = (100, 150, 250)
vp.camera_dir = (0,0,-1)
vp.zoom_all(size=(600,800))

for i in range(pipeline.num_frames):
    name = 'step' + str(i) + '.png'
    vp.render_image(filename=name,
                size=(600,800),
                alpha=True,
                frame=i)
    print_content = 'image ' + str(i) + ' rendered.\n'
    print(print_content)

