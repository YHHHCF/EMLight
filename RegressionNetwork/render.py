import bpy
import sys

def render(exr_path, output_image_path, mode):
    # Clear all existing objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Set the rendering settings for transparent background
    bpy.context.scene.render.film_transparent = True

    # Set the camera
    if not bpy.data.cameras:
        bpy.ops.object.camera_add(location=(0, -8, 0))
    camera = bpy.context.scene.camera or bpy.data.objects['Camera']
    camera.location = (0, -8, 0)
    camera.rotation_euler = (1.5708, 0, 0)

    # Set the rendering settings
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = output_image_path
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Disable all lights in the scene
    for light in bpy.data.lights:
        bpy.data.lights.remove(light, do_unlink=True)

    # Load the EXR file
    img = bpy.data.images.load(exr_path)

    # Set the EXR as the background image in the World settings
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")

    world = bpy.context.scene.world

    # Use nodes for the world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Add Environment Texture node
    env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')
    env_tex_node.image = img

    # Add Background node
    bg_node = nodes.new(type='ShaderNodeBackground')

    # Add World Output node
    output_node = nodes.new(type='ShaderNodeOutputWorld')

    # Link nodes
    links.new(env_tex_node.outputs['Color'], bg_node.inputs['Color'])
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    # Update the scene
    bpy.context.view_layer.update()

    # Add a mirrored sphere
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=128, radius=1, location=(0, 0, 0))
    sphere = bpy.context.active_object

    # Create a new material with a mirror effect
    mat = bpy.data.materials.new(name="Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Add necessary nodes for mirror material
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    if mode == 'mirror':
        material_node = nodes.new(type='ShaderNodeBsdfGlossy')
        material_node.inputs['Roughness'].default_value = 0.0
    elif mode == 'diffuse':
        material_node = nodes.new(type='ShaderNodeBsdfDiffuse')
        material_node.inputs['Roughness'].default_value = 0.0
    elif mode == 'matte':
        material_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        material_node.inputs['Metallic'].default_value = 1.0
        material_node.inputs['Roughness'].default_value = 0.5

    # Link nodes
    links.new(material_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Assign material to the sphere
    sphere.data.materials.append(mat)

    # Render the scene
    bpy.ops.render.render(write_still=True)

if __name__=="__main__":
    # Get the EXR file path and output image path from command line arguments
    exr_path = "AG8A8645-others-40-1.79482-0.98343_gt.exr"

    render(exr_path, './output_mirror.png', mode='mirror')
    render(exr_path, './output_diffuse.png', mode='diffuse')
    render(exr_path, './output_matte.png', mode='matte')
