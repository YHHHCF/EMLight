import bpy
import sys
import os

# Function to clean up memory
def clean_memory():
    # Clear all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Remove all meshes, materials, and other data blocks
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.images:
        bpy.data.images.remove(block)
    for block in bpy.data.textures:
        bpy.data.textures.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)

def render(exr_path, output_image_path, mode):
    clean_memory()

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
    camera.location = (0, -12, 0)
    camera.rotation_euler = (1.5708, 0, 0)

    # Set the rendering settings
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = output_image_path
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Set lower resolution for rendering
    bpy.context.scene.render.resolution_x = 256  # Set width of the output image
    bpy.context.scene.render.resolution_y = 192  # Set height of the output image
    bpy.context.scene.render.resolution_percentage = 100  # Percentage of the resolution

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
    # base_dir = "./results/semantic_model/results/"
    base_dir = "./qualitative_eval/"
    save_dir = base_dir + "rendered/"
    nms = os.listdir(base_dir)

    i = 0
    for nm in nms:
        if nm.endswith('.exr'):
            hdr_path = base_dir + nm
            mirror_path = save_dir + nm.replace('.exr', '_mirror.png')
            matte_path = save_dir + nm.replace('.exr', '_matte.png')
            diffuse_path = save_dir + nm.replace('.exr', '_diffuse.png')

            render(hdr_path, mirror_path, mode='mirror')
            render(hdr_path, matte_path, mode='matte')
            render(hdr_path, diffuse_path, mode='diffuse')
            i += 1
            print(i)
