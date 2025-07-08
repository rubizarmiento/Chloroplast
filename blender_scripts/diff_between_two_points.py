import bpy
import bmesh

obj = bpy.context.active_object

if obj is None or obj.type != 'MESH' or bpy.context.mode != 'EDIT_MESH':
    print("❌ Please enter Edit Mode and select a mesh object.")
else:
    bm = bmesh.from_edit_mesh(obj.data)
    selected_verts = [v.co for v in bm.verts if v.select]

    if len(selected_verts) != 2:
        print("❌ Please select exactly two vertices.")
    else:
        v1, v2 = selected_verts
        dx = v2.x - v1.x
        dy = v2.y - v1.y
        dz = v2.z - v1.z

        print(f" First vertex: {v1.x:.4f} {v1.y:.4f} {v1.z:.4f}")
        print(f" Second vertex: {v2.x:.4f} {v2.y:.4f} {v2.z:.4f}")

        print("Difference between the two vertices:")
        print(f"{dx:.4f} {dy:.4f} {dz:.4f}")
