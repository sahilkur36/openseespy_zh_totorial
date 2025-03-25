import openseespy.opensees as ops
import opstool as opst
from pathlib import Path

ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)
# 材料参数
E, nu, rho = 2.06e11, 0.3, 7850  # Pa, kg/m3
ops.nDMaterial("ElasticIsotropic", 1, E, nu, rho)
# 薄板纤维截面
secTag = 11
ops.section("PlateFiber", secTag, 1, 0.004) # 4mm

# 准备读取GMSH文件
GMSH2OPS = opst.pre.Gmsh2OPS(ndm=3, ndf=6)

mesh_file = Path(__file__).parent / "street_light.msh"
GMSH2OPS.read_gmsh_file(mesh_file.absolute())

# 根据GMSH文件创建OpenSeesPy节点命令
node_tags = GMSH2OPS.create_node_cmds()

dim_entity_tags = GMSH2OPS.get_dim_entity_tags()
dim_entity_tags_2D = [item for item in dim_entity_tags if item[0] == 2]

# 根据GMSH文件创建OpenSeesPy元素命令
ele_tags_n4 = GMSH2OPS.create_element_cmds(
    ops_ele_type="ASDShellQ4",  # OpenSeesPy 单元类型
    ops_ele_args=[
        secTag
    ],  # 单元额外参数(e.g., section tag)
    dim_entity_tags=dim_entity_tags_2D,
)

removed_node_tags = opst.pre.remove_void_nodes()
print(removed_node_tags)

# 获取边界实体标签以创建约束
bottom_boundary_dim_tags = GMSH2OPS.get_boundary_dim_tags(
    physical_group_names="bottom_boundary", include_self=True)
print(bottom_boundary_dim_tags)

top_boundary_dim_tags = GMSH2OPS.get_boundary_dim_tags(
    physical_group_names="top_boundary", include_self=True)
print(top_boundary_dim_tags)

# 约束底面
fix_ntags = GMSH2OPS.create_fix_cmds(dim_entity_tags=bottom_boundary_dim_tags, dofs=[1] * 6)

# 约束顶面（连接一个质量点）
light_tag = 1000000
ops.node(light_tag, 1.2, 0, 9)
ops.mass(light_tag, 30,30,30,0,0,0)  # 质量30kg
# 在节点和灯的单元之间创建刚臂
for dim, tag in top_boundary_dim_tags:
    if dim == 0:
        ops.rigidLink('beam', light_tag, tag)


opst.vis.pyvista.set_plot_props(notebook=False)
opst.vis.pyvista.plot_model(show_outline=True).show()

opst.post.save_eigen_data(odb_tag="eigen", mode_tag=60)

fig = opst.vis.pyvista.plot_eigen(mode_tags=9, odb_tag="eigen", subplots=True)
fig.show()

modal_props, eigen_vectors = opst.post.get_eigen_data(odb_tag="eigen")
modal_props = modal_props.to_pandas()
print(modal_props.head())

print(modal_props.loc[[1, 47, 48, 60], "eigenFrequency"])