import matplotlib.pyplot as plt
import opstool as opst
import opstool.vis.pyvista as opsvis
from opstool.utils.consts import CONSTANTS

CONSTANTS.set_output_dir("Case4_Lamp_on_the_bridge/_OPSTOOL_ODB")
opst.post.loadODB("2")
fig = opst.vis.pyvista.plot_eigen(mode_tags=9, odb_tag="eigen", subplots=True)
fig.show()

# opsvis.set_plot_props(point_size=0.2,
#                       line_width=0.0,
#                       show_mesh_edges=False,
#                       cmap="turbo",
#                       title_font_size=12,)

# opsvis.plot_nodal_responses_animation(
#     odb_tag="2",
#     framerate=30,
#     scale=2,
#     savefig="NodalRespAnimation.gif",
#     resp_type="disp",
#     resp_dof=["UX", "UY", "UZ"],
# ).show()

# opsvis.plot_unstruct_responses_animation(
#     odb_tag="2",
#     ele_type="Shell",
#     framerate=20,
#     savefig="UnstructRespAnimation.gif",
#     resp_type="sectionForces",
#     resp_dof="MXY",
# ).close()