import matplotlib.pyplot as plt
import opstool as opst
import opstool.vis.pyvista as opsvis
from opstool.utils.consts import CONSTANTS

CONSTANTS.set_output_dir("Case4_Lamp_on_the_bridge/_OPSTOOL_ODB")
opst.post.loadODB("2")

opsvis.set_plot_props(point_size=0,
                      line_width=5,
                      cmap="turbo",
                      title_font_size=12,)

# opsvis.plot_nodal_responses_animation(
#     odb_tag="2",
#     framerate=20,
#     scale=2,
#     savefig="NodalRespAnimation.gif",
#     resp_type="disp",
#     resp_dof=["UX", "UY", "UZ"],
# ).close()

opsvis.plot_unstruct_responses_animation(
    odb_tag="2",
    ele_type="Shell",
    framerate=20,
    savefig="UnstructRespAnimation.gif",
    resp_type="sectionForces",
    resp_dof="MXY",
).close()