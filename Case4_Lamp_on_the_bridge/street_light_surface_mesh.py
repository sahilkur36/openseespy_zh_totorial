import gmsh
import numpy as np
import sys
import math
from scipy.interpolate import BSpline,make_interp_spline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional
from pathlib import Path
from typing import Callable, Union

def plot_spline(
    spline_x:BSpline, 
    spline_y:BSpline, 
    spline_z:BSpline, 
    control_points: List[List[float]],
    num_points: int = 100,
    title: Optional[str] = "样条曲线"
) -> None:
    """
    绘制3D样条曲线及其控制点
    
    参数:
        spline_x: x方向的三次样条
        spline_y: y方向的三次样条
        spline_z: z方向的三次样条
        control_points: 控制点列表
        num_points: 用于绘制样条的点数
        title: 图表标题
    """
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 生成用于绘制样条的t值
    t = np.linspace(0, 1, num_points)
    
    # 计算样条曲线上的点
    x_curve = [float(spline_x(t_val)) for t_val in t]
    y_curve = [float(spline_y(t_val)) for t_val in t]
    z_curve = [float(spline_z(t_val)) for t_val in t]
    
    # 绘制样条曲线
    ax.plot(x_curve, y_curve, z_curve, 'b-', linewidth=2, label='样条曲线')
    
    # 提取控制点的坐标
    x_points = [p[0] for p in control_points]
    y_points = [p[1] for p in control_points]
    z_points = [p[2] for p in control_points]
    
    # 绘制控制点
    ax.scatter(x_points, y_points, z_points, color='red', s=50, label='控制点')
    
    # 添加标签和图例
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title(title)
    ax.legend()
    
    # 设置等比例坐标轴
    max_range = max([
        max(x_curve) - min(x_curve),
        max(y_curve) - min(y_curve),
        max(z_curve) - min(z_curve)
    ])
    
    mid_x = (max(x_curve) + min(x_curve)) / 2
    mid_y = (max(y_curve) + min(y_curve)) / 2
    mid_z = (max(z_curve) + min(z_curve)) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()

def create_smooth_spline(t_values, points, k=3, smoothing=None):
    """创建平滑的B样条曲线"""
    # 确保t值严格递增
    t_unique, indices = np.unique(t_values, return_index=True)
    p_unique = [points[i] for i in indices]
    
    if len(t_unique) < k+1:
        # 如果点太少，降低样条阶数
        k = len(t_unique) - 1
    
    # 创建B样条
    spl = make_interp_spline(t_unique, p_unique, k=k, bc_type='natural')
    return spl

def create_bent_pipe(
    control_points: List[List[float]],
    radius_function: Callable[[float], float],
    num_sections: int=40,
    num_circle_points: int=16,
    mesh_size: float=0.1,
    filename:Union[str,Path]=Path(__file__).parent / "bent_pipe.msh",
    mesh_end_surface:bool=False,
    plot_spline:bool=False
):
    # 初始化Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("BentPipe")
    
    # 过滤掉重合的控制点
    filtered_points = [control_points[0]]
    for point in control_points[1:]:
        if np.linalg.norm(np.array(point) - np.array(filtered_points[-1])) > 1e-6:
            filtered_points.append(point)
    
    control_points = filtered_points
    
    # 使用简单的索引参数化（避免距离计算）
    t_values = np.linspace(0, 1, len(control_points))
    
    # 提取坐标
    x_points = [p[0] for p in control_points]
    y_points = [p[1] for p in control_points]
    z_points = [p[2] for p in control_points]
    
    # 创建样条
    spline_x = create_smooth_spline(t_values, x_points, k=3)
    spline_y = create_smooth_spline(t_values, y_points, k=3) 
    spline_z = create_smooth_spline(t_values, z_points, k=3)
    
    if plot_spline:
        plot_spline(spline_x, spline_y, spline_z, control_points)
    
    # 计算样条上的点和切向量
    section_points = []
    circle_points = []
    all_lines = []
    
    for i in range(num_sections + 1):
        t = i / num_sections
        
        # 计算样条上的点和切向量
        point = [float(spline_x(t)), float(spline_y(t)), float(spline_z(t))]
        
        # 计算切向量
        tx = float(spline_x.derivative()(t))
        ty = float(spline_y.derivative()(t))
        tz = float(spline_z.derivative()(t))
        
        magnitude = math.sqrt(tx*tx + ty*ty + tz*tz)
        tangent = [tx/magnitude, ty/magnitude, tz/magnitude]
        
        # 计算法向量和副法向量
        # 选择辅助向量
        aux_vector = [0, 1, 0]
        dot_product = sum(aux_vector[j] * tangent[j] for j in range(3))
        
        if abs(dot_product) > 0.9:
            aux_vector = [0, 0, 1]
            dot_product = sum(aux_vector[j] * tangent[j] for j in range(3))
        
        normal = [aux_vector[j] - dot_product * tangent[j] for j in range(3)]
        normal_norm = math.sqrt(sum(x*x for x in normal))
        normal = [x/normal_norm for x in normal]
        
        binormal = [
            tangent[1]*normal[2] - tangent[2]*normal[1],
            tangent[2]*normal[0] - tangent[0]*normal[2],
            tangent[0]*normal[1] - tangent[1]*normal[0]
        ]
        
        # 计算当前半径
        current_radius = radius_function(t)
        
        # 创建截面上的点
        section_circle_points = []
        for j in range(num_circle_points):
            angle = j * 2 * math.pi / num_circle_points
            offset_x = current_radius * (normal[0] * math.cos(angle) + binormal[0] * math.sin(angle))
            offset_y = current_radius * (normal[1] * math.cos(angle) + binormal[1] * math.sin(angle))
            offset_z = current_radius * (normal[2] * math.cos(angle) + binormal[2] * math.sin(angle))
            
            x = point[0] + offset_x
            y = point[1] + offset_y
            z = point[2] + offset_z
            
            pt = gmsh.model.occ.addPoint(x, y, z, mesh_size)
            section_circle_points.append(pt)
        
        circle_points.append(section_circle_points)
    
    # 创建每个截面的圆周线
    bottom_circle_lines = []
    top_circle_lines = []
    for i in range(num_sections + 1):
        section_lines = []
        for j in range(num_circle_points):
            next_j = (j + 1) % num_circle_points
            line = gmsh.model.occ.addLine(circle_points[i][j], circle_points[i][next_j])
            section_lines.append(line)
        all_lines.append(section_lines)

        if i == 0:
            bottom_circle_lines.append(section_lines)
        elif i == num_sections:
            top_circle_lines.append(section_lines)
    
    # 连接相邻截面形成管壁
    connecting_lines = []
    for i in range(num_sections):
        section_connecting_lines = []
        for j in range(num_circle_points):
            line = gmsh.model.occ.addLine(circle_points[i][j], circle_points[i+1][j])
            section_connecting_lines.append(line)
        connecting_lines.append(section_connecting_lines)
    
    # 创建四边形面
    all_surfaces = []
    
    # 创建管壁四边形面
    for i in range(num_sections):
        for j in range(num_circle_points):
            next_j = (j + 1) % num_circle_points
            
            # 四边形的四条边
            loop_lines = [
                all_lines[i][j],
                connecting_lines[i][next_j],
                -all_lines[i+1][j],  # 注意符号，表示方向
                -connecting_lines[i][j]
            ]
            
            curve_loop = gmsh.model.occ.addCurveLoop(loop_lines)
            
            # 使用addSurfaceFilling代替addPlaneSurface
            # 这个函数可以创建非平面的表面
            surface = gmsh.model.occ.addSurfaceFilling(curve_loop)
            all_surfaces.append(surface)
    
    # 创建端面（可选）
    # 第一个端面
    first_loop = gmsh.model.occ.addCurveLoop(all_lines[0])
    first_surface = gmsh.model.occ.addSurfaceFilling(first_loop)
    
    # 最后一个端面
    last_loop = gmsh.model.occ.addCurveLoop(all_lines[-1])
    last_surface = gmsh.model.occ.addSurfaceFilling(last_loop)
    
    # 同步几何模型
    gmsh.model.occ.synchronize()
    
    if mesh_end_surface:
        all_surfaces.append(first_surface)
        all_surfaces.append(last_surface)
    else:
        # 我们将通过设置端面的可见性来控制它们是否被网格化
        gmsh.model.setVisibility([(2, first_surface)], 0)  # 隐藏第一个端面
        gmsh.model.setVisibility([(2, last_surface)], 0)   # 隐藏最后一个端面
    
    # 创建管壁表面的物理组
    gmsh.model.addPhysicalGroup(2, all_surfaces, tag=1, name="tube_surface")

    # 创建底层管壁和上层管壁的物理组
    gmsh.model.addPhysicalGroup(1, bottom_circle_lines[0], tag=2, name="bottom_boundary")
    gmsh.model.addPhysicalGroup(1, top_circle_lines[0], tag=3, name="top_boundary")
    
    # 设置结构化网格
    for surface in all_surfaces:
        gmsh.model.mesh.setRecombine(2, surface)
    
    # 只网格化可见的实体
    gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
    
    # 生成网格
    gmsh.model.mesh.generate(2)
    
    # 保存网格
    gmsh.write(str(filename))
    
    # 显示GUI
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    
    # 清理
    gmsh.finalize()

def create_street_light():
    """创建路灯示例"""
    # 参数定义
    height = 9.0
    horizontal_length = 1.0
    horizontal_slope = 0.0874886    # 5°
    arc_radius = 0.4
    base_radius = 0.09
    top_radius = 0.03
    
    max_angle = math.pi/2 - math.atan(horizontal_slope)
    arc_y = arc_radius * math.sin(max_angle)
    arc_x = arc_radius * (1 - math.cos(max_angle))
    column_height = height - arc_y - horizontal_slope * (horizontal_length - arc_x)
    
    # 定义轴线控制点
    control_points = [
        [0,0,0],
        [0,0,column_height/3],
        [0,0,column_height/3*2],
        [0,0,column_height*0.9],
        [0,0,column_height],
    ]
    
    # 添加弧段上的点
    arc_steps = 4  # 增加点数使曲线更平滑
    for i in range(arc_steps):
        angle = i * max_angle / (arc_steps)
        # 修正圆弧方向，使其从下往上弯曲
        x = arc_radius * (1 - math.cos(angle))  # x值从0开始增加
        z = column_height + arc_radius * math.sin(angle)  # y值从低点开始增加
        control_points.append([x, 0, z])
    
    arc_end = arc_x
    arc_height = column_height+arc_y
    # 添加水平段
    control_points.append([arc_end, 0, arc_height])
    h_steps = 3
    for i in range(1, h_steps + 1):
        x = arc_end + (horizontal_length - arc_end) * i / h_steps
        control_points.append([x, 0, arc_height + horizontal_slope * (x - arc_end)])

    # 定义半径变化函数
    def radius_function(t):
        return base_radius * (1 - t) + top_radius * t
    
    # # 绘制轴线
    # x = [point[0] for point in control_points]
    # z = [point[2] for point in control_points]
    # for i in range(len(x)):
    #     print(f"{x[i]}, {z[i]}")
    # plt.plot(x, z, 'ro-')
    # plt.axis('equal')
    # plt.show()
    

    # 创建路灯
    create_bent_pipe(
        control_points=control_points,
        radius_function=radius_function,
        num_sections=20,
        num_circle_points=16,
        mesh_size=0.4,
        filename=Path(__file__).parent / "street_light.msh",
        mesh_end_surface=False,
        plot_spline=False
    )

if __name__ == "__main__":
    create_street_light()