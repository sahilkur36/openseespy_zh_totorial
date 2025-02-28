import opstool as opst
from opstool.pre.section.sec_mesh import FiberSecMesh as Section
import openseespy.opensees as ops
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal

from shapely import Geometry

from section import box_section_geoms, extract_boundaries
from material import add_materials, concrete_tag, bar_tag
from unitsystem import UNIT

def create_pile_section(Diameter: float = 2.0, 
                         coverThick: float = 0.08, 
                         target_rho: float = 0.02, 
                         initial_d_bar: float = 32, 
                         concrete_tag: concrete_tag = "C40", 
                         bar_tag: bar_tag = "HRB400",
                         fixed_gap: float = None):
    """
    创建桩基截面并自动调整配筋率到目标值
    
    参数:
    -----
    Diameter: 桩基直径(m)
    coverThick: 保护层厚度(m)
    target_rho: 目标配筋率(默认0.02)
    initial_d_bar: 初始钢筋直径(mm)
    concrete_tag: 混凝土标号
    bar_tag: 钢筋类型
    fixed_gap: 固定钢筋间距(m)，若为None则使用自动计算的间距
    
    返回:
    -----
    SEC: 截面对象
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # 材料标签定义
    matTagC = 1      # 保护层混凝土
    matTagCCore = 2  # 核心混凝土
    matTagS = 3      # 钢筋
    matTagSteel = 4  # 带约束的钢筋

    # 创建几何形状
    core_patch = opst.pre.section.create_circle_patch(
        [0, 0], 
        (Diameter / 2.0 - coverThick) * UNIT.m, 
        holes=None, 
        angle1=0.0, 
        angle2=360, 
        n_sub=40, 
        material=None
    )
    coverlines = list(core_patch.geom.exterior.coords)
    cover_patch = opst.pre.section.create_circle_patch(
        [0, 0], 
        Diameter / 2.0 * UNIT.m, 
        holes=[coverlines], 
        angle1=0.0, 
        angle2=360, 
        n_sub=40, 
        material=None
    )

    # 创建截面对象
    SEC = Section()
    SEC.add_patch_group(dict(cover=cover_patch, core=core_patch))
    SEC.set_mesh_size(dict(cover=coverThick * UNIT.m, core=0.2 * UNIT.m))
    SEC.set_mesh_color(dict(cover="gray", core="green"))
    SEC.set_ops_mat_tag(dict(cover=matTagC, core=matTagCCore))
    SEC.mesh()
    
    # 计算有效截面积
    core_area = core_patch.geom.area
    
    # 确定钢筋间距
    rebar_lines = opst.pre.section.offset(coverlines, d=initial_d_bar * UNIT.mm / 2)
    circumference = 2 * np.pi * (Diameter / 2.0 - coverThick - initial_d_bar / 2000) * UNIT.m
    
    # 如果提供了固定间距，使用固定间距
    if fixed_gap is not None:
        gap = fixed_gap
        print(f"使用固定钢筋间距: {gap:.4f} m")
    else:
        # 自动计算间距 - 基于常见配筋率估计值
        bar_area = np.pi * (initial_d_bar / 2) ** 2 * UNIT.mm ** 2
        target_total_bar_area = core_area * target_rho
        estimated_num_bars = target_total_bar_area / bar_area
        gap = circumference / estimated_num_bars
        print(f"自动计算的初始钢筋间距: {gap:.4f} m")
    
    # 迭代调整钢筋直径以达到目标配筋率
    current_d_bar = initial_d_bar
    max_iterations = 10  # 最大迭代次数
    tolerance = 0.0005   # 配筋率容差
    
    # 添加初始钢筋
    SEC.add_rebar_line(
        points=rebar_lines,
        dia=current_d_bar * UNIT.mm,
        gap=gap,
        color="red",
        ops_mat_tag=matTagSteel,
    )
    
    # 计算初始配筋率
    sec_props = SEC.get_frame_props(display_results=False)
    current_rho = sec_props['rho_rebar']
    print(f"初始配置: 钢筋直径 = {current_d_bar:.1f} mm, 配筋率 = {current_rho:.4f}, 目标配筋率 = {target_rho:.4f}")
    
    # 如果初始配筋率已经足够接近目标值，则不需要迭代
    if abs(current_rho - target_rho) <= tolerance:
        print(f"初始配筋率已达到目标: {current_rho:.4f}")
    else:
        for iteration in range(max_iterations):
            # 计算需要调整的钢筋直径
            adjustment_factor = math.sqrt(target_rho / current_rho)
            # 限制调整幅度，避免过大变化
            new_d_bar = current_d_bar * min(max(adjustment_factor, 0.9), 1.1)
            
            # 确保钢筋直径是实际可用规格（通常按2mm递增）
            new_d_bar = round(new_d_bar / 2) * 2
            
            # 如果钢筋直径没变，但还未达到目标，则微调
            if new_d_bar == current_d_bar and abs(current_rho - target_rho) > tolerance:
                if current_rho < target_rho:
                    new_d_bar += 2  # 增加2mm
                else:
                    new_d_bar -= 2  # 减少2mm
            
            # 确保钢筋直径在合理范围内
            new_d_bar = max(min(new_d_bar, 50), 12)  # 一般12mm-50mm是常用范围
            
            # 如果钢筋直径没变，跳过本次迭代
            if new_d_bar == current_d_bar:
                continue
                
            # 更新钢筋
            current_d_bar = new_d_bar
            
            # 清除所有现有钢筋
            SEC = Section()  # 重新创建截面对象
            SEC.add_patch_group(dict(cover=cover_patch, core=core_patch))
            SEC.set_mesh_size(dict(cover=0.1 * UNIT.m, core=0.2 * UNIT.m))
            SEC.set_mesh_color(dict(cover="gray", core="green"))
            SEC.set_ops_mat_tag(dict(cover=matTagC, core=matTagCCore))
            SEC.mesh()
            
            # 添加新钢筋
            SEC.add_rebar_line(
                points=rebar_lines,
                dia=current_d_bar * UNIT.mm,
                gap=gap,
                color="red",
                ops_mat_tag=matTagSteel,
            )
            
            # 计算当前配筋率
            sec_props = SEC.get_frame_props(display_results=False)
            current_rho = sec_props['rho_rebar']
            
            print(f"迭代 {iteration+1}: 钢筋直径 = {current_d_bar:.1f} mm, 配筋率 = {current_rho:.4f}, 目标 = {target_rho:.4f}")
            
            # 检查是否达到目标
            if abs(current_rho - target_rho) <= tolerance:
                print(f"已达到目标配筋率: {current_rho:.4f}")
                break
    
    # 最终计算属性并显示结果
    sec_props = SEC.get_frame_props(display_results=True)
    SEC.centring()

    # 添加材料
    sec_params = {'roucc': sec_props['rho_rebar'], 'd': Diameter}
    global cover,core,bar
    cover,core,bar = add_materials(concrete_tag, bar_tag, sec_params, matTagC, matTagCCore, matTagS, matTagSteel)

    print(f"最终配筋率: {sec_props['rho_rebar']:.4f}")
    print(f"钢筋数量: {SEC.get_rebars_num()}")
    print(f"钢筋直径: {current_d_bar:.1f} mm")
    print(f"钢筋间距: {gap:.4f} m")
    
    return SEC

def create_pier_section(width:float = 900,height:float = 300,
                        coverThick: float = 0.08,
                        d_bar: float = 32, 
                        concrete_tag: concrete_tag = "C40", 
                        bar_tag: bar_tag = "HRB400",
                        gap: float = 0.15):
    
    section_geometry = box_section_geoms(width,height)

    # 获取边界
    exterior, interiors = extract_boundaries(section_geometry, unit='cm')

    outer_cover_lines = opst.pre.section.offset(exterior, d=coverThick * UNIT.m)
    outer_bar_lines = opst.pre.section.offset(outer_cover_lines, d=d_bar/2 * UNIT.mm)
    outer_bar_lines2 = opst.pre.section.offset(outer_bar_lines, d=d_bar * UNIT.mm)

    inner_cover_lines = []
    inner_bar_lines = []
    for interior in interiors:
        inner_cover_lines.append(opst.pre.section.offset(interior, d=-coverThick * UNIT.m))
        inner_bar_lines.append(opst.pre.section.offset(interior, d=-coverThick * UNIT.m-d_bar/2 * UNIT.mm))

    outer_cover = opst.pre.section.create_polygon_patch(exterior, holes=[outer_cover_lines])
    core_patch = opst.pre.section.create_polygon_patch(outer_cover_lines, holes=inner_cover_lines)
    inner_cover = opst.pre.section.create_polygon_patch(inner_cover_lines[0], holes=interiors)

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # 材料标签定义
    matTagC = 1      # 保护层混凝土
    matTagCCore = 2  # 核心混凝土
    matTagS = 3      # 钢筋
    matTagSteel = 4  # 带约束的钢筋

    # 创建截面对象
    SEC = Section()
    SEC.add_patch_group(dict(outer_cover=outer_cover, inner_cover=inner_cover, core=core_patch))
    SEC.set_mesh_size(dict(outer_cover=coverThick * UNIT.m, inner_cover=coverThick * UNIT.m, core=0.5 * UNIT.m))
    SEC.set_mesh_color(dict(outer_cover="gray", inner_cover="gray", core="green"))
    SEC.set_ops_mat_tag(dict(outer_cover=matTagC, inner_cover=matTagC, core=matTagCCore))
    SEC.mesh()

    # 添加初始钢筋(外侧两遍)
    SEC.add_rebar_line(
        points=outer_bar_lines,
        dia=d_bar * UNIT.mm,
        gap=gap,
        color="red",
        ops_mat_tag=matTagSteel,
    )
    SEC.add_rebar_line(
        points=outer_bar_lines2,
        dia=d_bar * UNIT.mm,
        gap=gap,
        color="red",
        ops_mat_tag=matTagSteel,
    )
    # 添加初始钢筋(内侧一遍)
    for barline in inner_bar_lines:
        SEC.add_rebar_line(
            points=barline,
            dia=d_bar * UNIT.mm,
            gap=gap,
            color="red",
            ops_mat_tag=matTagSteel,
        )
    # 最终计算属性并显示结果
    sec_props = SEC.get_frame_props(display_results=True)
    SEC.centring()

    # 添加材料
    sec_params = {'roucc': sec_props['rho_rebar'], 'd': d_bar}
    global cover, core, bar
    cover, core, bar = add_materials(concrete_tag, bar_tag, sec_params, matTagC, matTagCCore, matTagS, matTagSteel)

    print(f"配筋率: {sec_props['rho_rebar']:.4f}")
    print(f"钢筋数量: {SEC.get_rebars_num()}")
    print(f"钢筋直径: {d_bar:.1f} mm")
    print(f"钢筋间距: {gap:.4f} m")

    return SEC

def plot_stress_strain(MC:opst.anlys.MomentCurvature = None, Mode:Literal["stress", "strain"] = "stress"):
    # 获取纤维应力应变数据
    fiber_data = MC.get_fiber_data()
    fiber_data_last = fiber_data.isel(Steps=-1)
    y = fiber_data_last.sel(Properties="yloc")
    z = fiber_data_last.sel(Properties="zloc")
    matTag = fiber_data_last.sel(Properties="mat")
    stress = fiber_data_last.sel(Properties="stress")
    strain = fiber_data_last.sel(Properties="strain")

    plt.figure()
    if Mode == "stress":
        # 绘制应力分布图
        s = plt.scatter(y, z, c=stress, s=50, cmap="rainbow")
        plt.title("应力分布")
    elif Mode == "strain":
        # 绘制应变分布图
        s = plt.scatter(y, z, c=strain, s=50, cmap="rainbow")
        plt.title("应变分布")
    
    plt.colorbar(s)
    plt.xlabel("y轴 (m)")
    plt.ylabel("z轴 (m)")
    plt.show()

def perform_mc_analysis(SEC:Section, axial_force:float, dir = Literal["y","z"]):
    """
    接受SEC对象和轴压力，进行MC分析并返回MC对象，不绘制图片
    
    参数:
    -----
    SEC: 截面对象
    axial_force: 轴力(kN)，正为拉，负为压
    
    返回:
    -----
    MC: 弯矩曲率分析对象
    """
    # 分析截面
    sec_tag = 1
    SEC.to_opspy_cmds(secTag=sec_tag, GJ=100000)
    
    # 创建弯矩曲率分析对象
    MC = opst.anlys.MomentCurvature(sec_tag=sec_tag, axial_force=axial_force * UNIT.kN)
    MC.analyze(axis=dir, incr_phi=1e-5, limit_peak_ratio=0.8, smart_analyze=True)
    
    return MC

if __name__ == "__main__":
    # 打印单位信息
    # UNIT.print()

    # Case1：创建桩基（圆形）截面，指定直径、保护层厚度和目标配筋率
    SEC = create_pile_section(
        Diameter=2.0, 
        coverThick=0.08, 
        target_rho=0.01, 
        initial_d_bar=32, 
        fixed_gap=0.15  # 固定钢筋间距为15cm
    )

    # Case2：创建墩柱截面（圆端形）截面，制定宽度和高度等参数
    # SEC = create_pier_section(width=900,height=300,)
    
    # 显示截面
    SEC.view(fill=True)
    plt.axis("equal")
    plt.show()
    
    # 进行弯矩曲率分析
    MC = perform_mc_analysis(SEC,axial_force=-5000 * UNIT.kN, dir="y")  # 轴力(kN)正为拉，负为压
    # 绘制完整的弯矩-曲率关系图
    # MC.plot_M_phi()
    # plt.show()

    # 获取弯矩曲率数据
    # phi, M = MC.get_M_phi()

    # 绘制各纤维应力应变分布
    plot_stress_strain(MC, Mode="stress")
    plot_stress_strain(MC, Mode="strain")

    # 计算钢筋屈服极限状态
    phiy, My = MC.get_limit_state(
        matTag=4,  # 钢筋材料标签
        threshold=2e-3,  # 钢筋屈服应变
    )

    # 计算混凝土压溃极限状态
    global core
    phiu_c, Mu_c = MC.get_limit_state(
        matTag=2,  # 核心混凝土材料标签
        threshold=core.ecu,  # 极限压应变
    )
    # 或通过峰值强度下降率来计算
    drop_ratio = 0.2
    phiu_p, Mu_p = MC.get_limit_state(peak_drop=drop_ratio)

    print(f"极限状态1 (钢筋屈服): phi_y={phiy:.4f}, My={My:.2f}")
    if phiu_c <= phiu_p:
        casestr = "核心混凝土压溃"
        phiu = phiu_c
        Mu = Mu_c
    else:
        casestr = f"抗弯强度降至{1-drop_ratio:.2f}倍Mu"
        phiu = phiu_p
        Mu = Mu_p
    print(f"极限状态2 ({casestr}): phi_u={phiu:.4f}, Mu={Mu:.2f}")

    # 双线性简化
    phi_eq, M_eq = MC.bilinearize(phiy, My, phiu, plot=True)
    plt.show()
    
    # 计算曲率延性系数
    ductility = phiu / phiy
    print(f"曲率延性系数: {ductility:.2f}")