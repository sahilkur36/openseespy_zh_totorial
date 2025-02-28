
from dataclasses import dataclass
from typing import Literal, TypeAlias
import openseespy.opensees as ops
import math

from Mander import Mander
from unitsystem import UNIT

# 类型别名定义
concrete_tag: TypeAlias = Literal["C25", "C30", "C35", "C40", "C45", "C50", "C55", "C60", "C65", "C70", "C75", "C80"]
bar_tag: TypeAlias = Literal["HRB335", "HRB400", "HRB500"]

# 混凝土材料参数数据类
@dataclass
class Concrete04Params:
    fc: float      # 混凝土抗压强度
    ec: float      # 对应抗压强度的应变
    ecu: float     # 极限应变
    Ec: float      # 弹性模量
    ft: float      # 抗拉强度
    et: float      # 对应抗拉强度的应变

    def __str__(self):
        return f"fc={self.fc:.5g},Ec={self.Ec:.3g},ec={self.ec:.3g},ecu={self.ecu:.3g},ft={self.ft:.3g},et={self.et:.3g}"

    def define(self, mat_Tag: int):
        """定义OpenSees中的混凝土材料"""
        ops.uniaxialMaterial("Concrete04", mat_Tag, self.fc, self.ec, self.ecu, self.Ec, self.ft, self.et)

# 钢筋材料参数数据类
@dataclass
class BarParams:
    fy: float      # 屈服强度
    Es: float      # 弹性模量
    Esh: float     # 硬化模量
    eps_sh: float  # 硬化开始的应变
    eps_ult: float # 极限应变
    fu: float      # 极限强度

    def __str__(self):
        return f"fy={self.fy:.5g},Es={self.Es:.3g},Esh={self.Esh:.3g},eps_sh={self.eps_sh:.3g},eps_ult={self.eps_ult:.3g},fu={self.fu:.3g}"

    def define(self, mat_Tag: int):
        """定义OpenSees中的钢筋材料"""
        ops.uniaxialMaterial("ReinforcingSteel", mat_Tag, self.fy, self.fu, self.Es, self.Esh, self.eps_sh, self.eps_ult)

def get_fc_Ec_from_tag(concrete_tag: concrete_tag):
    """
    根据混凝土标号获取抗压强度和弹性模量
    
    参数:
    -----
    concrete_tag: 混凝土标号
    
    返回:
    -----
    fc: 混凝土抗压强度
    Ec: 混凝土弹性模量
    """
    def factor1(R):
        """计算混凝土强度转换系数1"""
        if R <= 50:
            return 0.76
        elif R == 80:
            return 0.82
        else:
            return 0.76 + 0.06 * (R - 50) / 30
            
    def factor2(R):
        """计算混凝土强度转换系数2"""
        if R < 40:
            return 1.00
        elif R == 80:
            return 0.87
        else:
            return 1 - 0.13 * (R - 40) / 40

    # 从标号中提取数字部分
    R = eval("".join(list(filter(str.isdigit, concrete_tag))))
    # 计算抗压强度(MPa)
    fc = -R * 0.88 * factor1(R) * factor2(R) * UNIT.MPa
    # 计算弹性模量(MPa)
    Ec = 5000 * math.sqrt(-fc / UNIT.MPa) * UNIT.MPa
    return fc, Ec

def get_concrete_cover(
        concrete_tag: concrete_tag = "C40",
        print_params: bool = False,
        tension_factor: float = 0.0,
    ):
    """
    获取保护层混凝土参数
    
    参数:
    -----
    concrete_tag: 混凝土标号
    print_params: 是否打印参数
    tension_factor: 抗拉强度相较于抗压强度的比例，默认不考虑抗拉，为0.0
    
    返回:
    -----
    params: 混凝土参数对象
    """
    # 计算保护层混凝土参数
    fc, Ec = get_fc_Ec_from_tag(concrete_tag)
    ec = -1.4e-3                    # 极限强度对应应变
    ecu = -4.0e-3                   # 压溃强度对应应变
 
    ft = -fc * tension_factor       # 抗拉强度(取抗压强度的tension_factor倍)
    et = ft / Ec                    # 极限拉应变

    params = Concrete04Params(fc, ec, ecu, Ec, ft, et)
    if print_params:
        print(f"保护层混凝土参数: {params}")
    return params

def get_concrete_core(
        concrete_tag: concrete_tag = "C40", 
        section_type: str = "circular", 
        section_params: dict = None,
        print_params: bool = False,
        tension_factor: float = 0.0
    ):
    """
    使用Mander模型获取核心混凝土参数
    
    参数:
    -----
    concrete_tag: 混凝土标号
    section_type: 截面类型('circular', 'rectangular', 'arbitrary')
    section_params: 截面参数字典
    print_params: 是否打印参数
    
    返回:
    -----
    params: 核心混凝土参数对象
    """
    # 从标号获取强度和弹性模量
    fc, Ec = get_fc_Ec_from_tag(concrete_tag)
    
    # 创建Mander模型实例
    mander = Mander()
    
    # 如果未提供截面参数，初始化空字典
    if section_params is None:
        section_params = {}
    
    # 根据截面类型计算核心混凝土参数
    if section_type.lower() == "circular":
        # 提取圆形截面参数
        d = section_params.get('d', 1.0)                    # 直径 (m)
        coverThick = section_params.get('coverThick', 0.08) # 保护层厚度 (m)
        roucc = section_params.get('roucc', 0.03)           # 纵向钢筋配筋率
        s = section_params.get('s', 0.1)                    # 横向钢筋间距 (m)
        ds = section_params.get('ds', 0.014)                # 横向钢筋直径 (m)
        fyh = section_params.get('fyh', 400)                # 横向钢筋屈服强度 (MPa)
        hoop = section_params.get('hoop', "Spiral")         # 箍筋类型 (Spiral或Circular)
        
        # 使用Mander模型计算核心混凝土参数
        fccore, eccore, ecucore = mander.circular(hoop, d, coverThick, roucc, s, ds, fyh, -fc / UNIT.MPa)
        
    elif section_type.lower() == "rectangular":
        # 提取矩形截面参数
        lx = section_params.get('lx', 4.0)                  # x方向长度 (m)
        ly = section_params.get('ly', 6.5)                  # y方向长度 (m)
        coverThick = section_params.get('coverThick', 0.08) # 保护层厚度 (m)
        roucc = section_params.get('roucc', 0.03)           # 纵向钢筋配筋率
        sl = section_params.get('sl', 0.1)                  # 纵向钢筋间距 (m)
        dsl = section_params.get('dsl', 0.032)              # 纵向钢筋直径 (m)
        roux = section_params.get('roux', 0.00057)          # x方向横向钢筋配筋率
        rouy = section_params.get('rouy', 0.00889)          # y方向横向钢筋配筋率
        st = section_params.get('st', 0.3)                  # 横向钢筋间距 (m)
        dst = section_params.get('dst', 0.018)              # 横向钢筋直径 (m)
        fyh = section_params.get('fyh', 500)                # 横向钢筋屈服强度 (MPa)
        
        # 使用Mander模型计算核心混凝土参数
        fccore, eccore, ecucore = mander.rectangular(lx, ly, coverThick, roucc, sl, dsl, roux, rouy, st, dst, fyh, -fc / UNIT.MPa)
        
    elif section_type.lower() == "arbitrary":
        # 对于任意截面，选择最合适的近似方法
        # 如果提供了直径，使用圆形近似，否则使用矩形近似
        if 'd' in section_params:
            # 使用圆形近似
            d = section_params.get('d', 1.0)
            coverThick = section_params.get('coverThick', 0.08)
            roucc = section_params.get('roucc', 0.03)
            s = section_params.get('s', 0.1)
            ds = section_params.get('ds', 0.014)
            fyh = section_params.get('fyh', 400)
            hoop = section_params.get('hoop', "Spiral")
            
            fccore, eccore, ecucore = mander.circular(hoop, d, coverThick, roucc, s, ds, fyh, -fc / UNIT.MPa)
        else:
            # 使用矩形近似或等效矩形截面
            lx = section_params.get('lx', 4.0)
            ly = section_params.get('ly', 6.5)
            coverThick = section_params.get('coverThick', 0.08)
            roucc = section_params.get('roucc', 0.03)
            sl = section_params.get('sl', 0.1)
            dsl = section_params.get('dsl', 0.032)
            roux = section_params.get('roux', 0.00057)
            rouy = section_params.get('rouy', 0.00889)
            st = section_params.get('st', 0.3)
            dst = section_params.get('dst', 0.018)
            fyh = section_params.get('fyh', 500)
            
            fccore, eccore, ecucore = mander.rectangular(lx, ly, coverThick, roucc, sl, dsl, roux, rouy, st, dst, fyh, -fc / UNIT.MPa)
    else:
        raise ValueError(f"不支持的截面类型: {section_type}")
    
    # 转换为适当的单位
    fccore = fccore * UNIT.MPa
    
    # 计算拉伸参数
    ftc = -fc * tension_factor        # 抗拉强度(取抗压强度的tension_factor倍)
    etc = ftc / Ec                    # 极限拉应变
    
    params = Concrete04Params(fccore, eccore, ecucore, Ec, ftc, etc)
    if print_params:
        print(f"核心混凝土参数: {params}")
    
    return params

def get_bar(bar_tag: bar_tag = "HRB400", print_params: bool = False):
    """
    获取钢筋参数
    
    参数:
    -----
    bar_tag: 钢筋类型
    print_params: 是否打印参数
    
    返回:
    -----
    params: 钢筋参数对象
    """
    # 从标号获取屈服强度
    fy = eval("".join(list(filter(str.isdigit, bar_tag)))) * UNIT.MPa  # 屈服强度 (MPa)
    Es = 200 * UNIT.GPa                                                # 初始弹性模量(200GPa)
    Esh = 2 * UNIT.GPa                                                 # 应变硬化模量 (GPa，通常取Es的1%~2%)
    eps_sh = 0.015                                                     # 应变硬化起始应变
    eps_ult = 0.1                                                      # 峰值强度时的应变  
    fu = (Esh * (eps_ult - eps_sh) + fy)                              # 峰值抗拉强度 (MPa)

    params = BarParams(fy, Es, Esh, eps_sh, eps_ult, fu)
    if print_params:
        print(f"钢筋参数: {params}")
        
    return params

def add_materials(concrete_tag: concrete_tag = "C40", 
                  bar_tag: bar_tag = "HRB400", 
                  sec_params: dict = {}, 
                  matTagC: int = 1, 
                  matTagCCore: int = 2, 
                  matTagS: int = 3, 
                  matTagSteel: int = 4) -> tuple[Concrete04Params, Concrete04Params, BarParams]:
    """
    添加材料定义到OpenSees
    
    参数:
    -----
    concrete_tag: 混凝土标号
    bar_tag: 钢筋类型
    sec_params: 截面参数
    matTagC: 保护层混凝土材料标签
    matTagCCore: 核心混凝土材料标签
    matTagS: 钢筋材料标签
    matTagSteel: 钢筋材料(含限制条件)标签
    """
    # 定义材料
    # 保护层混凝土
    cover = get_concrete_cover(concrete_tag, print_params=True)
    cover.define(matTagC)

    # 核心混凝土
    core = get_concrete_core(concrete_tag, section_type="circular", section_params=sec_params, print_params=True)
    core.define(matTagCCore)

    # 钢筋
    bar = get_bar(bar_tag, print_params=True)
    bar.define(matTagS)
    # 添加最大应变限制
    ops.uniaxialMaterial('MinMax', matTagSteel, matTagS, '-max', 0.1)

    return cover, core, bar
