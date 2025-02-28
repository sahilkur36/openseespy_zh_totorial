from sectionproperties.pre import Geometry
from shapely import Polygon
import numpy as np
import opstool as opst
from typing import Literal
from unitsystem import UNIT

def create_arc(center, radius, start_angle, end_angle, num_points=20):
        """
        Create a half-circle arc.
        
        :param center: Tuple (x, y) representing the center of the circle.
        :param radius: Radius of the circle.
        :param start_angle: Starting angle of the arc in degrees.
        :param end_angle: Ending angle of the arc in degrees.
        :param num_points: Number of points to generate the arc.
        :return: Shapely LineString representing the half-circle arc.
        """
        angles = [np.radians(start_angle + i * (end_angle - start_angle) / (num_points - 1))
                for i in range(num_points)]
        points = [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
                for angle in angles]

        return points

def box_section_geoms(width:float,height:float):
    d_width_concrete = 105
    chamfer_width,chamfer_height = 120,60
    d_height_concrete = 70
    R = height/2
    # clockwise(八边形)
    inner_points = [
        (-width/2+d_width_concrete+chamfer_width, height/2-d_height_concrete),
        (width/2-d_width_concrete-chamfer_width, height/2-d_height_concrete),
        (width/2-d_width_concrete, height/2-d_height_concrete-chamfer_height),
        (width/2-d_width_concrete, -height/2+d_height_concrete+chamfer_height),
        (width/2-d_width_concrete-chamfer_width, -height/2+d_height_concrete),
        (-width/2+d_width_concrete+chamfer_width, -height/2+d_height_concrete),
        (-width/2+d_width_concrete, -height/2+d_height_concrete+chamfer_height),
        (-width/2+d_width_concrete, height/2-d_height_concrete-chamfer_height)]

    outer_points = []
    outer_points.append((-width/2+R,R))
    outer_points.append((width/2-R,R))
    outer_points.extend(create_arc((width/2-R,0),R,90,-90))
    outer_points.append((width/2-R,-R))
    outer_points.append((-width/2+R,-R))
    outer_points.extend(create_arc((-width/2+R,0),R,270,90))

    combinedgeom = opst.pre.section.create_polygon_patch(outer_points, holes=[inner_points])
    
    return combinedgeom

def extract_boundaries(geometry:Geometry,unit:Literal["m","cm","mm"] = "cm"):
    """
    从Geometry对象中提取边界线，并进行单位换算
    
    参数:
    -----
    geometry: sectionproperties.pre.Geometry
        Geometry对象
    unit: Literal["m","cm","mm"]
        目标单位，默认为"cm"
    
    返回:
    -----
    exterior: list
        外部边界坐标列表 [(x1, y1), (x2, y2), ...]，已转换为指定单位
    interiors: list
        内部边界坐标列表的列表 [[(x1, y1), ...], [(x2, y2), ...], ...]，已转换为指定单位
    """
    # 使用UNIT对象获取换算系数
    if unit == "m":
        factor = UNIT.m
    elif unit == "cm":
        factor = UNIT.cm
    elif unit == "mm":
        factor = UNIT.mm
    else:
        factor = UNIT.cm  # 默认使用厘米
    
    polygon = geometry.geom
    
    # 获取外部边界并进行单位换算
    exterior_coords = list(polygon.exterior.coords)
    exterior = [(x * factor, y * factor) for x, y in exterior_coords]
    
    # 获取内部边界并进行单位换算
    interiors = []
    for interior in polygon.interiors:
        interior_coords = list(interior.coords)
        converted_interior = [(x * factor, y * factor) for x, y in interior_coords]
        interiors.append(converted_interior)
    
    return exterior, interiors



if __name__ == "__main__":
    # 生成箱墩截面polygon
    combinedgeom = box_section_geoms(width=900, height=300)
    # 绘制截面
    combinedgeom.plot_geometry()