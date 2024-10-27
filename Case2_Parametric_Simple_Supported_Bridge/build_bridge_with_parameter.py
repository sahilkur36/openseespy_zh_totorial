from __future__ import annotations
import openseespy.opensees as ops
import numpy as np
from pyDOE import lhs
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple, Literal
from pathlib import Path
from scipy.stats import uniform, norm, lognorm
import opstool as opst
from collections import defaultdict
import json
from rich.console import Group
from rich.panel import Panel
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

def progress_listener(progress_queue, total_tasks):
    # 创建进度条实例
    task_progress = Progress(
        TextColumn("[bold blue]{task.fields[name]}: {task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    overall_progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
    )
    progress_group = Group(
        Panel(task_progress),
        Panel(overall_progress),
    )

    with Live(progress_group, refresh_per_second=10):
        overall_task = overall_progress.add_task("总进度", total=total_tasks)
        task_ids = {}
        completed_tasks = 0

        while completed_tasks < total_tasks:
            try:
                message = progress_queue.get(timeout=0.1)
            except Exception:
                continue

            if message[0] == 'DONE':
                worker_id = message[1]
                completed_tasks += 1
                overall_progress.advance(overall_task)
                # 标记任务完成
                task_progress.update(task_ids[worker_id], completed=task_progress.tasks[task_ids[worker_id]].total)
                task_progress.stop_task(task_ids[worker_id])
                continue

            elif message[0] == 'PROGRESS':
                worker_id = message[1]
                progress_value = message[2]
                total_value = message[3]

                if worker_id not in task_ids:
                    task_id = task_progress.add_task(f"Worker {worker_id}", total=total_value, name=f"Worker {worker_id}")
                    task_ids[worker_id] = task_id

                task_progress.update(task_ids[worker_id], completed=progress_value)

@dataclass
class BridgeSample:
    pier_height: float
    span_length: float
    num_spans: int
    bearing_type: str
    friction_force: float
    initial_stiffness: float
    earthquake_record: str

    @staticmethod
    def samples_to_dict(samples: List[BridgeSample]) -> List[dict]:
        """ 将 BridgeSample 列表转换为 JSON 可保存的字典 """
        return [
            {
                'pier_height': sample.pier_height,
                'span_length': sample.span_length,
                'num_spans': sample.num_spans,
                'bearing_type': sample.bearing_type,
                'friction_force': sample.friction_force,
                'initial_stiffness': sample.initial_stiffness,
                'earthquake_record': str(sample.earthquake_record)  # 转为字符串存储
            }
            for sample in samples
        ]
    
    @staticmethod
    def save_samples_as_json(samples, file_path: Path):
        with open(file_path, 'w') as f:
            json.dump(BridgeSample.samples_to_dict(samples), f, indent=4)

    @staticmethod
    def load_samples_from_json(file_path: Path) -> List[BridgeSample]:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [
            BridgeSample(
                pier_height=item['pier_height'],
                span_length=item['span_length'],
                num_spans=item['num_spans'],
                bearing_type=item['bearing_type'],
                friction_force=item['friction_force'],
                initial_stiffness=item['initial_stiffness'],
                earthquake_record=Path(item['earthquake_record'])  # 转回 Path 对象
            )
            for item in data
        ]

@dataclass
class BridgeParameters:
    pier_height_range: Tuple[float, float]
    span_length_range: Tuple[float, float]
    num_spans_range: Tuple[int, int]
    bearing_types: List[str]
    friction_force_range: Tuple[float, float]
    initial_stiffness_range: Tuple[float, float]
    earthquake_records: List[str]
    samples_file:Path = Path("./samples.txt")

    def sample_parameters(self, num_samples: int) -> List[BridgeSample]:
        if self.samples_file.exists():
            samples = BridgeSample.load_samples_from_json(self.samples_file)
            return samples

        lhs_samples = lhs(7, samples=num_samples)  # Sampling for continuous variables

        # 定义每个连续变量的分布函数（此处仅做示意）
        dist_funcs = {
            'pier_height': uniform(loc=self.pier_height_range[0], scale=self.pier_height_range[1] - self.pier_height_range[0]),
            'span_length': uniform(loc=self.span_length_range[0], scale=self.span_length_range[1] - self.span_length_range[0]),
            'num_spans': uniform(loc=self.num_spans_range[0], scale=self.num_spans_range[1] - self.num_spans_range[0]),
            'friction_force': norm(loc=(self.friction_force_range[0] + self.friction_force_range[1]) / 2, scale=(self.friction_force_range[1] - self.friction_force_range[0]) / 6),
            'initial_stiffness': lognorm(s=0.5, scale=np.exp((np.log(self.initial_stiffness_range[0]) + np.log(self.initial_stiffness_range[1])) / 2)),
        }

        samples = []
        for i in range(num_samples):
            # 通过PPF函数将LHS采样转换为实际参数值
            sample = BridgeSample(
                pier_height=dist_funcs['pier_height'].ppf(lhs_samples[i, 0]),
                span_length=dist_funcs['span_length'].ppf(lhs_samples[i, 1]),
                num_spans=int(dist_funcs['num_spans'].ppf(lhs_samples[i, 2])),
                bearing_type=self.bearing_types[int(lhs_samples[i, 3] * len(self.bearing_types))],
                friction_force=dist_funcs['friction_force'].ppf(lhs_samples[i, 4]),
                initial_stiffness=dist_funcs['initial_stiffness'].ppf(lhs_samples[i, 5]),
                earthquake_record=self.earthquake_records[int(lhs_samples[i, 6] * len(self.earthquake_records))]
            )
            samples.append(sample)
        return samples

def recorderfunc(results: defaultdict):
    Accel = ops.nodeAccel(1000+int(10/2)+1, 1)  # 获取主梁中间节点的x向加速度
    results["Accel"].append(Accel)
    disp = ops.nodeDisp(100+5+1, *[1,5])  # 获取左墩墩顶节点的x向位移和转角
    results["Disp"].append(disp)
    for etag in ops.getEleTags():  # 可以很方便的遍历单元号（如果需要）
        # 提单元内力
        resp = ops.eleResponse(etag, "localForce")
        results[f"AxisForce-Ele{etag}"].append(resp)
        # 提取单元变形
        resp = ops.eleResponse(etag, "deformation")
        results[f"AxisDefo-Ele{etag}"].append(resp)
        # 提取单元上材料应力应变
        resp = ops.eleResponse(etag, "material", "stressStrain")
        results[f"stressStrain-Ele{etag}"].append(resp)
        
    results["Time"].append(ops.getTime())  # 别忘了每步对应的时间

def define_sections(verbose=False):
    # 定义截面
    # 定义非线性柱的材料
    # ------------------------------------------
    # 混凝土                          tag     f'c    ec0    f'cu    ecu
    # 约束核心混凝土
    ops.uniaxialMaterial("Concrete01", 1, -41000, -0.004, -34470, -0.014)
    # 非约束保护层混凝土
    ops.uniaxialMaterial("Concrete01", 2, -34470, -0.002, -25000, -0.006)
    # 钢筋材料
    fy, E, b = 400e3, 206.84e6, 0.01  # 屈服应力, 杨氏模量,硬化率
    ops.uniaxialMaterial("Steel01", 3, fy, E, b)
    # 墩柱尺寸
    colWidth, colDepth = 1, 2
    cover = 0.08
    As = 0.02  # 钢筋面积

    # 从参数派生的一些变量
    y1, z1 = colDepth / 2.0, colWidth / 2.0
    # 创建纤维截面
    ops.section("Fiber", 991, "-GJ", 1e10)
    # 创建混凝土核心纤维
    ops.patch("rect", 1, 10, 1, cover - y1, cover - z1, y1 - cover, z1 - cover)
    # 创建混凝土保护层纤维（顶部、底部、左侧、右侧）
    ops.patch("rect", 2, 10, 1, -y1, z1 - cover, y1, z1)
    ops.patch("rect", 2, 10, 1, -y1, -z1, y1, cover - z1)
    ops.patch("rect", 2, 2, 1, -y1, cover - z1, cover - y1, z1 - cover)
    ops.patch("rect", 2, 2, 1, y1 - cover, cover - z1, y1, z1 - cover)
    # 创建钢筋纤维（右侧、中间、左侧）
    ops.layer("straight", 3, 3, As, y1 - cover, z1 - cover, y1 - cover, cover - z1)
    ops.layer("straight", 3, 2, As, 0.0, z1 - cover, 0.0, cover - z1)
    ops.layer("straight", 3, 3, As, cover - y1, z1 - cover, cover - y1, cover - z1)

    pier_sec = 1    # 墩柱截面编号
    ops.uniaxialMaterial("Elastic", 103, 1e10)
    ops.section("Aggregator", pier_sec, *[103, "T"], "-section", 991)

    # 并行计算时不输出
    if verbose:
        print(f"墩柱截面已定义！tag={pier_sec}")
    return pier_sec

def define_gravity_loads(ts_tag:int=1,pattern_tag:int=1,direction:Literal['X','Y','Z'] = 'Z', g:float=-9.81):
    directionDict = {'X': 1, 'Y': 2, 'Z': 3}
    # ====== 定义重力荷载 ====== #

    # 创建带有线性时间序列的Plain荷载模式
    ops.timeSeries('Linear', ts_tag)
    ops.pattern('Plain', pattern_tag, ts_tag)

    # 为每个点创建荷载
    for node in ops.getNodeTags():
        # 读取对应方向节点质量并乘以重力加速度
        dirdof = directionDict[direction]
        P = ops.nodeMass(node,dirdof)*g    # 节点z向质量*重力加速度
        if P > 0:
            load = [0.0 for _ in range(6)]
            load[dirdof-1] = P
            ops.load(node, *load)

def gravity_analysis(verbose=False):
    # ====== 静力分析设置 ====== #
    Nsteps = 10
    ops.system('BandGeneral')   # 求解器类型，BandGeneral适用于带状矩阵，如梁柱结构
    ops.constraints('Transformation')  # 约束处理方法，Transformation，适用于大多数问题
    ops.numberer('RCM')  # 节点编号方法，RCM (Reverse Cuthill-McKee)算法，可以减少带宽
    ops.test('NormDispIncr', 1.0e-12, 15, 3)  # 收敛测试:位移增量范数,容差1.0e-12,最大迭代数15
    ops.algorithm('Newton')  # 解算法，Newton-Raphson法，适用于大多数非线性问题
    ops.integrator('LoadControl', 1 / Nsteps) # Nsteps与步长的乘积一定要为1，代表施加一倍荷载，乘积为2代表施加两倍荷载
    ops.analysis('Static')
    
    for i in range(Nsteps):
        ok = ops.analyze(1)

    # 设置重力荷载为常量并重置域中的时间(这一步很重要，避免自重继续随时间增加)
    ops.loadConst('-time', 0.0)
    if verbose: 
        print("重力分析完成")

def build_bridge_model(params,progress_queue = None,task_id = None, verbose=False):
    """
    Args:
        params (object): 包含桥梁参数的对象，包括跨度长度、墩高、支座类型、初始刚度、摩擦力和地震记录文件路径等。
        verbose (bool): 是否打印详细信息。默认为True。
    Returns:
        list: 包含分析结果的列表，每个元素为主梁中点在每个时间步的位移。
    Raises:
        NotImplementedError: 如果某代码未实现，则抛出此异常。
    示例:
        from pathlib import Path
        params = {
            'span_length': 30,
            'pier_height': 10,
            'bearing_type': 'Rubber',
            'initial_stiffness': 1000,
            'friction_force': 0.5,
            'earthquake_record': Path('path/to/record.txt')
        }
        results = build_bridge_model(params)

    """
    ops.wipe()
    
    # 单位规定：kN, m, s
    # 创建ModelBuilder（三维模型，每个节点6个自由度）
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # ====== 参数定义 ====== #
    L = params.span_length  # 跨度长度(m)
    H = params.pier_height  # 墩高(m)
    nH, nL = 5, 10  # 高度和长度方向的分段数

    g = 9.81  # 重力加速度(m/s^2)
    m_girder = 50  # 每延米主梁质量(ton/m)
    m_pier = 2.5*1*2  # 每延米墩柱质量(ton/m)

    # 创建墩的节点并添加质量
    y = 0.0  # 所有点y坐标均为0
    for i in range(2):
        x = i * L
        for j in range(nH+1):
            z = j * H / nH
            node_tag = (i+1)*100 + j+1
            ops.node(node_tag, x, y, z)
            M = m_pier*H
            m = M/nH/2 if (i == 0 or i == nH) else M/nH
            ops.mass(node_tag, m, m, m, 0.0, 0.0, 0.0)

    # 创建主梁的节点
    for i in range(nL+1):
        x = i * L / nL
        z = H
        node_tag = 1000 + i+1
        ops.node(node_tag, x, y, z)
        M = m_girder*L
        m = M/nL/2 if (i == 0 or i == nL) else M/nL
        ops.mass(node_tag, m, m, m, 0.0, 0.0, 0.0)

    # 固定墩柱底部
    ops.fix(101, 1, 1, 1, 1, 1, 1)
    ops.fix(201, 1, 1, 1, 1, 1, 1)

    pier_sec = define_sections(verbose)

    # 定义单元
    # 桥墩单元
    # 定义单元之前先要定义几何变换
    ops.geomTransf('Linear', 1, 0, 1, 0)
    # 设置元素长度方向的积分点数量(dispBeamColumn需要)
    NP = 5
    # 使用Lobatto积分，id为2
    ops.beamIntegration('Lobatto', 2, pier_sec, NP)
    # 使用塑性梁柱单元创建桥墩
    for i in range(2):
        for j in range(nH):
            node1 = (i+1)*100 + j+1
            node2 = (i+1)*100 + j+2
            ele_tag = (i+1)*100 + j+1
            # 倒数第二个参数是几何变换的tag，最后一个参数是积分点的tag
            ops.element('dispBeamColumn', ele_tag, node1, node2, 1, 2)
        if verbose:
            print(f"{'左' if i==0 else '右'}桥墩单元已创建！")
            
    # 定义梁元素的几何变换
    ops.geomTransf('Linear', 2, 0, 1, 0)

    # 创建弹性梁单元
    for i in range(nL):
        ele_tag = 1000 + i + 1
        node1 = 1000 + i + 1
        node2 = node1 + 1
        #                                    tag, ndI,     ndJ,    A,     E,   Iz,   Iy,    G,    J, transfTag
        ops.element('elasticBeamColumn', ele_tag, node1, node2, 0.86, 210e6, 23.2, 2.32, 81e6, 3.13, 2)

    # 定义支座
    rigid_tag, free_tag = 9903, 9904
    ops.uniaxialMaterial('Elastic', rigid_tag, 1e7)
    ops.uniaxialMaterial('Elastic', free_tag, 10)

    if params.bearing_type == 'Rubber':
        n_bear = 10
        ops.uniaxialMaterial('Elastic', 9901, n_bear*params.initial_stiffness)
        ops.uniaxialMaterial('Steel01', 9902, params.friction_force, n_bear*params.initial_stiffness, 0.000001)
        ops.element('zeroLength', 11, *[100+nH+1, 1001], '-mat', *[9902, 9902, 9901, rigid_tag, free_tag, free_tag], '-dir', *[1, 2, 3, 4, 5, 6])
        ops.element('zeroLength', 21, *[200+nH+1, 1000+nL+1], '-mat', *[9902, 9902, 9901, rigid_tag, free_tag, free_tag], '-dir', *[1, 2, 3, 4, 5, 6])
    elif params.bearing_type == 'Frame':
        ops.element('zeroLength', 11, *[100+nH+1, 1001], '-mat', *[rigid_tag, rigid_tag, rigid_tag, rigid_tag, rigid_tag, rigid_tag], '-dir', *[1, 2, 3, 4, 5, 6])
        ops.element('zeroLength', 21, *[200+nH+1, 1000+nL+1], '-mat', *[rigid_tag, rigid_tag, rigid_tag, rigid_tag, rigid_tag, rigid_tag], '-dir', *[1, 2, 3, 4, 5, 6])
    else:
        raise NotImplementedError(f"支座类型{params.bearing_type}未实现")
    if verbose:
        print("支座已创建！")

    # 定义重力荷载
    define_gravity_loads(ts_tag=1, pattern_tag=1, direction='Z', g=-9.81)
    # 进行重力分析
    gravity_analysis(verbose)

def earthquake_analysis(params,
                        progress_queue = None,
                        task_id = None,
                        verbose=False,
                        *args):
    # 地震动分析
    # 删除旧的分析
    ops.wipeAnalysis()

    # 修改分析设置以适应地震分析
    ops.system('BandGeneral')
    ops.constraints('Plain')
    ops.test('NormDispIncr', 1.0e-12,  10)
    ops.algorithm('Newton')
    ops.numberer('RCM')
    ops.integrator('Newmark',  0.5,  0.25)
    ops.analysis('Transient')

    # 记录一下地震分析之前的结构周期
    import math
    numEigen = 3
    eigenValues = ops.eigen(numEigen)
    periods = [2*math.pi/math.sqrt(eigenValues[i]) for i in range(numEigen)]
    if verbose:
        print("地震分析开始时的周期(s):", periods)

    # 利用前两阶周期计算瑞利阻尼因子
    xi = 0.05 # 阻尼比0.05
    omega1,omega2 = 2*math.pi/periods[0],2*math.pi/periods[1]
    # 计算瑞丽阻尼系数 α 和 β
    alpha = 2 * xi * (omega1 * omega2) / (omega1 + omega2)
    beta = 2 * xi / (omega1 + omega2)
    ops.rayleigh(alpha, 0.0, beta, 0.0)

    # 读取地震记录
    wavepath = params.earthquake_record
    # 设置时间序列
    dt_path = wavepath.parent / 'dt.txt'
    step_path = wavepath.parent / 'motionStep.txt'
    waveid = int(wavepath.stem)
    with open(dt_path, 'r') as f:
        dt_lines = f.readlines()
        dt = float(dt_lines[waveid].strip())
    with open(step_path, 'r') as f:
        nsteps = f.readlines()
        nsteps = int(nsteps[waveid].strip())

    ts_tag = 3
    gfactor = 9.81
    ops.timeSeries('Path', ts_tag, '-filePath', str(wavepath), '-dt', dt, '-factor', gfactor)
    # 创建地震动荷载模式
    axis = 1    # 地震动输入方向
    ops.pattern('UniformExcitation', 2, axis, '-accel', ts_tag)

    # 定义一个字典用于存储结果
    RESULTS = defaultdict(lambda: [])
    RESULTS["params"] = params  # 保存参数

    results = run_earthquake_analysis(nsteps = nsteps,
                            dt = dt,
                            recorderfunc=recorderfunc,
                            recorderdict=RESULTS,
                            progress_queue = progress_queue,
                            task_id = task_id,
                            verbose=verbose,
                            on_notebook=False)
    return results

def run_earthquake_analysis(nsteps:int,
                            dt:float,
                            recorderfunc,
                            recorderdict:defaultdict,
                            progress_queue = None,
                            task_id = None,
                            verbose:bool=False,
                            on_notebook=False):
    if on_notebook:
        from IPython.display import clear_output

    # 初始化地震分析
    analysis = opst.SmartAnalyze(analysis_type="Transient", printPer=1e5)
    segs = analysis.transient_split(nsteps)
    process_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    
    for idx, seg in enumerate(segs):
        if on_notebook:
            clear_output(wait=True)  # 清除之前的输出，不是分析必要的，只是为了看起来更美观
        analysis.TransientAnalyze(dt)  # 分析一个dt

        # 第二种记录方法，每步利用提取函数提取响应
        recorderfunc(recorderdict)

        # 发送进度更新
        progress_queue.put(('PROGRESS', task_id, idx + 1, nsteps))

    # 任务完成后，发送完成信号
    progress_queue.put(('DONE', task_id))
                
    return recorderdict

def run_simulation(sample,progress_queue = None,task_id = None, verbose=False):
    build_bridge_model(sample,progress_queue,task_id)
    # 地震分析
    results = earthquake_analysis(sample,progress_queue,task_id,verbose)
    return results

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define the path to the earthquake records
    wavepath = Path(__file__).parent / 'Waves'
    # Define the path to save the sample file
    sample_file = Path(__file__).parent / 'samples.txt'
    # 定义参数范围
    bridge_params = BridgeParameters(
        pier_height_range=(10, 30),
        span_length_range=(30, 40),
        num_spans_range=(3, 6),
        bearing_types=["Rubber", "Frame"],
        friction_force_range=(2000, 5000),
        initial_stiffness_range=(5000, 10000),
        earthquake_records=[file for file in wavepath.glob("*.txt") if file.stem.isdigit()]
    )

    samples = bridge_params.sample_parameters(num_samples=5)
    BridgeSample.save_samples_as_json(samples, bridge_params.samples_file)
    
    # for sample in samples:
    #     print(sample)

    # Run the main simulation
    # print(samples[0])
    # results = build_bridge_model(samples[0])

    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results = pool.map(run_simulation, samples)

    # 创建进程间队列
    progress_queue = mp.Queue()
    total_tasks = len(samples)

    # 启动进度监听器
    listener = mp.Process(target=progress_listener, args=(progress_queue, total_tasks))
    listener.start()

    # 启动子进程
    processes = []
    for task_id, sample in enumerate(samples):
        recorderdict = defaultdict()
        p = mp.Process(target=run_simulation, args=(
            sample, progress_queue, task_id))
        processes.append(p)
        p.start()

    # 等待所有子进程完成
    for p in processes:
        p.join()

    # 等待监听器完成
    listener.join()


    # # Convert results to numpy array for visualization
    # results_array = np.array(results)

    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # for i, result in enumerate(results_array):
    #     plt.plot(result, label=f'Sample {i+1}')
    
    # plt.xlabel('Time Step')
    # plt.ylabel('Displacement')
    # plt.title('Bridge Displacement Over Time')
    # plt.legend()
    # plt.show()