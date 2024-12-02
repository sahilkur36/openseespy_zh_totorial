"""

演示如何创建一组动态进度条，
显示多个任务（示例中为安装应用程序）的多级进度，
每个任务由多个步骤组成。

"""

import time

from rich.console import Group
from rich.panel import Panel
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


def run_steps(name, step_times, app_steps_task_id):
    """运行单个应用程序的步骤，并更新相应的进度条。"""

    for idx, step_time in enumerate(step_times):
        # 为此步骤添加进度条（时间已过 + 旋转器）
        action = step_actions[idx]
        step_task_id = step_progress.add_task("", action=action, name=name)

        # 运行步骤，更新进度
        for _ in range(step_time):
            time.sleep(0.5)
            step_progress.update(step_task_id, advance=1)

        # 完成后停止并隐藏此步骤的进度条
        step_progress.stop_task(step_task_id)
        step_progress.update(step_task_id, visible=False)

        # 步骤完成后也更新当前应用程序的进度条
        app_steps_progress.update(app_steps_task_id, advance=1)


# 当前应用程序的进度条，仅显示已用时间，
# 安装应用程序时将保持可见
current_app_progress = Progress(
    TimeElapsedColumn(),
    TextColumn("{task.description}"),
)

# 单个应用程序步骤的进度条（步骤完成后将隐藏）
step_progress = Progress(
    TextColumn("  "),
    TimeElapsedColumn(),
    TextColumn("[bold purple]{task.fields[action]}"),
    SpinnerColumn("simpleDots"),
)
# 当前应用程序的进度条（步骤进度）
app_steps_progress = Progress(
    TextColumn(
        "[bold blue]应用程序 {task.fields[name]} 的进度: {task.percentage:.0f}%"
    ),
    BarColumn(),
    TextColumn("（{task.completed} / {task.total} 步骤完成）"),
)
# 总进度条
overall_progress = Progress(
    TimeElapsedColumn(), BarColumn(), TextColumn("{task.description}")
)
# 进度条组；
# 一些始终可见，其他在进度完成时将消失
progress_group = Group(
    Panel(Group(current_app_progress, step_progress, app_steps_progress)),
    overall_progress,
)

# 元组指定每个应用程序的每个步骤所需的时间
step_actions = ("下载", "配置", "构建", "安装")
apps = [
    ("one", (2, 1, 4, 2)),
    ("two", (1, 3, 8, 4)),
    ("three", (2, 1, 3, 2)),
]

# 创建总进度条
overall_task_id = overall_progress.add_task("", total=len(apps))

# 使用自己的实时实例作为包含进度条组的上下文管理器，
# 允许并行运行多个不同的进度条，
# 并动态显示/隐藏它们
with Live(progress_group):
    for idx, (name, step_times) in enumerate(apps):
        # 更新总进度条上的消息
        top_descr = "[bold #AAAAAA]（%d / %d 应用程序已安装）" % (idx, len(apps))
        overall_progress.update(overall_task_id, description=top_descr)

        # 为此应用程序的步骤添加进度条，并运行步骤
        current_task_id = current_app_progress.add_task("正在安装应用程序 %s" % name)
        app_steps_task_id = app_steps_progress.add_task(
            "", total=len(step_times), name=name
        )
        run_steps(name, step_times, app_steps_task_id)

        # 停止并隐藏此特定应用程序的步骤进度条
        app_steps_progress.update(app_steps_task_id, visible=False)
        current_app_progress.stop_task(current_task_id)
        current_app_progress.update(
            current_task_id, description="[bold green]应用程序 %s 已安装!" % name
        )

        # 此任务完成后增加总进度
        overall_progress.update(overall_task_id, advance=1)

    # 总进度条上的消息最终更新
    overall_progress.update(
        overall_task_id, description="[bold green]%s 个应用程序已安装，完成!" % len(apps)
    )