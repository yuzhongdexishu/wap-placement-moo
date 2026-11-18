# WAP 主链

主链只有两个 MCP Server、两个 Agent 和八个工具：

1. BIM：解析 IFC、网格化、导出 PLY、建立 Sionna 场景。
2. WAP：定义问题、调用 Sionna 构建代理、生成候选布局、GA 优化并直接导出 600 点结果。

运行输入保存在 `data/ifc/`、`data/radio_materials/` 和 `data/regions/`。
后者包含最新区域 JSON 及 16 个三维优先区域体积。每个场景在 1F、2F、3F
各取 200 个有效接收点。WAP 数量从 N=1 起确定：先用整数规划对
600 点的 RSRP ≥ −140 dBm 和最小 4 m 站位间距求严格下界，再逐个 N
检查基础档的全部门限（RSRP ≥ −140 dBm、RSRQ ≥ −19.5 dB、SINR ≥ 0 dB）。
GA 随后在选出的 N 下
优化综合分；最终 600 点必须全部达标，否则不导出结果。只有找到的 N
等于 RSRP 集合覆盖的严格下界（或所有更小 N 均已证明不可行）时，
才报告“最小数量已证明”；否则报告“搜索预算内首个找到的可行数量”。
`result/<stage>/process/wap_layout_optimization/02_wap_count_search.json`
记录逐个 N 的判定依据与所选站位。本流程尚不计算业务速率或中断概率。
“最小数量已证明”只针对选定的 600 个接收点和 150 个候选站位；
它不证明整个连续作业区域处处达标。正式部署前应另用独立位置和现场测量验证。

frame 与 enclosure 均按 SC1～SC8 连续编号。frame 原 SC5～SC9 已依次
改为 SC4～SC8；几何和任务内容保持一致，区域 JSON 的 `previous_region_id`
保留了改号前的标识，`source.scenario` 保留原始附表编号。

```powershell
python -m pip install -r requirements.txt
python run_all.py
python send_task_moo.py
```

也可运行 `_launch_services2.ps1` 在后台启动服务。运行时自动创建 `result/`；
主结果写入 `result/frame/final_results/` 与 `result/enclosure/final_results/`。
`04_dense_domain_validation.json` 对同一代理图上的独立有效区域探针给出未达标数；
数量最优性证明只覆盖正式选定的 600 个接收点和 150 个候选 WAP 站位。

MOO 完成后运行 `python run_experiment_expert.py` 与
`python run_experiment_geometric.py`，再运行
`python run_baseline_comparison.py` 把 PM 与原始 expert/geometry 站位放进同一张
Sionna 代理图，重新计算所有方法的对照分数。两组对照使用相同 WAP 数和 600 点，
分别报告综合目标值 F 和未达标接收点数。F 始终按逐点三指标归一化后的加权均分计算，约束是否满足另行记录，不修改 F。GA 对不可行布局使用失效点数排序，内部排序值不是最终目标值。
绘图代码已恢复：`python plot_uplift.py` 仅绘制两阶段的 F 柱图，未达标布局用斜线和 `*` 标识，具体失效数见 `result/comparison_table.csv`。所有图片文字统一为 Times New Roman。
`python plot_group_violin.py` 绘制三项指标分布，
`python render_floor_bases.py` 和 `python plot_registered_floor_heatmaps.py` 按原建筑图相机参数生成与各楼层配准的热力图；`python plot_stacked_registered_heatmaps.py` 将三层合成新的三维概览。热力层取自 0.75 m 间距的 Sionna 路径增益网格，覆盖建筑有效楼板区域（包括前厅），不再按接收器采样点距离裁掉网格。零路径增益网格仅以色阶最低色显示，不能解释为有限的射频测量值；各图的零路径网格数量见 `plots/registered_heatmaps/grid_coverage_report.json`。`python plot_3d_metric_heatmaps.py` 保留旧版三维图供历史检查。
绘图脚本校验区域哈希、600 点集合、对照的原站位评分来源和同一张传播图。工具 7 在确定 WAP 数量后，会额外生成多个满足 600 点门限的初始布局，避免 GA 只从一个可行布局出发。
六模型任务矩阵可由 `python export_task_table.py` 导出，
`python send_task2.py --dry-run` 可预览任务，当前五档功率配置的完整六模型矩阵共有 480 个任务（6 模型 × 2 阶段 × 5 档 × 8 场景）。SP1–SP5 的发射功率依次为 13、18、23、28、33 dBm，载波频率均为 3.5 GHz；MOO 使用 SP3/SC1/DeepSeek。当前 PM 与 expert/geometry 对比使用每 TX 1,000,000 样本；最新结果见 `SP3_RUN_REPORT.md`。现存六模型 12 项结果属于旧三档配置，需重新运行才代表新配置。
