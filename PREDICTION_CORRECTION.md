# 二次曲面预测均值方向修正（第一版）

## 运行

在 MATLAB 项目目录下运行：

```matlab
runSMDPredictionComparison
```

默认对 SMD1–SMD8 的 `(2+3)` 维问题，每题用 10 个相同种子分别运行原版和修正版，共 160 次。预算仍为 `UmaxFEs=2500`、`UmaxImprFEs=350`、`LmaxFEs=250`、`LmaxImprFEs=25`；种群规模继续由原算法的 `4+floor(3*log(BI.dim))` 决定，联合维度 5 时为 8。

输出保存到 `results_prediction_comparison/时间戳/`。每次运行独立保存：

```text
SMD1/DU2_DL3/baseline/SMD1_DU2_DL3_run01.mat
SMD1/DU2_DL3/prediction/SMD1_DU2_DL3_run01.mat
```

MAT 中保存 `ins`（包括全部原有逐代/个体日志）、顶层 `predictionLog`、`BI` 和 `metadata`（种子与运行前后 RNG 状态）。批次还保存源代码快照、配置、逐次结果和问题级汇总 `comparison_summary.csv/.mat`。不会把十次运行合并成一条轨迹；汇总文件只是另行计算统计量。已有批次目录会被拒绝覆盖。

先做较小的对比也可运行：

```matlab
runSMDPredictionComparison([],1,false,[1 5 8])
```

加入限幅版本，比较三个版本：

```matlab
runSMDPredictionComparison([],10,true)
```

直接调用算法仍用原来的 `BLCMAES(BI)` 接口：

```matlab
BI.enablePredictionCorrection = true;
BI.predictionCorrection = struct('weight',0.5, ...
    'startGeneration',10,'correctionInterval',1, ...
    'limitCorrection',false);
ins = BLCMAES(BI);
save('my_prediction_run.mat','ins','BI');
```

将 `BI.enablePredictionCorrection=false` 或省略此字段，即关闭策略。若还要保存原版的逐代对照向量，设置 `BI.predictionCorrection.logEnabled=true`。现有 `runSMDGenerationDiagnostics` 仍保持默认原版记录入口。本策略及专用对照日志只支持二维上层的 SMD1–SMD8。

## UF 的含义

`BLEA_D/ulTestProblem.m` 的 SMD 函数在返回前执行 `functionValue=-functionValue`。`BLCMAES` 直接将这个返回值保存为 `POP(i).UF`，无约束时 `fit=UF`，并按 `sort(-[POP.fit])` 选择精英。因此此项目和已有 `_population.csv` 中的 **UF 越大越好，是原始最小化目标的负值**。

在线拟合直接使用 UF，并要求 Hessian 负定、取极大值点。原始上层目标为 `-UF`。当前仓库没有此前离线拟合脚本可供逐行核验；上述符号核查基于实际评价函数和日志来源，与用户描述的“离线取极大值”一致。

## 采样、历史及代数

`BLCMAES.m` 的原有执行顺序是：

1. 以 `CMA.xmean` 采样联合向量，取前 `BI.u_dim` 个坐标为上层个体；
2. 对每个个体执行原有下层搜索和上层评价；
3. 原有适应度赋值、refinement、终止检查；
4. `updateCMAES` 正常更新均值、`ps`、`pc`、`C`、`sigma` 和特征分解；
5. 新增 `applyUpperPredictionCorrection`，只替换上层均值坐标；
6. 原有逐代日志记录实际下一代采样均值。

在第 g 次循环开始记录 `m_before`。本代最终 `POP(i).UX` 与 `POP(i).UF` 成对进入历史，包含原有 refinement 接受后的结果，与先前种群明细 CSV 的时点一致。不会再把独立历史 elite 重复加入种群档案。

从 `g=10` 起，每一代都拟合并尝试修正（`correctionInterval=1`）。每次仍保留 `[g-9,g]` 这十代，从中按到 `m_before` 的欧氏距离选择最近 40 个有限样本。同距离按历史入档顺序打破平局，不使用 UF 排序。正常 CMA 更新虽然已完成，拟合参考点仍是更新前的 `m_before`，没有任何 g+1 代样本。

日志中的第 g 行描述“采样第 g 代的均值 → 采样第 g+1 代的均值”。原有汇总日志第 0 行是初始均值，非终止第 g 行是更新后的均值。因此预测日志第 g 行的 `m_before` 对应原有日志第 g-1 行；`m_new` 对应原有日志第 g 行。原算法在终止代不更新参数，此时预测日志标记 `terminal_no_update`，`m_cma` 等不存在的量为 NaN。

## 拟合及失败处理

`fitLocalUpperQuadratic.m` 使用选中 40 点的最大距离作为 s，并按 `(x-m_before)/s` 归一化。设计矩阵完整包含：

```text
[1, z1, z2, z1^2, z1*z2, z2^2]
```

最小二乘系数为 `a=design\UF`，Hessian 为 `[2*a(4),a(5);a(5),2*a(6)]`，预测点为 `m_before+s*(-(H\a(2:3)))'`。没有显式矩阵逆，没有单位方向归一化。

样本不足 40、尺度接近零、设计矩阵条件数超过 `1e10`、Hessian 的 `rcond<1e-10`、Hessian 不满足负定条件（即 `chol(-H)` 失败）、预测或修正不是有限实数时，本代保持正常 CMA 均值。日志保存具体原因，以及选中样本、距离、尺度、系数、Hessian 和条件数。理论最优坐标不参与拟合或修正。

## 向量与状态

主实验把正常 CMA-ES 均值和预测点作为两个点进行加权混合：

```text
d_cma  = m_cma - m_before
d_pred = x_pred - m_before
alpha  = 0.5
m_new  = (1-alpha)*m_cma + alpha*x_pred
d_new  = m_new - m_before
```

因此 `d_new=0.5*d_cma+0.5*d_pred`。这里的两个 0.5 分别是正常 CMA-ES
更新均值、预测点的系数；并非旧版“保留完整 CMA 位移后再额外叠加预测位移”的公式。

修正只作用于 `CMA.xmean(1:2)`。下层均值、`pc`、`ps`、`C`、`sigma`、`B`、`D`、`invsqrtC` 均保留刚完成的原有更新。修正相当于更新之后额外平移采样中心，**不把未经真实评价的预测位移计入进化路径**。下一次更新的 `xold` 自然读取这个实际采样中心，因此相对步长仍从正确中心计算。这是一种明确的外部均值干预，不宣称与将合成位移重新用于标准 CMA 路径更新等价。

原算法维护的是上下层联合协方差矩阵。下一代上层采样的边缘分布中心为新上层均值，协方差为 `sigma^2*C(1:2,1:2)`，联合分布其余状态保持原有更新结果。

原始预测点允许位于边界外；合成均值若越界，沿用原有中点修复方式：越界坐标设为“正常 CMA 均值该坐标与相应边界”的中点。日志单独保存 `boundary_adjustment`，因此边界处理后实际位移可能不再严格等于未修复的 `d_new`。

可选限幅开关为 `limitCorrection=true`。此时先限制相对于正常 CMA-ES 均值的加权修正向量 `weight*(x_pred-m_cma)` 的模长，上限为：

```text
maxCorrectionSigmaFactor * sigma_after * sqrt(trace(C_after(1:2,1:2)))
```

默认因子为 1，表示上层更新后分布的 RMS 半径。限幅只同比缩短修正向量，不改变其符号或方向；随后再检查边界。主实验此开关关闭。

## 逐代日志及对比解释

`predictionLog` 的主要字段包括 `generation`、`correction_scheduled`、`m_before`、`m_cma`、`x_pred`、`d_cma`、`d_pred`、`d_new`、`m_new`、`valid_fit`、`weight_used`、`correction_norm`、`generation_best_UF`、`best_so_far_UF`、`elite_UF`、`UFE`、`LFE_total`。`model` 保存拟合追溯信息。`correction_scheduled=true` 表示这一代命中了配置的执行时点；若模型仍无效，会保留对应失败原因。`best_so_far_UF` 是已见种群及 elite 中最大的 UF，`generation_best_UF` 沿用原算法 RF-aware 最好个体选择。

为分析修正是否过大、外推或破坏联合 CMA-ES 状态，日志还保存纯诊断字段：`normal_move_norm`、`predicted_move_norm`、`requested_correction_norm`、`correction_to_cma_ratio`、`correction_to_sigma_ratio`、`correction_mahalanobis`、`cos_pred_cma`、`cos_correction_pc`、`cos_correction_ps`；二次曲面质量字段 `fit_train_RMSE/R2`、`fit_LOOCV_RMSE/R2`、`z_pred_norm`、`prediction_to_nearest_sample(_normalized)`、`prediction_in_hull`、曲率/样本分布条件数；以及联合协方差耦合字段 `upper_lower_covariance_norm`、`upper_lower_correlation_norm`、`conditional_lower_shift_*`。所有量只读现有历史样本及当前 CMA 状态。

第 `t` 条日志在第 `t+1` 代真实评价完成后，会补写 `next_generation_best_UF_delta`、`next_population_UF_median_delta`、`next_elite_UF_delta`、`next_best_so_far_UF_delta` 和下一代 UFE/LFE 差值。这些结果只用于离线评估“第 t 代修正后下一代种群是否变好”，绝不参与第 `t` 代或后续代的修正决策。

`weight_requested` / `prediction_weight_requested` 是预测点系数，`normal_weight_requested=1-weight_requested` 是正常 CMA-ES 均值系数；对应的 `*_used` 字段保存限幅后的有效系数。`correction_actual=m_new-m_cma` 与 `correction_norm` 反映含边界处理后的实际修正。边界修复后不能再用单个标量权重概括实际修正。`d_new` 保存合成但尚未边界修复的向量，`d_actual=m_new-m_before` 保存实际移动。

新增策略函数没有调用评价函数、下层优化或 RNG；原有采样、评价与停止公式不变。改变轨迹后终止代数和下层收敛过程可能不同，因此两版累计 UFE/LFE 本来就可能不同。“不增加额外评价”表示拟合和修正本身不调用真实评价，并不承诺两个版本的总次数相同。

汇总中的上层误差为 `abs(UF-BI.u_fopt)`，未进行精度下限截断。成功标准沿用 `error < BI.u_ftol`，当前代码最终设置的 `BI.u_ftol` 为 `1e-6`。理论最优目标仅用于原算法已有终止判断和实验误差统计，不提供预测方向。费用降低百分比为正表示减少；同时报告共同成功的配对运行结果，避免把失败后更早停止误认为成功加速。应同时查看误差、成功数、UFE 与 LFE；本版不预设一定加速。

`testPredictionCorrection` 检查包含交叉项的曲面恢复、未来样本排除、病态/退化拒绝、0.5/0.5 端点混合、从第 10 代起逐代调度、状态与 RNG/评价计数不变、限幅与边界，以及 SMD1/5/8 固定种子的小规模真实运行。其第一个参数应指向修改前 `BLCMAES.m` 的只读备份目录；该目录仅作为测试基准，实际算法仍只有一个 `BLCMAES.m`。

本次已完成的测试使用 `results_prediction_tests/v1_smoke_retry/reference/` 中保存的修改前快照，可将该目录作为参数重新验证。正式对比结果及解释见 `PREDICTION_CORRECTION_RESULTS.md`。
