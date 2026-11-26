function writePredictionRefinementReport(experiment)
%WRITEPREDICTIONREFINEMENTREPORT Chinese, data-backed offline study report.
file=fullfile(experiment.outputRoot,'prediction_refinement_analysis.txt');
fid=fopen(file,'w','n','UTF-8'); assert(fid>=0); cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
D=experiment.details; A=experiment.analysis; audits=experiment.audit;
fprintf(fid,'BL-CMA-ES：预测点邻域历史知识再利用实验\n\n');
fprintf(fid,'实验模式：%s。八个标签的默认S0/S1对照仅改变历史检索中心；S2默认关闭。\n',strjoin(experiment.config.options.modes,','));
fprintf(fid,'P1原解析二次机制、40个样本、最近10代窗口、欧氏距离、负定Hessian和原边界修复均保留。\n');
fprintf(fid,'完整历史由原始A0逐代记录重建，逐代验证截取的最近10代档案与原日志完全一致。\n');
fprintf(fid,'H2从完整历史重新检索，不限于H1；仍应用与H1完全相同的最近10代筛选条件。\n');
fprintf(fid,'P1未通过原有效性判断的机会保留失败记录，后续模型不启动，也不额外真实评价。\n');
fprintf(fid,'P2保留原解析完整二次模型。G1/G2、R1/R2复用原GPR/RBF及200候选、5起点寻优参数。\n');
fprintf(fid,'Q1/Q2为新增quadprog约束拟合：对原最小化目标F=-UF施加严格凸性。\n');
fprintf(fid,'等价UF Hessian严格负定且为对角阵，模型有5个系数、无交叉项；不等同于原完整二次模型。\n');
fprintf(fid,'G/R/Q的S0与S1共用m_t和H1尺度作输入坐标、共用200个候选位置和原局部区域。\n');
fprintf(fid,'P1/P2保留原解析寻优与边界规则，原默认最大距离为Inf，可能位于G/R/Q的局部球外。\n');
fprintf(fid,'因此同一模型的S0/S1配对是检索中心对照；跨模型结果同时涉及模型/求解规则差异。\n');
fprintf(fid,'S2若启用，只在P1 Mahalanobis邻域与各自原允许区域的交集寻优，不扩大边界或原距离限制。\n');
fprintf(fid,'本实验不注入预测点、不生成随机辅助个体、不改变原CMA-ES更新和终止条件。\n\n');
fprintf(fid,'主运行：%d次，隔离审计通过%d次，复用原先已核验主运行%d次。\n',height(audits), ...
    sum(audits.isolation_verified&audits.main_trajectory_equal&audits.main_rng_equal),sum(audits.reused_main));
fprintf(fid,'全部t>=10可对照状态=%d，原P1有效状态=%d。\n',height(experiment.samples),sum(experiment.samples.P1_valid));
fprintf(fid,'实际新增旁路UFE=%d、LFE=%d；耗时=%.3f秒。\n',sum(D.extra_UFE),sum(D.extra_LFE),sum(audits.diagnostic_seconds));
fprintf(fid,'坐标逐元素完全相同且下层条件一致时复用评价，共%d条；仅接近的点没有删除。\n\n',sum(D.evaluation_reused));
fprintf(fid,'以下有效率分母为原P1有效的全部机会；条件质量指标仅取本模型真实评价成功的记录。\n');
fprintf(fid,'先在独立运行内汇总，再跨运行统计；配对检验的单位是独立运行，不是代数。\n');
fprintf(fid,'预设的同模型S1/S0及重选模型/P1比较在每题内作Holm校正，其他两两比较另作探索性校正。\n\n');
S=A.summary;
for smd=unique(S.SMD)'
    fprintf(fid,'SMD%d\n',smd);
    sample=A.sampleSummary(A.sampleSummary.SMD==smd,:);
    fprintf(fid,'H1/H2平均重合率=%.2f%%；每状态H2平均新样本=%.2f个。\n', ...
        100*sample.overlap_mean_mean,sample.H2_new_samples_mean_mean);
    fprintf(fid,'模型：有效率；优于正常最优(全部机会/有效预测)；平均排名；潜在父代率；下层条件精度达标率\n');
    for model=string(experiment.config.options.models)
        row=S(S.SMD==smd&S.model==model,:);
        fprintf(fid,'%s：%.1f%%；%.1f%% / %.1f%%；%.3f；%.1f%%；%.1f%%\n', ...
            model,100*row.valid_prediction_rate_mean,100*row.effective_best_improvement_rate_mean, ...
            100*row.conditional_best_improvement_rate_mean,row.mean_rank_mean, ...
            100*row.conditional_parent_rate_mean,100*row.lower_precision_rate_mean);
    end
    P=A.pairwise; P=P(P.SMD==smd&P.family=="planned_refinement",:);
    for i=1:height(P)
        fprintf(fid,'%s相对%s：胜/平/负=%d/%d/%d，配对运行%d，UF差值中位数=%+.6g，归一化差值中位数=%+.6g，p=%.5g，Holm p=%.5g。\n', ...
            P.variant_model(i),P.reference_model(i),P.variant_wins(i),P.ties(i),P.reference_wins(i), ...
            P.paired_runs(i),P.median_run_UF_delta(i),P.median_run_normalized_UF_delta(i), ...
            P.UF_delta_p_raw(i),P.UF_delta_p_holm(i));
    end
    C=A.complementarity;
    if ~isempty(C)
        subset=C(C.SMD==smd,:);
        for i=1:height(subset)
            fprintf(fid,'%s/%s：优于P1有效率=%.1f%%，事后最优预测点份额=%.1f%%，P1不胜正常最优时救援次数=%d。\n', ...
                subset.mode(i),subset.model(i),100*subset.effective_better_than_P1_rate_mean(i), ...
                100*subset.fractional_best_rate_mean(i),round(subset.rescue_count_sum(i)));
        end
    end
    fprintf(fid,'\n');
end
fprintf(fid,'解释限制\n');
fprintf(fid,'多模型最佳、至少一个优于P1及潜在父代率均为事后潜力，未执行在线选点或实际父代选择。\n');
fprintf(fid,'方向余弦、距离和归一化距离只记录，没有阈值筛点或改变搜索方向。\n');
fprintf(fid,'零长度方向余弦记为NaN，不强行赋值。归一化距离为Mahalanobis距离除以sigma，不含sqrt(DU)因子。\n');
fprintf(fid,'下层条件误差和双方下层精度均达标子集已保存；部分问题的近似下层解会影响预测质量解释。\n');
fprintf(fid,'最终上下层误差及主UFE/LFE属于同一A0主运行，见main_summary；不存在八个独立算法的最终收敛结果。\n');
fprintf(fid,'本轮结果不能直接证明多预测点协同注入的最终精度或加速，需要另做实际注入实验。\n');
end
