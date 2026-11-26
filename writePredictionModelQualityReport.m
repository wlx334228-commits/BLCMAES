function writePredictionModelQualityReport(experiment)
%WRITEPREDICTIONMODELQUALITYREPORT Data-backed Chinese diagnostic report.
file=fullfile(experiment.outputRoot,'prediction_model_analysis.txt');
fid=fopen(file,'w','n','UTF-8'); assert(fid>=0); cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'BL-CMA-ES 四种代理模型预测点质量对比\n\n');
fprintf(fid,'一、实验方法\n');
fprintf(fid,'A0为参考主运行，正常终止；预测点不注入，不生成辅助个体，不更新主档案。\n');
fprintf(fid,'在t>=10且主算法确实评价了t+1代的每个状态，离线生成并真实评价预测点。\n');
fprintf(fid,'QP、完整三次CP、r^3核加线性尾项RBF、fitrgp Matérn 5/2 GPR，共用最近10代中\n');
fprintf(fid,'距更新后均值最近的40条真实记录，归一化中心/尺度及200个候选位置相同。\n');
fprintf(fid,'共同区域为原始变量边界与 ||z||<=1 的交集，半径来自样本范围，与eta无关。\n');
fprintf(fid,'每个模型取预测UF最优5个起点，用相同fmincon设置最小化负预测UF。\n');
fprintf(fid,'GPR在标准化UF上固定Sigma=0.01作正则化，拟合核参数；只最大化均值，标准差只记录。\n');
fprintf(fid,'QP_analytic额外保留原负定Hessian解析驻点方法及原边界修复，不属于四模型共同寻优比较。\n');
fprintf(fid,'该解析参照可能位于共同局部区域之外；此情况在逐次记录中有单独标记。\n');
fprintf(fid,'各模型下层评价从同一CMA状态初始化，使用同一下层随机种子、算法和预算。\n');
fprintf(fid,'正常候选比较值取初次真实评价后、额外精炼前的UF，避免不同精炼次数影响比较。\n');
fprintf(fid,'重复点只记录、不强行删点；共同候选、模型和CMA状态保存在逐状态MAT中。\n');
fprintf(fid,'UF=-原上层目标，LF=-原下层目标，UF越大越好。\n\n');
A=experiment.audit; D=experiment.details;
fprintf(fid,'二、规模及正确性\n');
fprintf(fid,'独立主运行=%d，配对搜索状态=%d，预测记录=%d（含解析参照）。\n',height(A),height(D)/5,height(D));
fprintf(fid,'主轨迹、最终结果和最终随机数状态完全相同：%d/%d次独立运行。\n', ...
    sum(A.isolation_verified&A.main_trajectory_equal&A.main_rng_equal),height(A));
fprintf(fid,'额外诊断UFE=%d，LFE=%d，均未计入A0主评价次数。\n', ...
    sum(experiment.details.extra_UFE),sum(experiment.details.extra_LFE));
fprintf(fid,'GPR有训练优化警告但模型数值可用的记录=%d，详见gpr_warnings。\n\n', ...
    sum(D.model=="GPR"&strlength(D.gpr_warnings)>1));
fprintf(fid,'三、统计口径\n');
fprintf(fid,'先在每次独立运行内汇总，再跨运行比较。有效改善率分母为全部预测机会，\n');
fprintf(fid,'条件改善率分母为有效预测且真实评价成功的次数。前中后期仅在事后按UFE进度划分。\n');
fprintf(fid,'配对Wilcoxon使用各运行的配对UF差值中位数；四模型6个比较在每题内作Holm校正，\n');
fprintf(fid,'解析参照比较另成一族。原始p值同时保存，少于3次配对运行不报告检验p值。\n\n');
S=experiment.analysis.summary; ids=["QP","CP","RBF","GPR","QP_analytic"];
for smd=unique(S.SMD)'
    fprintf(fid,'SMD%d\n',smd);
    fprintf(fid,'模型 | 有效率 | 优于正常最优(全部机会/有效预测) | 平均排名 | 潜在父代率 | 下层条件精度达标率\n');
    for id=ids
        sub=S(S.SMD==smd&S.model==id,:);
        fprintf(fid,'%s | %.1f%% | %.1f%% / %.1f%% | %.3f | %.1f%% | %.1f%%\n', ...
            id,100*sub.valid_prediction_rate_mean,100*sub.effective_best_improvement_rate_mean, ...
            100*sub.conditional_best_improvement_rate_mean,sub.mean_rank_mean, ...
            100*sub.conditional_parent_rate_mean,100*sub.lower_precision_rate_mean);
    end
    P=experiment.analysis.pairwise;
    sub=P(P.SMD==smd&P.reference_model=="QP"&P.family=="fair_four_models",:);
    for i=1:height(sub)
        fprintf(fid,'%s对QP：胜/平/负=%d/%d/%d，配对运行=%d，运行级UF改善中位数=%+.6g，p=%.5g，Holm p=%.5g，QP无效时优于正常最优%d次。\n', ...
            sub.variant_model(i),sub.variant_wins(i),sub.ties(i),sub.reference_wins(i), ...
            sub.paired_runs(i),sub.median_run_UF_delta(i),sub.UF_delta_p_raw(i), ...
            sub.UF_delta_p_holm(i),sub.variant_good_when_reference_invalid(i));
    end
    fprintf(fid,'\n');
end
fprintf(fid,'四、相对原解析模型的互补性\n');
P=experiment.analysis.pairwise;
for i=1:height(P)
    if P.variant_model(i)=="QP_analytic" && P.reference_model(i)~="QP"
        fprintf(fid,'SMD%d %s：解析QP无效时自身有效%d次，其中优于正常最优%d次；解析QP排名落后于父代门槛时优于正常最优%d次。\n', ...
            P.SMD(i),P.reference_model(i),P.reference_only_valid(i), ...
            P.reference_good_when_variant_invalid(i),P.reference_good_when_variant_poor(i));
    end
end
fprintf(fid,'\n五、解释边界\n');
comparisons=P(P.reference_model=="QP"&P.family=="fair_four_models",:);
fprintf(fid,'在相同寻优规则下与QP比较的%d组检验中，新模型显著更好%d组，QP显著更好%d组，\n', ...
    height(comparisons),sum(comparisons.variant_UF_advantage),sum(comparisons.reference_UF_advantage));
fprintf(fid,'无显著差异%d组（按独立运行UF差值中位数的配对检验及Holm校正）。\n', ...
    sum(~comparisons.variant_UF_advantage&~comparisons.reference_UF_advantage));
for i=find(comparisons.variant_UF_advantage)'
    fprintf(fid,'其中SMD%d的%s相对QP显著更好：Holm p=%.5g。\n', ...
        comparisons.SMD(i),comparisons.variant_model(i),comparisons.UF_delta_p_holm(i));
end
fprintf(fid,'应区分模型类别变化与解析/数值寻优规则变化，不能将统一寻优QP与原解析QP差异归于模型类别。\n');
fprintf(fid,'下层条件精度达标率偏低的问题，真实UF比较会受到近似下层解影响；应检查逐次的\n');
fprintf(fid,'lower_conditional_error和配对表内双方下层精度均达标的子集。条件最优值未用于预测和在线决定。\n');
fprintf(fid,'本实验验证预测候选质量，不能直接推出完整优化算法的最终精度或评价次数优势。\n');
fprintf(fid,'协同注入是否有效仍需结合位置互补性、可靠下层评价及独立完整优化实验验证。\n');
end
