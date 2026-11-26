function analysis=summarisePredictionRefinement(details,samples,audit,options,outputRoot)
%SUMMARISEPREDICTIONREFINEMENT Run-level inference; all geometry is descriptive.
eligible=details(details.eligible_opportunity,:);
assert(~isempty(eligible),'No valid original P1 opportunities; inspect failure records.');
% Adapt the reusable quality aggregator without changing the delivered raw
% field meanings: its internal reference field is rebound to P1 explicitly.
eligible.improvement_vs_QP=eligible.improvement_vs_P1;
pairs={'P1','P2';'G1','G2';'Q1','Q2';'R1','R2';'P1','G2';'P1','Q2';'P1','R2'};
if ismember('S2',options.modes)
    pairs=[pairs;{'P2','P2_S2';'G2','G2_S2';'Q2','Q2_S2';'R2','R2_S2'; ...
        'P1','P2_S2';'P1','G2_S2';'P1','Q2_S2';'P1','R2_S2'}];
end
settings=struct('modelIds',{options.models},'prefix','prediction_refinement', ...
    'referenceId','P1','plannedPairs',{pairs});
analysis=summarisePredictionModelQuality(eligible,outputRoot,settings);
[geometry,complement]=predictionRefinementRelations(details);
writetable(geometry,fullfile(outputRoot,'prediction_refinement_geometry.csv'));
if isempty(complement)
    complement=table(strings(0,1),'VariableNames',{'model'});
    compRun=complement; compSummary=complement;
else
    [compRun,compSummary]=summariseComplement(complement);
end
writetable(complement,fullfile(outputRoot,'prediction_refinement_complementarity_details.csv'));
writetable(compRun,fullfile(outputRoot,'prediction_refinement_complementarity_run.csv'));
writetable(compSummary,fullfile(outputRoot,'prediction_refinement_complementarity.csv'));
sampleRun=sampleSummary(samples);
writetable(sampleRun,fullfile(outputRoot,'prediction_refinement_sample_run_summary.csv'));
sampleOverall=aggregate(sampleRun,{'SMD'});
writetable(sampleOverall,fullfile(outputRoot,'prediction_refinement_sample_summary.csv'));
mainSummary=mainOutcomes(audit);
writetable(mainSummary,fullfile(outputRoot,'prediction_refinement_main_summary.csv'));
analysis.geometry=geometry; analysis.complementarityDetails=complement;
analysis.complementarityRun=compRun; analysis.complementarity=compSummary;
analysis.sampleRunSummary=sampleRun; analysis.sampleSummary=sampleOverall;
analysis.mainSummary=mainSummary;
plotResults(analysis,outputRoot,options.models);
end

function [runSummary,summary]=summariseComplement(C)
C=C(C.eligible_opportunity,:);
keys=unique(C(:,{'SMD','run_no','seed','mode','model'}),'rows'); rows=struct([]);
for i=1:height(keys)
    sub=C(C.SMD==keys.SMD(i)&C.run_no==keys.run_no(i)&C.mode==keys.mode(i)&C.model==keys.model(i),:);
    row=struct('SMD',keys.SMD(i),'run_no',keys.run_no(i),'seed',keys.seed(i), ...
        'mode',keys.mode(i),'model',keys.model(i),'opportunities',height(sub), ...
        'valid_count',sum(sub.candidate_valid),'best_count',sum(sub.is_best_refined), ...
        'unique_best_count',sum(sub.is_unique_best_refined), ...
        'fractional_best_count',sum(sub.best_fractional_credit), ...
        'fractional_best_rate',mean(sub.best_fractional_credit), ...
        'better_than_P1_count',sum(sub.better_than_P1), ...
        'effective_better_than_P1_rate',mean(sub.better_than_P1), ...
        'P1_poor_opportunities',sum(sub.P1_not_better_than_normal_best), ...
        'rescue_count',sum(sub.rescues_normal_best), ...
        'effective_rescue_rate',mean(sub.rescues_normal_best), ...
        'conditional_rescue_rate',conditional(sub.rescues_normal_best,sub.P1_not_better_than_normal_best), ...
        'any_refined_better_than_P1_rate',mean(sub.any_refined_better_than_P1), ...
        'any_rescues_normal_best_rate',mean(sub.any_rescues_normal_best), ...
        'all_refined_valid_rate',mean(sub.all_refined_valid), ...
        'posthoc_best_UF_improvement_median',finiteMedian(sub.best_refined_UF_minus_P1));
    rows=append(rows,row);
end
runSummary=struct2table(rows); summary=aggregate(runSummary,{'SMD','mode','model'});
end

function result=sampleSummary(S)
keys=unique(S(:,{'SMD','run_no','seed'}),'rows'); rows=struct([]);
for i=1:height(keys)
    run=S(S.SMD==keys.SMD(i)&S.run_no==keys.run_no(i),:); valid=run(run.P1_valid,:);
    row=struct('SMD',keys.SMD(i),'run_no',keys.run_no(i),'seed',keys.seed(i), ...
        'all_opportunities',height(run),'eligible_P1_opportunities',height(valid), ...
        'P1_valid_rate',mean(run.P1_valid),'overlap_mean',finiteMean(valid.overlap_rate), ...
        'overlap_median',finiteMedian(valid.overlap_rate), ...
        'H2_new_samples_mean',finiteMean(valid.H2_new_sample_count), ...
        'H2_min_distance_median',finiteMedian(valid.H2_distance_min), ...
        'H2_median_distance_median',finiteMedian(valid.H2_distance_median), ...
        'H2_max_distance_median',finiteMedian(valid.H2_distance_max), ...
        'H2_design_condition_median',finiteMedian(valid.H2_quadratic_design_condition), ...
        'H2_centre_inside_hull_rate',finiteMean(valid.H2_centre_inside_hull), ...
        'H1_centre_inside_hull_rate',finiteMean(valid.H1_centre_inside_hull), ...
        'total_diagnostic_seconds',sum(run.state_total_seconds));
    rows=append(rows,row);
end
result=struct2table(rows);
end

function result=aggregate(T,keys)
groups=unique(T(:,keys),'rows'); rows=struct([]);
numericNames=T.Properties.VariableNames(varfun(@isnumeric,T,'OutputFormat','uniform'));
numericNames=numericNames(~ismember(numericNames,[keys,{'run_no','seed'}]));
for i=1:height(groups)
    select=true(height(T),1); row=struct();
    for name=keys
        select=select&T.(name{1})==groups.(name{1})(i); row.(name{1})=groups.(name{1})(i);
    end
    sub=T(select,:); row.independent_runs=height(sub);
    for name=numericNames
        values=sub.(name{1}); finite=values(isfinite(values));
        row.([name{1} '_sum'])=sum(finite);
        row.([name{1} '_mean'])=finiteMean(finite);
        row.([name{1} '_median'])=finiteMedian(finite);
        if isempty(finite), row.([name{1} '_IQR'])=NaN; else, row.([name{1} '_IQR'])=iqr(finite); end
    end
    rows=append(rows,row);
end
result=struct2table(rows);
end

function output=mainOutcomes(T)
rows=struct([]);
for smd=unique(T.SMD)'
    sub=T(T.SMD==smd,:); row=struct('SMD',smd,'variant',"A0",'runs',height(sub));
    for name={'final_upper_error','final_lower_error','main_UFE','main_LFE'}
        values=sub.(name{1}); row.([name{1} '_median'])=median(values);
        row.([name{1} '_IQR'])=iqr(values);
    end
    rows=append(rows,row);
end
output=struct2table(rows);
end

function plotResults(A,outputRoot,ids)
folder=fullfile(outputRoot,'figures'); if ~isfolder(folder), mkdir(folder); end
for metric={'effective_best_improvement_rate_mean','conditional_parent_rate_mean'}
    f=figure('Visible','off','Position',[50 50 1400 700],'Color','white'); tiledlayout(2,4,'TileSpacing','compact');
    for smd=unique(A.summary.SMD)'
        nexttile; values=NaN(size(ids));
        for i=1:numel(ids)
            index=A.summary.SMD==smd&A.summary.model==string(ids{i});
            values(i)=A.summary.(metric{1})(index);
        end
        bar(values); xticks(1:numel(ids)); xticklabels(ids); xtickangle(45); ylim([0 1]);
        title(sprintf('SMD%d',smd)); grid on;
    end
    exportgraphics(f,fullfile(folder,[metric{1} '.png']),'Resolution',150);
    savefig(f,fullfile(folder,[metric{1} '.fig'])); close(f);
end
end

function value=conditional(x,valid)
if any(valid), value=mean(x(valid)); else, value=NaN; end
end
function value=finiteMean(x)
x=x(isfinite(x)); if isempty(x), value=NaN; else, value=mean(x); end
end
function value=finiteMedian(x)
x=x(isfinite(x)); if isempty(x), value=NaN; else, value=median(x); end
end
function rows=append(rows,row)
if isempty(rows), rows=row; else, rows(end+1,1)=row; end
end
