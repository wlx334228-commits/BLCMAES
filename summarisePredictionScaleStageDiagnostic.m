function [runSummary,stageSummary,pairedTests] = ...
    summarisePredictionScaleStageDiagnostic(candidates)
%SUMMARISEPREDICTIONSCALESTAGEDIAGNOSTIC Aggregate by independent run.
% Candidate-level observations are first reduced within (run,stage,eta), so
% the paired tests never incorrectly treat two samples from one state as two
% independent experimental repetitions.

assert(istable(candidates) && ~isempty(candidates), ...
    'Candidate diagnostic table must be nonempty.');
required = {'SMD','run_no','seed','stage','eta','UF','diagnostic_UFE', ...
    'diagnostic_LFE','better_than_direct','better_than_normal_pair_rate', ...
    'delta_UF_to_direct','virtual_enters_parent','better_than_history_best', ...
    'actual_distance_to_prediction','relative_cma_distance', ...
    'standardized_distance','boundary_repair_count', ...
    'diagnostic_invalid_evaluation'};
assert(all(ismember(required,candidates.Properties.VariableNames)), ...
    'Candidate diagnostic table is missing required columns.');

runNames = {'SMD','run_no','seed','stage','eta','candidate_count', ...
    'Q_rank','Q_pred','delta_UF_median','delta_UF_mean', ...
    'virtual_parent_rate','history_improvement_rate', ...
    'actual_distance_median','relative_cma_distance_median', ...
    'standardized_distance_median','boundary_repairs', ...
    'diagnostic_invalid_evaluations','diagnostic_UFE','diagnostic_LFE'};
runRows = cell(0,numel(runNames));
for smdNo = unique(candidates.SMD)'
    smdRows = candidates(candidates.SMD == smdNo,:);
    for stageCell = unique(smdRows.stage)'
        stage = tableText(stageCell,1);
        stageRows = smdRows(strcmp(smdRows.stage,stage),:);
        for runNo = unique(stageRows.run_no)'
            runRowsOnly = stageRows(stageRows.run_no == runNo,:);
            for eta = unique(runRowsOnly.eta)'
                subset = runRowsOnly(abs(runRowsOnly.eta-eta) < 1e-12,:);
                runRows(end+1,:) = {smdNo,runNo,subset.seed(1),stage,eta, ...
                    height(subset),mean(subset.better_than_normal_pair_rate), ...
                    mean(double(subset.better_than_direct)), ...
                    median(subset.delta_UF_to_direct), ...
                    mean(subset.delta_UF_to_direct), ...
                    mean(double(subset.virtual_enters_parent)), ...
                    mean(double(subset.better_than_history_best)), ...
                    median(subset.actual_distance_to_prediction), ...
                    median(subset.relative_cma_distance), ...
                    median(subset.standardized_distance), ...
                    sum(subset.boundary_repair_count), ...
                    sum(double(subset.diagnostic_invalid_evaluation)), ...
                    sum(subset.diagnostic_UFE),sum(subset.diagnostic_LFE)}; %#ok<AGROW>
            end
        end
    end
end
runSummary = cell2table(runRows,'VariableNames',runNames);

summaryNames = {'SMD','stage','eta','independent_runs','candidate_count', ...
    'Q_rank_median','Q_rank_IQR','Q_pred_median','Q_pred_IQR', ...
    'delta_UF_median','delta_UF_IQR','virtual_parent_rate_median', ...
    'virtual_parent_rate_IQR','history_improvement_rate_median', ...
    'history_improvement_rate_IQR','actual_distance_median', ...
    'actual_distance_IQR','relative_cma_distance_median', ...
    'relative_cma_distance_IQR','standardized_distance_median', ...
    'standardized_distance_IQR','boundary_repairs_total', ...
    'diagnostic_invalid_evaluations_total','diagnostic_UFE_total', ...
    'diagnostic_LFE_total','diagnostic_UFE_per_run_median', ...
    'diagnostic_LFE_per_run_median'};
summaryRows = cell(0,numel(summaryNames));
for smdNo = unique(runSummary.SMD)'
    smdRows = runSummary(runSummary.SMD == smdNo,:);
    for stageCell = unique(smdRows.stage)'
        stage = tableText(stageCell,1);
        stageRows = smdRows(strcmp(smdRows.stage,stage),:);
        for eta = unique(stageRows.eta)'
            subset = stageRows(abs(stageRows.eta-eta) < 1e-12,:);
            [Qrank,QrankIQR] = medianIQR(subset.Q_rank);
            [Qpred,QpredIQR] = medianIQR(subset.Q_pred);
            [delta,deltaIQR] = medianIQR(subset.delta_UF_median);
            [parent,parentIQR] = medianIQR(subset.virtual_parent_rate);
            [history,historyIQR] = medianIQR(subset.history_improvement_rate);
            [distance,distanceIQR] = medianIQR(subset.actual_distance_median);
            [relative,relativeIQR] = medianIQR(subset.relative_cma_distance_median);
            [standardised,standardisedIQR] = medianIQR(subset.standardized_distance_median);
            summaryRows(end+1,:) = {smdNo,stage,eta,height(subset), ...
                sum(subset.candidate_count),Qrank,QrankIQR,Qpred,QpredIQR, ...
                delta,deltaIQR,parent,parentIQR,history,historyIQR, ...
                distance,distanceIQR,relative,relativeIQR, ...
                standardised,standardisedIQR,sum(subset.boundary_repairs), ...
                sum(subset.diagnostic_invalid_evaluations), ...
                sum(subset.diagnostic_UFE),sum(subset.diagnostic_LFE), ...
                median(subset.diagnostic_UFE),median(subset.diagnostic_LFE)}; %#ok<AGROW>
        end
    end
end
stageSummary = cell2table(summaryRows,'VariableNames',summaryNames);
pairedTests = makePairedTests(runSummary);
end

function tests = makePairedTests(runSummary)
names = {'SMD','stage','comparison','reference_eta','variant_eta', ...
    'paired_runs','Q_rank_p','Q_pred_p','delta_UF_p', ...
    'Q_rank_p_holm','Q_pred_p_holm','delta_UF_p_holm', ...
    'Q_rank_reference_median','Q_rank_variant_median', ...
    'Q_pred_reference_median','Q_pred_variant_median', ...
    'delta_UF_reference_median','delta_UF_variant_median'};
rows = cell(0,numel(names));
for smdNo = unique(runSummary.SMD)'
    smdRows = runSummary(runSummary.SMD == smdNo,:);
    for stageCell = unique(smdRows.stage)'
        stage = tableText(stageCell,1);
        stageRows = smdRows(strcmp(smdRows.stage,stage),:);
        if ~any(abs(stageRows.eta-0.5) < 1e-12), continue; end
        for eta = unique(stageRows.eta)'
            if abs(eta-0.5) < 1e-12, continue; end
            reference = stageRows(abs(stageRows.eta-0.5) < 1e-12,:);
            variant = stageRows(abs(stageRows.eta-eta) < 1e-12,:);
            commonRuns = intersect(reference.run_no,variant.run_no);
            reference = sortrows(reference(ismember(reference.run_no,commonRuns),:),'run_no');
            variant = sortrows(variant(ismember(variant.run_no,commonRuns),:),'run_no');
            if numel(commonRuns) < 2, continue; end
            rankTest = pairedWilcoxonSignedRank(variant.Q_rank,reference.Q_rank);
            predTest = pairedWilcoxonSignedRank(variant.Q_pred,reference.Q_pred);
            deltaTest = pairedWilcoxonSignedRank(variant.delta_UF_median, ...
                reference.delta_UF_median);
            rows(end+1,:) = {smdNo,stage,sprintf('eta_%03d_vs_eta_050', ...
                round(100*eta)),0.5,eta,numel(commonRuns),rankTest.p, ...
                predTest.p,deltaTest.p,NaN,NaN,NaN,median(reference.Q_rank), ...
                median(variant.Q_rank),median(reference.Q_pred), ...
                median(variant.Q_pred),median(reference.delta_UF_median), ...
                median(variant.delta_UF_median)}; %#ok<AGROW>
        end
    end
end
tests = cell2table(rows,'VariableNames',names);
if isempty(tests), return; end
for smdNo = unique(tests.SMD)'
    for stageCell = unique(tests(tests.SMD == smdNo,:).stage)'
        stage = tableText(stageCell,1);
        index = tests.SMD == smdNo & strcmp(tests.stage,stage);
        tests.Q_rank_p_holm(index) = holmAdjust(tests.Q_rank_p(index));
        tests.Q_pred_p_holm(index) = holmAdjust(tests.Q_pred_p(index));
        tests.delta_UF_p_holm(index) = holmAdjust(tests.delta_UF_p(index));
    end
end
end

function adjusted = holmAdjust(values)
adjusted = NaN(size(values));
valid = find(isfinite(values));
if isempty(valid), return; end
[ordered,order] = sort(values(valid));
adjustedOrdered = NaN(size(ordered)); previous = 0;
for index = 1:numel(ordered)
    previous = max(previous,min(1,(numel(ordered)-index+1)*ordered(index)));
    adjustedOrdered(index) = previous;
end
adjusted(valid(order)) = adjustedOrdered;
end

function [middle,spread] = medianIQR(values)
values = sort(values(isfinite(values)));
if isempty(values), middle = NaN; spread = NaN; return; end
middle = median(values);
if numel(values) < 2, spread = NaN; return; end
spread = percentile(values,75)-percentile(values,25);
end

function value = percentile(values,percent)
position = 1+(numel(values)-1)*percent/100;
lower = floor(position); upper = ceil(position);
if lower == upper
    value = values(lower);
else
    value = values(lower)+(position-lower)*(values(upper)-values(lower));
end
end

function text = tableText(values,index)
if iscell(values), text = values{index}; else, text = char(values(index)); end
end
