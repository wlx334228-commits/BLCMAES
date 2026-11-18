function comparisonTable = compareContinuousScaleBranchesToA1(branchSummary)
%COMPARECONTINUOUSSCALEBRANCHESTOA1 Pair every eta branch with A1.
% A1 contains the same direct-prediction mechanism but no auxiliary
% candidates.  This table keeps its whole-branch outcomes separate from the
% auxiliary-only candidate metrics, which are undefined for A1 by design.

rows = repmat(emptyRow(),0,1);
predictionRows = branchSummary(~strcmp(branchSummary.eta_label,'A1'),:);
for index = 1:height(predictionRows)
    branch = predictionRows(index,:);
    pairedA1 = branchSummary(strcmp(branchSummary.eta_label,'A1') & ...
        branchSummary.SMD == branch.SMD & ...
        branchSummary.run_no == branch.run_no & ...
        branchSummary.state_index == branch.state_index & ...
        branchSummary.repeat_no == branch.repeat_no & ...
        branchSummary.branch_seed == branch.branch_seed,:);
    if height(pairedA1) ~= 1
        error('BLCMAES:ContinuousScalePairing', ...
            'Expected exactly one matched A1 row for SMD%d run%d state%d.', ...
            branch.SMD,branch.run_no,branch.state_index);
    end
    a1 = pairedA1(1,:);
    row = emptyRow();
    row.SMD = branch.SMD; row.run_no = branch.run_no;
    row.state_index = branch.state_index;
    row.state_id = tableText(branch.state_id);
    row.phase = tableText(branch.phase); row.eta = branch.eta;
    row.eta_label = tableText(branch.eta_label);
    row.repeat_no = branch.repeat_no;
    row.branch_seed = branch.branch_seed;
    row.branch_overall_history_improvement = branch.overall_history_improvement;
    row.A1_overall_history_improvement = a1.overall_history_improvement;
    row.improvement_minus_A1 = branch.overall_history_improvement - ...
        a1.overall_history_improvement;
    row.branch_final_upper_error = branch.final_upper_error;
    row.A1_final_upper_error = a1.final_upper_error;
    row.upper_error_minus_A1 = branch.final_upper_error - a1.final_upper_error;
    row.branch_final_lower_error = branch.final_lower_error;
    row.A1_final_lower_error = a1.final_lower_error;
    row.lower_error_minus_A1 = branch.final_lower_error - a1.final_lower_error;
    row.branch_UFE_delta = branch.branch_UFE_delta;
    row.A1_UFE_delta = a1.branch_UFE_delta;
    row.UFE_delta_minus_A1 = branch.branch_UFE_delta - a1.branch_UFE_delta;
    row.branch_LFE_delta = branch.branch_LFE_delta;
    row.A1_LFE_delta = a1.branch_LFE_delta;
    row.LFE_delta_minus_A1 = branch.branch_LFE_delta - a1.branch_LFE_delta;
    rows(end+1,1) = row; %#ok<AGROW>
end
comparisonTable = struct2table(rows);
if ~isempty(comparisonTable)
    comparisonTable = sortrows(comparisonTable, ...
        {'SMD','run_no','state_index','eta','repeat_no'});
end
end

function text = tableText(value)
if iscell(value), value = value{1}; end
if isstring(value), value = value(1); end
if ischar(value) && size(value,1) > 1, value = value(1,:); end
text = char(value);
end

function row = emptyRow()
row = struct('SMD',NaN,'run_no',NaN,'state_index',NaN,'state_id','', ...
    'phase','','eta',NaN,'eta_label','','repeat_no',NaN,'branch_seed',NaN, ...
    'branch_overall_history_improvement',NaN, ...
    'A1_overall_history_improvement',NaN,'improvement_minus_A1',NaN, ...
    'branch_final_upper_error',NaN,'A1_final_upper_error',NaN, ...
    'upper_error_minus_A1',NaN, ...
    'branch_final_lower_error',NaN,'A1_final_lower_error',NaN, ...
    'lower_error_minus_A1',NaN, ...
    'branch_UFE_delta',NaN,'A1_UFE_delta',NaN,'UFE_delta_minus_A1',NaN, ...
    'branch_LFE_delta',NaN,'A1_LFE_delta',NaN,'LFE_delta_minus_A1',NaN);
end
