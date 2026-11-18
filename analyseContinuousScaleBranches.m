function [analysisTable,regressionTable] = analyseContinuousScaleBranches( ...
    branchSummary,snapshotTable)
%ANALYSECONTINUOUSSCALEBRANCHES Aggregate only at independent-run level.
% Candidate/generation rows are intentionally not treated as independent
% observations.  A branch state is first summarized within its run; any
% regression is withheld unless enough distinct main runs exist.

metricNames = {'auxiliary_improvement_over_start', ...
    'auxiliary_vs_normal_mean_difference','overall_history_improvement'};
metricLabels = {'auxiliary_direct_benefit','auxiliary_vs_normal', ...
    'overall_search_benefit'};
analysisRows = repmat(emptyAnalysisRow(),0,1);
predictionRows = branchSummary(~strcmp(branchSummary.eta_label,'A1'),:);
for smdNo = unique(predictionRows.SMD)'
    smdRows = predictionRows(predictionRows.SMD == smdNo,:);
    dCut = median(smdRows.d,'omitnan');
    qCut = median(smdRows.q,'omitnan');
    for eta = unique(smdRows.eta)'
        etaRows = smdRows(smdRows.eta == eta,:);
        groups = {true(height(etaRows),1),'all'; ...
            etaRows.d <= dCut,'d_low'; etaRows.d > dCut,'d_high'; ...
            etaRows.q <= qCut,'q_low'; etaRows.q > qCut,'q_high'};
        for groupIndex = 1:size(groups,1)
            subset = etaRows(groups{groupIndex,1},:);
            if isempty(subset), continue; end
            % First average repeated matched branch seeds within a state;
            % then average state values within each independent main run.
            stateValues = aggregateByState(subset);
            runValues = aggregateByRun(stateValues);
            row = emptyAnalysisRow();
            row.SMD = smdNo; row.eta = eta; row.group = groups{groupIndex,2};
            row.independent_runs = height(runValues);
            row.state_count = height(stateValues);
            row.d_median = median(stateValues.d,'omitnan');
            row.q_median = median(stateValues.q,'omitnan');
            row.auxiliary_improvement_median = ...
                median(runValues.auxiliary_improvement_over_start,'omitnan');
            row.auxiliary_vs_normal_median = ...
                median(runValues.auxiliary_vs_normal_mean_difference,'omitnan');
            row.overall_improvement_median = ...
                median(runValues.overall_history_improvement,'omitnan');
            row.auxiliary_parent_rate_median = ...
                median(runValues.auxiliary_parent_selection_rate,'omitnan');
            row.auxiliary_winning_generation_rate_median = ...
                median(runValues.auxiliary_winning_generation_rate,'omitnan');
            row.UFE_delta_median = median(runValues.branch_UFE_delta,'omitnan');
            row.LFE_delta_median = median(runValues.branch_LFE_delta,'omitnan');
            analysisRows(end+1,1) = row; %#ok<AGROW>
        end
    end
end
analysisTable = struct2table(analysisRows);
if ~isempty(analysisTable)
    analysisTable = sortrows(analysisTable,{'SMD','eta','group'});
end
regressionTable = stateScaleRegression(predictionRows,metricNames,metricLabels);
% This input is retained so an invocation with an empty branch table still
% makes it clear that state data was expected, rather than silently passing.
assert(istable(snapshotTable),'snapshotTable must be a table.');
end

function values = aggregateByState(rows)
keys = unique(strcat(string(rows.SMD),'_',string(rows.run_no),'_', ...
    string(rows.state_index),'_',string(rows.eta)),'stable');
values = table();
for index = 1:numel(keys)
    mask = strcat(string(rows.SMD),'_',string(rows.run_no),'_', ...
        string(rows.state_index),'_',string(rows.eta)) == keys(index);
    part = rows(mask,:);
    values = [values; table(part.SMD(1),part.run_no(1),part.state_index(1), ...
        part.eta(1),part.phase(1),mean(part.d,'omitnan'),mean(part.q,'omitnan'), ...
        mean(part.auxiliary_improvement_over_start,'omitnan'), ...
        mean(part.auxiliary_vs_normal_mean_difference,'omitnan'), ...
        mean(part.overall_history_improvement,'omitnan'), ...
        mean(part.auxiliary_parent_selection_rate,'omitnan'), ...
        mean(part.auxiliary_winning_generation_rate,'omitnan'), ...
        mean(part.branch_UFE_delta,'omitnan'),mean(part.branch_LFE_delta,'omitnan'), ...
        'VariableNames',{'SMD','run_no','state_index','eta','phase','d','q', ...
        'auxiliary_improvement_over_start', ...
        'auxiliary_vs_normal_mean_difference','overall_history_improvement', ...
        'auxiliary_parent_selection_rate','auxiliary_winning_generation_rate', ...
        'branch_UFE_delta','branch_LFE_delta'})]; %#ok<AGROW>
end
end

function values = aggregateByRun(stateValues)
keys = unique(strcat(string(stateValues.SMD),'_',string(stateValues.run_no)),'stable');
values = table();
metricNames = stateValues.Properties.VariableNames(6:end);
for index = 1:numel(keys)
    mask = strcat(string(stateValues.SMD),'_',string(stateValues.run_no)) == keys(index);
    part = stateValues(mask,:);
    row = table(part.SMD(1),part.run_no(1),'VariableNames',{'SMD','run_no'});
    for metricIndex = 1:numel(metricNames)
        row.(metricNames{metricIndex}) = mean(part.(metricNames{metricIndex}),'omitnan');
    end
    values = [values;row]; %#ok<AGROW>
end
end

function tableData = stateScaleRegression(rows,metricNames,metricLabels)
output = repmat(emptyRegressionRow(),0,1);
if isempty(rows)
    tableData = struct2table(output); return;
end
stateValues = aggregateByState(rows);
% The requested two models are retained verbatim.  The first three rows
% are feature ablations: they answer whether d or q has predictive value
% beyond eta before a joint state model is considered.
modelSpecs = { ...
    'scale_only',false,false,false; ...
    'd_only',true,false,false; ...
    'q_only',false,true,false; ...
    'model_A_d_and_q',true,true,false; ...
    'model_B_dq_interactions',true,true,true};
for metricIndex = 1:numel(metricNames)
    y = stateValues.(metricNames{metricIndex});
    for modelIndex = 1:size(modelSpecs,1)
        row = emptyRegressionRow();
        row.response = metricLabels{metricIndex};
        row.model = modelSpecs{modelIndex,1};
        runKeys = strcat(string(stateValues.SMD),'_',string(stateValues.run_no));
        row.independent_runs = numel(unique(runKeys));
        row.state_count = height(stateValues);
        if row.independent_runs < 8
            row.status = 'insufficient_independent_runs_for_interaction_validation';
            output(end+1,1) = row; %#ok<AGROW>
            continue;
        end
        % The requested core terms are first; problem and phase indicators
        % are nuisance controls, not a claim that either causes scale value.
        X = [ones(height(stateValues),1),stateValues.eta,stateValues.eta.^2];
        if modelSpecs{modelIndex,2}, X = [X,stateValues.d]; end %#ok<AGROW>
        if modelSpecs{modelIndex,3}, X = [X,stateValues.q]; end %#ok<AGROW>
        X = [X,problemControls(stateValues.SMD),phaseControls(stateValues.phase)];
        if modelSpecs{modelIndex,4}
            X = [X,stateValues.d.*stateValues.eta,stateValues.q.*stateValues.eta];
        end
        valid = all(isfinite(X),2) & isfinite(y);
        X = X(valid,:); yValid = y(valid); 
        runNo = runKeys(valid);
        if rank(X) < size(X,2) || numel(yValid) <= size(X,2)
            row.status = 'rank_deficient_or_insufficient_state_rows';
            output(end+1,1) = row; %#ok<AGROW>
            continue;
        end
        beta = X \ yValid;
        prediction = X*beta;
        row.in_sample_R2 = rSquared(yValid,prediction);
        row.parameter_count = size(X,2);
        [row.cv_R2,row.cv_RMSE] = leaveOneRunOut(X,yValid,runNo);
        row.status = 'fitted_leave_one_run_out';
        output(end+1,1) = row; %#ok<AGROW>
    end
end
tableData = struct2table(output);
end

function [R2,RMSE] = leaveOneRunOut(X,y,runNo)
predictions = NaN(size(y));
for heldOut = unique(runNo)'
    train = runNo ~= heldOut; test = runNo == heldOut;
    if sum(train) <= size(X,2) || rank(X(train,:)) < size(X,2), continue; end
    beta = X(train,:) \ y(train);
    predictions(test) = X(test,:)*beta;
end
valid = isfinite(predictions);
if ~any(valid), R2 = NaN; RMSE = NaN; return; end
R2 = rSquared(y(valid),predictions(valid));
RMSE = sqrt(mean((y(valid)-predictions(valid)).^2));
end

function values = problemControls(problem)
levels = unique(problem,'stable');
values = zeros(numel(problem),max(0,numel(levels)-1));
for index = 2:numel(levels)
    values(:,index-1) = problem == levels(index);
end
end

function values = phaseControls(phase)
phase = string(phase);
levels = unique(phase,'stable');
values = zeros(numel(phase),max(0,numel(levels)-1));
for index = 2:numel(levels)
    values(:,index-1) = phase == levels(index);
end
end

function value = rSquared(actual,prediction)
denominator = sum((actual-mean(actual)).^2);
if denominator <= 0, value = NaN; else, value = 1-sum((actual-prediction).^2)/denominator; end
end

function row = emptyAnalysisRow()
row = struct('SMD',NaN,'eta',NaN,'group','','independent_runs',NaN, ...
    'state_count',NaN,'d_median',NaN,'q_median',NaN, ...
    'auxiliary_improvement_median',NaN,'auxiliary_vs_normal_median',NaN, ...
    'overall_improvement_median',NaN,'auxiliary_parent_rate_median',NaN, ...
    'auxiliary_winning_generation_rate_median',NaN, ...
    'UFE_delta_median',NaN,'LFE_delta_median',NaN);
end

function row = emptyRegressionRow()
row = struct('response','','model','','independent_runs',NaN, ...
    'state_count',NaN,'parameter_count',NaN,'in_sample_R2',NaN, ...
    'cv_R2',NaN,'cv_RMSE',NaN,'status','');
end
