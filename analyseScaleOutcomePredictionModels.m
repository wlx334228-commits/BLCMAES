function result = analyseScaleOutcomePredictionModels(outcomes,config)
%ANALYSESCALEOUTCOMEPREDICTIONMODELS Grouped-CV M0/M1 eta selector.
% M0 has only u=eta-.5 and u^2.  M1 adds the requested d/q interactions.
% All splits keep every eta and repeat from one independent main run together.

assert(istable(outcomes),'outcomes must be a table.');
modelSpecs = { ...
    'M0',false,false; ...
    'M1_d_only',true,false; ...
    'M1_q_only',false,true; ...
    'M1',true,true};
result = emptyResult();
if isempty(outcomes)
    result.parameterTable = statusParameters(modelSpecs, ...
        'no_outcome_rows');
    return;
end
training = outcomes(outcomes.is_training_eta ~= 0 & ...
    isfinite(outcomes.relative_overall_gain) & isfinite(outcomes.d) & ...
    isfinite(outcomes.q) & isfinite(outcomes.eta),:);
runKeys = unique(string(training.run_key),'stable');
if numel(runKeys) < 5 || ~hasFiveRunsPerProblem(training)
    result.parameterTable = statusParameters(modelSpecs, ...
        'insufficient_independent_main_runs_for_grouped_fivefold_cv');
    return;
end
foldMap = assignFiveFolds(training);
folds = unique(foldMap.fold);
parameterRows = repmat(emptyParameterRow(),0,1);
predictionRows = repmat(emptyPredictionRow(),0,1);
selectionRows = repmat(emptySelectionRow(),0,1);
intermediateRows = repmat(emptyPredictionRow(),0,1);

for fold = folds(:)'
    testRunKeys = foldMap.run_key(foldMap.fold == fold);
    trainRows = training(~ismember(string(training.run_key),testRunKeys),:);
    testRows = outcomes(ismember(string(outcomes.run_key),testRunKeys),:);
    if isempty(trainRows) || isempty(testRows), continue; end
    testStateKeys = unique(string(testRows.state_key),'stable');
    fixedWritten = false(numel(testStateKeys),1);
    for specIndex = 1:size(modelSpecs,1)
        spec = struct('name',modelSpecs{specIndex,1}, ...
            'useD',modelSpecs{specIndex,2},'useQ',modelSpecs{specIndex,3});
        model = fitModelWithInternalRidgeSelection(trainRows,spec);
        parameterRows(end+1,1) = parameterRow(model,fold,spec, ...
            numel(unique(string(trainRows.run_key))),height(trainRows)); %#ok<AGROW>
        for stateIndex = 1:numel(testStateKeys)
            stateRows = testRows(string(testRows.state_key) == testStateKeys(stateIndex),:);
            discreteRows = stateRows(stateRows.is_training_eta ~= 0,:);
            discreteRows = sortrows(discreteRows,'eta');
            assert(height(discreteRows) == numel(config.discreteEtas), ...
                'A held-out state/repeat is missing one discrete eta branch.');
            predicted = predictModel(model,discreteRows);
            for rowIndex = 1:height(discreteRows)
                predictionRows(end+1,1) = predictionRow(discreteRows(rowIndex,:), ...
                    fold,spec.name,predicted(rowIndex),'discrete'); %#ok<AGROW>
            end
            [chosenEta,chosenPrediction] = chooseDiscreteEta( ...
                discreteRows.eta,predicted);
            selectedRow = discreteRows(abs(discreteRows.eta-chosenEta) < 10*eps,:);
            oracleRow = discreteRows(discreteRows.relative_overall_gain == ...
                max(discreteRows.relative_overall_gain),:);
            oracleEta = tieBrokenEta(oracleRow.eta);
            chosenRow = selectionRow(selectedRow,fold,spec.name, ...
                chosenEta,chosenPrediction,oracleEta, ...
                max(discreteRows.relative_overall_gain),model);
            chosenRow.selected_matches_oracle = ...
                double(abs(chosenEta-oracleEta) < 10*eps);
            chosenRow.predicted_observed_rank_correlation = ...
                rankCorrelation(predicted,discreteRows.relative_overall_gain);
            selectionRows(end+1,1) = chosenRow; %#ok<AGROW>
            if ~fixedWritten(stateIndex)
                for etaIndex = 1:numel(config.discreteEtas)
                    fixedRow = discreteRows(abs(discreteRows.eta- ...
                        config.discreteEtas(etaIndex)) < 10*eps,:);
                    fixedSelection = selectionRow(fixedRow,fold, ...
                        sprintf('fixed_eta_%03d',round(100*fixedRow.eta)), ...
                        fixedRow.eta,NaN,oracleEta, ...
                        max(discreteRows.relative_overall_gain),[]);
                    fixedSelection.selected_matches_oracle = ...
                        double(abs(fixedRow.eta-oracleEta) < 10*eps);
                    selectionRows(end+1,1) = fixedSelection; %#ok<AGROW>
                end
                oracleSelection = selectionRow(oracleRow(1,:),fold, ...
                    'oracle_discrete',oracleEta,NaN,oracleEta, ...
                    max(discreteRows.relative_overall_gain),[]);
                oracleSelection.selected_matches_oracle = 1;
                selectionRows(end+1,1) = oracleSelection; %#ok<AGROW>
                fixedWritten(stateIndex) = true;
            end
            intermediate = stateRows(stateRows.is_intermediate_eta ~= 0,:);
            intermediate = sortrows(intermediate,'eta');
            for rowIndex = 1:height(intermediate)
                predictedIntermediate = predictModel(model,intermediate(rowIndex,:));
                intermediateRows(end+1,1) = predictionRow( ...
                    intermediate(rowIndex,:),fold,spec.name, ...
                    predictedIntermediate,'intermediate'); %#ok<AGROW>
            end
        end
    end
end

result.parameterTable = struct2table(parameterRows);
result.predictionTable = struct2table(predictionRows);
result.selectionTable = struct2table(selectionRows);
result.intermediateTable = struct2table(intermediateRows);
[result.predictionMetrics,result.predictionMetricSummary] = ...
    summarisePredictionError(result.predictionTable,result.intermediateTable);
[result.runStrategySummary,result.strategySummary] = ...
    summariseStrategyOutcomes(result.selectionTable);
[result.pairedDifferences,result.pairedDifferenceSummary] = ...
    pairedStrategyDifferences(result.runStrategySummary);
end

function result = emptyResult()
result = struct('parameterTable',struct2table(repmat(emptyParameterRow(),0,1)), ...
    'predictionTable',struct2table(repmat(emptyPredictionRow(),0,1)), ...
    'selectionTable',struct2table(repmat(emptySelectionRow(),0,1)), ...
    'intermediateTable',struct2table(repmat(emptyPredictionRow(),0,1)), ...
    'predictionMetrics',struct2table(repmat(emptyMetricRow(),0,1)), ...
    'predictionMetricSummary',struct2table(repmat(emptyMetricSummaryRow(),0,1)), ...
    'runStrategySummary',struct2table(repmat(emptyStrategyRunRow(),0,1)), ...
    'strategySummary',struct2table(repmat(emptyStrategySummaryRow(),0,1)), ...
    'pairedDifferences',struct2table(repmat(emptyPairedDifferenceRow(),0,1)), ...
    'pairedDifferenceSummary',struct2table(repmat(emptyPairedSummaryRow(),0,1)));
end

function values = statusParameters(modelSpecs,status)
rows = repmat(emptyParameterRow(),0,1);
for index = 1:size(modelSpecs,1)
    row = emptyParameterRow();
    row.model = modelSpecs{index,1}; row.status = status;
    rows(end+1,1) = row; %#ok<AGROW>
end
values = struct2table(rows);
end

function tf = hasFiveRunsPerProblem(rows)
tf = true;
for smd = unique(rows.SMD)'
    tf = tf && numel(unique(rows.run_no(rows.SMD == smd))) >= 5;
end
end

function foldMap = assignFiveFolds(rows)
% Stratify the group folds by SMD, then by the deterministic run number.
uniqueRuns = unique(rows(:,{'SMD','run_no','run_key'}),'rows','stable');
fold = NaN(height(uniqueRuns),1);
for smd = unique(uniqueRuns.SMD)'
    indices = find(uniqueRuns.SMD == smd);
    [~,order] = sort(uniqueRuns.run_no(indices));
    indices = indices(order);
    fold(indices) = mod((0:numel(indices)-1)',5)+1;
end
foldMap = table(string(uniqueRuns.run_key),fold, ...
    'VariableNames',{'run_key','fold'});
end

function model = fitModelWithInternalRidgeSelection(rows,spec)
runKeys = unique(string(rows.run_key),'stable');
lambdaGrid = [1e-6,1e-4,1e-2,1e-1,1,10];
if numel(runKeys) < 3
    bestLambda = 1e-2;
else
    innerMap = assignInnerFolds(rows,min(5,numel(runKeys)));
    meanLoss = Inf(numel(lambdaGrid),1);
    for lambdaIndex = 1:numel(lambdaGrid)
        losses = NaN(numel(unique(innerMap.fold)),1);
        for fold = unique(innerMap.fold)'
            heldOut = innerMap.run_key(innerMap.fold == fold);
            train = rows(~ismember(string(rows.run_key),heldOut),:);
            validate = rows(ismember(string(rows.run_key),heldOut),:);
            localModel = fitModelWithLambda(train,spec,lambdaGrid(lambdaIndex));
            prediction = predictModel(localModel,validate);
            losses(fold) = mean((prediction-validate.relative_overall_gain).^2, ...
                'omitnan');
        end
        meanLoss(lambdaIndex) = mean(losses,'omitnan');
    end
    [~,bestIndex] = min(meanLoss);
    bestLambda = lambdaGrid(bestIndex);
end
model = fitModelWithLambda(rows,spec,bestLambda);
model.inner_lambda_grid = lambdaGrid;
end

function foldMap = assignInnerFolds(rows,foldCount)
uniqueRuns = unique(rows(:,{'SMD','run_no','run_key'}),'rows','stable');
fold = NaN(height(uniqueRuns),1);
for smd = unique(uniqueRuns.SMD)'
    indices = find(uniqueRuns.SMD == smd);
    [~,order] = sort(uniqueRuns.run_no(indices)); indices = indices(order);
    fold(indices) = mod((0:numel(indices)-1)',foldCount)+1;
end
foldMap = table(string(uniqueRuns.run_key),fold, ...
    'VariableNames',{'run_key','fold'});
end

function model = fitModelWithLambda(rows,spec,lambda)
model = struct('name',spec.name,'useD',spec.useD,'useQ',spec.useQ, ...
    'lambda',lambda,'d_mean',NaN,'d_std',NaN,'q_mean',NaN,'q_std',NaN, ...
    'beta',NaN(0,1),'status','fitted');
[X,model] = designMatrix(rows,model,true);
y = rows.relative_overall_gain(:);
valid = all(isfinite(X),2) & isfinite(y);
X = X(valid,:); y = y(valid);
if isempty(X)
    model.status = 'no_finite_training_rows'; return;
end
regularizer = lambda*eye(size(X,2));
model.beta = (X'*X+regularizer) \ (X'*y);
if any(~isfinite(model.beta))
    model.status = 'nonfinite_ridge_solution';
end
end

function [X,model] = designMatrix(rows,model,fitNormalizer)
u = rows.eta(:)-0.50;
u2 = u.^2;
% The eta=.50 reference is retained to preserve its state/repeat linkage,
% but all regressors are exactly zero there.  It therefore contributes no
% extra parameter information and is never treated as an independent state.
if fitNormalizer
    dRaw = log(1+max(rows.d(:),0));
    qRaw = log(1+max(rows.q(:),0));
    model.d_mean = mean(dRaw,'omitnan'); model.q_mean = mean(qRaw,'omitnan');
    model.d_std = max(std(dRaw,0,'omitnan'),sqrt(eps));
    model.q_std = max(std(qRaw,0,'omitnan'),sqrt(eps));
else
    dRaw = log(1+max(rows.d(:),0));
    qRaw = log(1+max(rows.q(:),0));
end
dTilde = (dRaw-model.d_mean)/model.d_std;
qTilde = (qRaw-model.q_mean)/model.q_std;
X = [u,u2];
if model.useD, X = [X,dTilde.*u,dTilde.*u2]; end %#ok<AGROW>
if model.useQ, X = [X,qTilde.*u,qTilde.*u2]; end %#ok<AGROW>
end

function prediction = predictModel(model,rows)
if isempty(model.beta) || any(~isfinite(model.beta))
    prediction = NaN(height(rows),1); return;
end
[X,~] = designMatrix(rows,model,false);
prediction = X*model.beta;
end

function [eta,prediction] = chooseDiscreteEta(etas,predictions)
maximum = max(predictions);
ties = etas(abs(predictions-maximum) <= 1e-12*max(1,abs(maximum)));
eta = tieBrokenEta(ties);
prediction = predictions(find(abs(etas-eta) < 10*eps,1,'first'));
end

function eta = tieBrokenEta(etas)
% A tie is not evidence for an intervention. Prefer the reference eta=.50,
% then the numerically smaller eta, so the decision is deterministic.
[~,order] = sortrows([abs(etas(:)-0.50),etas(:)],[1 2]);
eta = etas(order(1));
end

function rho = rankCorrelation(predicted,observed)
if any(~isfinite(predicted)) || any(~isfinite(observed)) || ...
        numel(predicted) < 2
    rho = NaN; return;
end
predictedRank = ordinalRanks(predicted);
observedRank = ordinalRanks(observed);
if std(predictedRank) <= 0 || std(observedRank) <= 0
    rho = NaN;
else
    correlation = corrcoef(predictedRank,observedRank);
    rho = correlation(1,2);
end
end

function ranks = ordinalRanks(values)
[~,order] = sort(values(:),'descend');
ranks = NaN(size(values(:)));
ranks(order) = 1:numel(values);
end

function row = parameterRow(model,fold,modelSpec,runCount,sampleCount)
row = emptyParameterRow();
row.fold = fold; row.model = modelSpec.name; row.status = model.status;
row.training_run_count = runCount; row.training_row_count = sampleCount;
row.lambda = model.lambda; row.d_log_mean = model.d_mean; row.d_log_std = model.d_std;
row.q_log_mean = model.q_mean; row.q_log_std = model.q_std;
if numel(model.beta) >= 2
    row.a1 = model.beta(1); row.a2 = model.beta(2);
end
position = 3;
if modelSpec.useD
    row.b_d_u = model.beta(position); row.c_d_u2 = model.beta(position+1);
    position = position+2;
end
if modelSpec.useQ
    row.b_q_u = model.beta(position); row.c_q_u2 = model.beta(position+1);
end
end

function row = predictionRow(outcome,fold,model,prediction,split)
row = emptyPredictionRow();
row.SMD = outcome.SMD; row.run_no = outcome.run_no; row.run_key = outcome.run_key;
row.state_index = outcome.state_index; row.state_key = outcome.state_key;
row.phase = outcome.phase; row.repeat_no = outcome.repeat_no;
row.fold = fold; row.model = model; row.split = split; row.eta = outcome.eta;
row.d = outcome.d; row.q = outcome.q; row.observed_y = outcome.relative_overall_gain;
row.predicted_y = prediction; row.absolute_error = abs(prediction-row.observed_y);
row.squared_error = (prediction-row.observed_y).^2;
end

function row = selectionRow(outcome,fold,policy,chosenEta,predictedGain, ...
        oracleEta,oracleGain,model)
row = emptySelectionRow();
row.SMD = outcome.SMD; row.run_no = outcome.run_no; row.run_key = outcome.run_key;
row.state_index = outcome.state_index; row.state_key = outcome.state_key;
row.phase = outcome.phase; row.repeat_no = outcome.repeat_no; row.fold = fold;
row.policy = policy; row.selected_eta = chosenEta;
row.predicted_y_at_selected = predictedGain;
row.actual_y_at_selected = outcome.relative_overall_gain;
row.actual_F_end = outcome.F_end; row.reference_F_end_eta050 = ...
    outcome.F_end_reference_eta050;
row.oracle_discrete_eta = oracleEta; row.oracle_discrete_y = oracleGain;
row.selection_regret_to_oracle = oracleGain-outcome.relative_overall_gain;
if ~isempty(model)
    [continuousEta,continuousGain] = maximizeContinuousPrediction(model,outcome);
    row.continuous_eta_prediction = continuousEta;
    row.continuous_predicted_y = continuousGain;
end
end

function [eta,prediction] = maximizeContinuousPrediction(model,outcome)
% For a fixed state the model is a quadratic in eta.  fminbnd is used as a
% bounded one-dimensional optimizer, and endpoints are checked explicitly
% because their maximum is a valid outcome for a concave/convex quadratic.
objective = @(candidate)-predictAtEta(model,outcome,candidate);
candidate = fminbnd(objective,0.20,1.00);
candidates = [0.20;candidate;1.00];
values = NaN(size(candidates));
for index = 1:numel(candidates)
    values(index) = predictAtEta(model,outcome,candidates(index));
end
[prediction,index] = max(values);
eta = candidates(index);
end

function value = predictAtEta(model,outcome,eta)
synthetic = outcome;
synthetic.eta = eta;
value = predictModel(model,synthetic);
end

function [runMetrics,summary] = summarisePredictionError(discrete,intermediate)
allRows = [discrete;intermediate];
rows = repmat(emptyMetricRow(),0,1);
if ~isempty(allRows)
    keys = unique(strcat(string(allRows.model),'|',string(allRows.split),'|', ...
        string(allRows.run_key)),'stable');
    for index = 1:numel(keys)
        part = allRows(strcat(string(allRows.model),'|',string(allRows.split),'|', ...
            string(allRows.run_key)) == keys(index),:);
        row = emptyMetricRow();
        row.model = part.model{1}; row.split = part.split{1};
        row.SMD = part.SMD(1); row.run_no = part.run_no(1); row.run_key = part.run_key{1};
        row.independent_state_repeats = numel(unique(string(part.state_key)));
        row.MAE = mean(part.absolute_error,'omitnan');
        row.RMSE = sqrt(mean(part.squared_error,'omitnan'));
        rows(end+1,1) = row; %#ok<AGROW>
    end
end
runMetrics = struct2table(rows);
summaryRows = repmat(emptyMetricSummaryRow(),0,1);
if ~isempty(runMetrics)
    keys = unique(strcat(string(runMetrics.model),'|',string(runMetrics.split), ...
        '|',string(runMetrics.SMD)), ...
        'stable');
    for index = 1:numel(keys)
        part = runMetrics(strcat(string(runMetrics.model),'|',string(runMetrics.split), ...
            '|',string(runMetrics.SMD)) ...
            == keys(index),:);
        row = emptyMetricSummaryRow(); row.model = part.model{1}; row.split = part.split{1};
        row.SMD = part.SMD(1);
        row.independent_runs = height(part);
        row.MAE_mean = mean(part.MAE,'omitnan'); row.MAE_std = std(part.MAE,0,'omitnan');
        row.RMSE_mean = mean(part.RMSE,'omitnan'); row.RMSE_std = std(part.RMSE,0,'omitnan');
        summaryRows(end+1,1) = row; %#ok<AGROW>
    end
end
summary = struct2table(summaryRows);
end

function [runSummary,summary] = summariseStrategyOutcomes(selections)
rows = repmat(emptyStrategyRunRow(),0,1);
if ~isempty(selections)
    keys = unique(strcat(string(selections.policy),'|',string(selections.run_key)), ...
        'stable');
    for index = 1:numel(keys)
        part = selections(strcat(string(selections.policy),'|',string(selections.run_key)) ...
            == keys(index),:);
        row = emptyStrategyRunRow(); row.policy = part.policy{1};
        row.SMD = part.SMD(1); row.run_no = part.run_no(1); row.run_key = part.run_key{1};
        row.state_repeat_count = numel(unique(string(part.state_key)));
        row.actual_y_mean = mean(part.actual_y_at_selected,'omitnan');
        row.actual_y_std = std(part.actual_y_at_selected,0,'omitnan');
        row.regret_mean = mean(part.selection_regret_to_oracle,'omitnan');
        row.oracle_match_rate = mean(part.selected_matches_oracle,'omitnan');
        row.rank_correlation_mean = mean( ...
            part.predicted_observed_rank_correlation,'omitnan');
        rows(end+1,1) = row; %#ok<AGROW>
    end
end
runSummary = struct2table(rows);
summaryRows = repmat(emptyStrategySummaryRow(),0,1);
if ~isempty(runSummary)
    keys = unique(strcat(string(runSummary.policy),'|',string(runSummary.SMD)), ...
        'stable');
    for index = 1:numel(keys)
        part = runSummary(strcat(string(runSummary.policy),'|', ...
            string(runSummary.SMD)) == keys(index),:);
        row = emptyStrategySummaryRow(); row.policy = char(keys(index));
        row.policy = part.policy{1}; row.SMD = part.SMD(1);
        row.independent_runs = height(part);
        row.actual_y_mean = mean(part.actual_y_mean,'omitnan');
        row.actual_y_std = std(part.actual_y_mean,0,'omitnan');
        row.actual_y_median = median(part.actual_y_mean,'omitnan');
        row.regret_mean = mean(part.regret_mean,'omitnan');
        row.oracle_match_rate = mean(part.oracle_match_rate,'omitnan');
        row.rank_correlation_mean = mean(part.rank_correlation_mean,'omitnan');
        summaryRows(end+1,1) = row; %#ok<AGROW>
    end
end
summary = struct2table(summaryRows);
end

function [differences,summary] = pairedStrategyDifferences(runSummary)
% Rows remain paired at the independent main-run level.  The output is
% intentionally raw paired evidence; downstream Wilcoxon analysis must use
% these rows, never the many within-run state/repeat records.
rows = repmat(emptyPairedDifferenceRow(),0,1);
comparisons = { ...
    'M1','M0'; 'M1','M1_d_only'; 'M1','M1_q_only'; ...
    'M1','fixed_eta_025'; 'M1','fixed_eta_050'; ...
    'M1','fixed_eta_075'; 'M1','fixed_eta_100'; ...
    'M0','fixed_eta_025'; 'M0','fixed_eta_050'; ...
    'M0','fixed_eta_075'; 'M0','fixed_eta_100'};
for smd = unique(runSummary.SMD)'
    for runNo = unique(runSummary.run_no(runSummary.SMD == smd))'
        part = runSummary(runSummary.SMD == smd & runSummary.run_no == runNo,:);
        for pairIndex = 1:size(comparisons,1)
            left = part(strcmp(part.policy,comparisons{pairIndex,1}),:);
            right = part(strcmp(part.policy,comparisons{pairIndex,2}),:);
            if height(left) ~= 1 || height(right) ~= 1, continue; end
            row = emptyPairedDifferenceRow(); row.SMD = smd; row.run_no = runNo;
            row.run_key = left.run_key{1}; row.policy_A = comparisons{pairIndex,1};
            row.policy_B = comparisons{pairIndex,2};
            row.delta_actual_y_A_minus_B = left.actual_y_mean-right.actual_y_mean;
            row.delta_regret_A_minus_B = left.regret_mean-right.regret_mean;
            rows(end+1,1) = row; %#ok<AGROW>
        end
    end
end
differences = struct2table(rows);
summaryRows = repmat(emptyPairedSummaryRow(),0,1);
if ~isempty(differences)
    keys = unique(strcat(string(differences.SMD),'|', ...
        string(differences.policy_A),'|',string(differences.policy_B)),'stable');
    for index = 1:numel(keys)
        part = differences(strcat(string(differences.SMD),'|', ...
            string(differences.policy_A),'|',string(differences.policy_B)) ...
            == keys(index),:);
        row = emptyPairedSummaryRow(); row.SMD = part.SMD(1);
        row.policy_A = part.policy_A{1}; row.policy_B = part.policy_B{1};
        row.independent_runs = height(part);
        row.delta_mean = mean(part.delta_actual_y_A_minus_B,'omitnan');
        row.delta_median = median(part.delta_actual_y_A_minus_B,'omitnan');
        row.delta_std = std(part.delta_actual_y_A_minus_B,0,'omitnan');
        row.A_better_count = sum(part.delta_actual_y_A_minus_B > 0);
        row.B_better_count = sum(part.delta_actual_y_A_minus_B < 0);
        summaryRows(end+1,1) = row; %#ok<AGROW>
    end
end
summary = struct2table(summaryRows);
end

function row = emptyParameterRow()
row = struct('fold',NaN,'model','','status','','training_run_count',NaN, ...
    'training_row_count',NaN,'lambda',NaN,'d_log_mean',NaN,'d_log_std',NaN, ...
    'q_log_mean',NaN,'q_log_std',NaN,'a1',NaN,'a2',NaN, ...
    'b_d_u',NaN,'c_d_u2',NaN,'b_q_u',NaN,'c_q_u2',NaN);
end

function row = emptyPredictionRow()
row = struct('SMD',NaN,'run_no',NaN,'run_key','','state_index',NaN, ...
    'state_key','','phase','','repeat_no',NaN,'fold',NaN,'model','', ...
    'split','','eta',NaN,'d',NaN,'q',NaN,'observed_y',NaN, ...
    'predicted_y',NaN,'absolute_error',NaN,'squared_error',NaN);
end

function row = emptySelectionRow()
row = struct('SMD',NaN,'run_no',NaN,'run_key','','state_index',NaN, ...
    'state_key','','phase','','repeat_no',NaN,'fold',NaN,'policy','', ...
    'selected_eta',NaN,'predicted_y_at_selected',NaN, ...
    'actual_y_at_selected',NaN,'actual_F_end',NaN, ...
    'reference_F_end_eta050',NaN,'oracle_discrete_eta',NaN, ...
    'oracle_discrete_y',NaN,'selection_regret_to_oracle',NaN, ...
    'selected_matches_oracle',NaN,'predicted_observed_rank_correlation',NaN, ...
    'continuous_eta_prediction',NaN,'continuous_predicted_y',NaN);
end

function row = emptyMetricRow()
row = struct('model','','split','','SMD',NaN,'run_no',NaN,'run_key','', ...
    'independent_state_repeats',NaN,'MAE',NaN,'RMSE',NaN);
end

function row = emptyMetricSummaryRow()
row = struct('model','','split','','SMD',NaN,'independent_runs',NaN, ...
    'MAE_mean',NaN,'MAE_std',NaN,'RMSE_mean',NaN,'RMSE_std',NaN);
end

function row = emptyStrategyRunRow()
row = struct('policy','','SMD',NaN,'run_no',NaN,'run_key','', ...
    'state_repeat_count',NaN,'actual_y_mean',NaN,'actual_y_std',NaN, ...
    'regret_mean',NaN,'oracle_match_rate',NaN, ...
    'rank_correlation_mean',NaN);
end

function row = emptyStrategySummaryRow()
row = struct('policy','','SMD',NaN,'independent_runs',NaN,'actual_y_mean',NaN, ...
    'actual_y_std',NaN,'actual_y_median',NaN,'regret_mean',NaN, ...
    'oracle_match_rate',NaN,'rank_correlation_mean',NaN);
end

function row = emptyPairedDifferenceRow()
row = struct('SMD',NaN,'run_no',NaN,'run_key','','policy_A','', ...
    'policy_B','','delta_actual_y_A_minus_B',NaN, ...
    'delta_regret_A_minus_B',NaN);
end

function row = emptyPairedSummaryRow()
row = struct('SMD',NaN,'policy_A','','policy_B','', ...
    'independent_runs',NaN,'delta_mean',NaN,'delta_median',NaN, ...
    'delta_std',NaN,'A_better_count',NaN,'B_better_count',NaN);
end
