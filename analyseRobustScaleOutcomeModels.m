function result = analyseRobustScaleOutcomeModels(outcomes,snapshots)
%ANALYSEROBUSTSCALEOUTCOMEMODELS Robust, per-problem M0/M1 reanalysis.
%
% The original experiment is not modified.  The primary response is the
% aligned raw change in true upper fitness, DeltaF = F_end(eta)-F_end(.50).
% Three secondary responses use a causal, prefix-median sF floor.  They are
% sensitivity analyses only.  Repeated matched branches are collapsed to one
% state-by-eta response before fitting, so they never inflate sample size.

requiredOutcome = {'SMD','run_no','run_key','state_index','phase','eta', ...
    'is_training_eta','is_intermediate_eta','is_reference_eta','d','q','sF', ...
    'F_end','F_end_reference_eta050','relative_overall_gain', ...
    'common_UFE_checkpoint','completed_generations'};
requiredSnapshot = {'SMD','run_no','state_index','generation','sF'};
assert(istable(outcomes) && all(ismember(requiredOutcome, ...
    outcomes.Properties.VariableNames)), ...
    'The outcome table does not have the expected scale-branch columns.');
assert(istable(snapshots) && all(ismember(requiredSnapshot, ...
    snapshots.Properties.VariableNames)), ...
    'The state snapshot table does not have the expected columns.');

alpha = [0.01 0.05 0.10];
stateScaleTable = prefixScaleTable(snapshots,alpha);
outcomes = attachRobustResponses(outcomes,stateScaleTable,alpha);
auditTable = makeAuditTable(outcomes);
collapsed = collapseMatchedRepeats(outcomes,alpha);

targetNames = [{'raw_delta_F'},arrayfun(@(a) alphaName(a),alpha, ...
    'UniformOutput',false)];
modelSpecs = { ...
    'M0',false,false; ...
    'M1_d_only',true,false; ...
    'M1_q_only',false,true; ...
    'M1',true,true};

parameterRows = repmat(emptyParameterRow(),0,1);
predictionRows = repmat(emptyPredictionRow(),0,1);
intermediateRows = repmat(emptyPredictionRow(),0,1);
selectionRows = repmat(emptySelectionRow(),0,1);

for targetIndex = 1:numel(targetNames)
    target = targetNames{targetIndex};
    for smd = unique(collapsed.SMD)'
        smdRows = collapsed(collapsed.SMD == smd,:);
        training = smdRows(smdRows.is_training_eta ~= 0 & ...
            isfinite(smdRows.(target)),:);
        uniqueRuns = unique(training.run_no);
        if numel(uniqueRuns) < 5
            for specIndex = 1:size(modelSpecs,1)
                parameterRows(end+1,1) = statusParameterRow(target,smd, ...
                    modelSpecs{specIndex,1}, ...
                    'insufficient_independent_runs_for_fivefold_cv'); %#ok<AGROW>
            end
            continue;
        end
        foldMap = assignRunFolds(uniqueRuns,5);
        for fold = 1:5
            testRuns = foldMap.run_no(foldMap.fold == fold);
            trainRows = training(~ismember(training.run_no,testRuns),:);
            testRows = smdRows(ismember(smdRows.run_no,testRuns),:);
            for specIndex = 1:size(modelSpecs,1)
                spec = struct('name',modelSpecs{specIndex,1}, ...
                    'useD',modelSpecs{specIndex,2}, ...
                    'useQ',modelSpecs{specIndex,3});
                model = fitRobustModelWithSelection(trainRows,target,spec);
                trainRunCount = numel(unique(trainRows.run_no));
                trainStateCount = height(unique(trainRows(:, ...
                    {'run_no','state_index'}),'rows'));
                parameterRows(end+1,1) = parameterRow(model,target,smd,fold, ...
                    spec,trainRunCount,trainStateCount); %#ok<AGROW>

                testStates = unique(testRows(:,{'run_no','state_index'}),'rows','stable');
                for stateNumber = 1:height(testStates)
                    stateRows = testRows(testRows.run_no == testStates.run_no(stateNumber) & ...
                        testRows.state_index == testStates.state_index(stateNumber),:);
                    discrete = sortrows(stateRows(stateRows.is_training_eta ~= 0,:), ...
                        'eta');
                    assert(height(discrete) == 4, ...
                        'Each collapsed state must contain eta=.25,.5,.75,1.');
                    predicted = predictRobustModel(model,discrete);
                    for rowIndex = 1:height(discrete)
                        predictionRows(end+1,1) = predictionRow(discrete(rowIndex,:), ...
                            target,spec.name,fold,'discrete',predicted(rowIndex)); %#ok<AGROW>
                    end
                    selectionRows = [selectionRows;selectionRowsForState( ...
                        discrete,predicted,target,spec.name,fold,true)]; %#ok<AGROW>

                    intermediate = sortrows(stateRows( ...
                        stateRows.is_intermediate_eta ~= 0,:),'eta');
                    for rowIndex = 1:height(intermediate)
                        value = predictRobustModel(model,intermediate(rowIndex,:));
                        intermediateRows(end+1,1) = predictionRow( ...
                            intermediate(rowIndex,:),target,spec.name,fold, ...
                            'intermediate',value); %#ok<AGROW>
                    end
                end
            end
            % Fixed policies and the raw oracle do not depend on a fitted
            % model.  Write them once per state and target/fold.
            testStates = unique(testRows(:,{'run_no','state_index'}),'rows','stable');
            for stateNumber = 1:height(testStates)
                stateRows = testRows(testRows.run_no == testStates.run_no(stateNumber) & ...
                    testRows.state_index == testStates.state_index(stateNumber),:);
                discrete = sortrows(stateRows(stateRows.is_training_eta ~= 0,:), ...
                    'eta');
                selectionRows = [selectionRows;selectionRowsForState( ...
                    discrete,NaN(height(discrete),1),target,'fixed',fold,false)]; %#ok<AGROW>
            end
        end
    end
end

result = struct();
result.auditTable = auditTable;
result.stateScaleTable = stateScaleTable;
result.collapsedOutcomes = collapsed;
result.parameterTable = struct2table(parameterRows);
result.predictionTable = struct2table(predictionRows);
result.intermediateTable = struct2table(intermediateRows);
result.selectionTable = struct2table(selectionRows);
[result.metricTable,result.metricSummary] = predictionMetrics( ...
    result.predictionTable,result.intermediateTable);
[result.runStrategySummary,result.strategySummary] = strategySummaries( ...
    result.selectionTable);
[result.pairedDifferences,result.pairedTests] = pairedStrategyTests( ...
    result.runStrategySummary);
end

function map = prefixScaleTable(snapshots,alpha)
rows = repmat(emptyStateScaleRow(),0,1);
for smd = unique(snapshots.SMD)'
    for runNo = unique(snapshots.run_no(snapshots.SMD == smd))'
        part = snapshots(snapshots.SMD == smd & snapshots.run_no == runNo,:);
        part = sortrows(part,{'generation','state_index'});
        finiteSF = part.sF(isfinite(part.sF) & part.sF > 0);
        assert(~isempty(finiteSF), ...
            'A run with valid snapshots has no positive finite sF.');
        for rowIndex = 1:height(part)
            prior = part.sF(1:rowIndex);
            prior = prior(isfinite(prior) & prior > 0);
            row = emptyStateScaleRow();
            row.SMD = smd; row.run_no = runNo;
            row.state_index = part.state_index(rowIndex);
            row.generation = part.generation(rowIndex);
            row.sF = part.sF(rowIndex);
            row.prefix_median_sF = median(prior);
            row.floor_a001 = alpha(1)*row.prefix_median_sF;
            row.floor_a005 = alpha(2)*row.prefix_median_sF;
            row.floor_a010 = alpha(3)*row.prefix_median_sF;
            rows(end+1,1) = row; %#ok<AGROW>
        end
    end
end
map = sortrows(struct2table(rows),{'SMD','run_no','state_index'});
end

function outcomes = attachRobustResponses(outcomes,map,alpha)
outcomes.raw_delta_F = outcomes.F_end-outcomes.F_end_reference_eta050;
keyOutcome = strcat(string(outcomes.SMD),'|',string(outcomes.run_no),'|', ...
    string(outcomes.state_index));
keyMap = strcat(string(map.SMD),'|',string(map.run_no),'|', ...
    string(map.state_index));
[present,location] = ismember(keyOutcome,keyMap);
assert(all(present),'Some branch outcomes have no matching state snapshot.');
outcomes.prefix_median_sF = map.prefix_median_sF(location);
for alphaIndex = 1:numel(alpha)
    name = alphaName(alpha(alphaIndex));
    floorName = floorNameForAlpha(alpha(alphaIndex));
    denominator = max(outcomes.sF,map.(floorName)(location));
    denominator = max(denominator,eps(max(1,max(abs(denominator)))));
    outcomes.(name) = outcomes.raw_delta_F./denominator;
end
end

function audit = makeAuditTable(outcomes)
rows = repmat(emptyAuditRow(),0,1);
for smd = unique(outcomes.SMD)'
    part = outcomes(outcomes.SMD == smd,:);
    row = emptyAuditRow(); row.SMD = smd; row.branch_rows = height(part);
    row.independent_runs = numel(unique(part.run_no));
    row.matched_state_repeats = height(unique(part(:,{'run_no','state_index','repeat_no'}),'rows'));
    row.min_sF = min(part.sF); row.sF_le_1e_6 = sum(part.sF <= 1e-6);
    row.max_abs_original_relative_y = max(abs(part.relative_overall_gain));
    row.max_abs_raw_delta_F = max(abs(part.raw_delta_F));
    row.eta050_max_abs_raw_delta = max(abs(part.raw_delta_F(part.is_reference_eta ~= 0)));
    row.common_checkpoint_groups = 0;
    keys = unique(part(:,{'run_no','state_index','repeat_no'}),'rows','stable');
    for keyIndex = 1:height(keys)
        match = part(part.run_no == keys.run_no(keyIndex) & ...
            part.state_index == keys.state_index(keyIndex) & ...
            part.repeat_no == keys.repeat_no(keyIndex),:);
        row.common_checkpoint_groups = row.common_checkpoint_groups + ...
            double(numel(unique(match.common_UFE_checkpoint)) == 1 && ...
            height(match) == 7);
    end
    rows(end+1,1) = row; %#ok<AGROW>
end
audit = struct2table(rows);
end

function collapsed = collapseMatchedRepeats(outcomes,alpha)
keys = unique(outcomes(:,{'SMD','run_no','run_key','state_index','phase','eta', ...
    'is_training_eta','is_intermediate_eta','is_reference_eta','d','q','sF', ...
    'prefix_median_sF'}),'rows','stable');
rows = repmat(emptyCollapsedRow(),0,1);
for keyIndex = 1:height(keys)
    key = keys(keyIndex,:);
    subset = outcomes(outcomes.SMD == key.SMD & outcomes.run_no == key.run_no & ...
        outcomes.state_index == key.state_index & abs(outcomes.eta-key.eta) < 1e-12,:);
    row = emptyCollapsedRow();
    row.SMD = key.SMD; row.run_no = key.run_no; row.run_key = key.run_key{1};
    row.state_index = key.state_index; row.phase = key.phase{1}; row.eta = key.eta;
    row.is_training_eta = key.is_training_eta;
    row.is_intermediate_eta = key.is_intermediate_eta;
    row.is_reference_eta = key.is_reference_eta;
    row.d = key.d; row.q = key.q; row.sF = key.sF;
    row.prefix_median_sF = key.prefix_median_sF;
    row.repeat_count = height(subset);
    row.raw_delta_F = mean(subset.raw_delta_F,'omitnan');
    row.raw_delta_F_median = median(subset.raw_delta_F,'omitnan');
    row.raw_delta_F_std = std(subset.raw_delta_F,0,'omitnan');
    row.F_end = mean(subset.F_end,'omitnan');
    row.F_end_reference_eta050 = mean(subset.F_end_reference_eta050,'omitnan');
    for alphaIndex = 1:numel(alpha)
        name = alphaName(alpha(alphaIndex));
        row.(name) = mean(subset.(name),'omitnan');
    end
    rows(end+1,1) = row; %#ok<AGROW>
end
collapsed = sortrows(struct2table(rows),{'SMD','run_no','state_index','eta'});
end

function foldMap = assignRunFolds(runNumbers,foldCount)
runNumbers = sort(runNumbers(:));
foldMap = table(runNumbers,mod((0:numel(runNumbers)-1)',foldCount)+1, ...
    'VariableNames',{'run_no','fold'});
end

function model = fitRobustModelWithSelection(rows,target,spec)
lambdaGrid = [1e-6 1e-4 1e-2 1e-1 1 10];
runNumbers = unique(rows.run_no);
innerCount = min(5,numel(runNumbers));
inner = assignRunFolds(runNumbers,innerCount);
loss = Inf(numel(lambdaGrid),1);
for lambdaIndex = 1:numel(lambdaGrid)
    foldLoss = NaN(innerCount,1);
    for fold = 1:innerCount
        heldOut = inner.run_no(inner.fold == fold);
        train = rows(~ismember(rows.run_no,heldOut),:);
        validate = rows(ismember(rows.run_no,heldOut),:);
        local = fitRobustModel(train,target,spec,lambdaGrid(lambdaIndex));
        prediction = predictRobustModel(local,validate);
        foldLoss(fold) = mean(huberLoss(prediction-validate.(target), ...
            local.huber_delta),'omitnan');
    end
    loss(lambdaIndex) = mean(foldLoss,'omitnan');
end
[~,best] = min(loss);
model = fitRobustModel(rows,target,spec,lambdaGrid(best));
model.inner_loss = loss;
end

function model = fitRobustModel(rows,target,spec,lambda)
model = struct('name',spec.name,'useD',spec.useD,'useQ',spec.useQ, ...
    'lambda',lambda,'d_mean',NaN,'d_std',NaN,'q_mean',NaN,'q_std',NaN, ...
    'beta',NaN(0,1),'huber_delta',NaN,'status','fitted');
[X,model] = robustDesignMatrix(rows,model,true);
y = rows.(target);
valid = all(isfinite(X),2) & isfinite(y);
X = X(valid,:); y = y(valid);
if isempty(y), model.status = 'no_finite_training_rows'; return; end
beta = (X'*X+lambda*eye(size(X,2))) \ (X'*y);
initialResidual = y-X*beta;
scale = robustScale(initialResidual);
model.huber_delta = 1.345*scale;
for iteration = 1:40
    residual = y-X*beta;
    weight = min(1,model.huber_delta./max(abs(residual),realmin));
    weightedX = X.*sqrt(weight);
    weightedY = y.*sqrt(weight);
    next = (weightedX'*weightedX+lambda*eye(size(X,2))) \ ...
        (weightedX'*weightedY);
    if norm(next-beta) <= 1e-9*(1+norm(beta))
        beta = next; break;
    end
    beta = next;
end
model.beta = beta;
if any(~isfinite(beta)), model.status = 'nonfinite_huber_ridge_solution'; end
end

function scale = robustScale(values)
values = values(isfinite(values));
if isempty(values), scale = 1; return; end
centre = median(values);
scale = 1.4826*median(abs(values-centre));
fallback = max(1e-12,0.01*median(abs(values)));
scale = max(scale,fallback);
end

function values = huberLoss(residual,delta)
if ~isfinite(delta) || delta <= 0, delta = robustScale(residual); end
absolute = abs(residual);
values = 0.5*residual.^2;
index = absolute > delta;
values(index) = delta*(absolute(index)-0.5*delta);
end

function [X,model] = robustDesignMatrix(rows,model,fitNormalizer)
u = rows.eta(:)-0.50; u2 = u.^2;
dRaw = log(1+max(rows.d(:),0));
qRaw = log(1+max(rows.q(:),0));
if fitNormalizer
    model.d_mean = mean(dRaw,'omitnan'); model.q_mean = mean(qRaw,'omitnan');
    model.d_std = max(std(dRaw,0,'omitnan'),sqrt(eps));
    model.q_std = max(std(qRaw,0,'omitnan'),sqrt(eps));
end
dTilde = (dRaw-model.d_mean)/model.d_std;
qTilde = (qRaw-model.q_mean)/model.q_std;
X = [u,u2];
if model.useD, X = [X,dTilde.*u,dTilde.*u2]; end %#ok<AGROW>
if model.useQ, X = [X,qTilde.*u,qTilde.*u2]; end %#ok<AGROW>
end

function prediction = predictRobustModel(model,rows)
if isempty(model.beta) || any(~isfinite(model.beta))
    prediction = NaN(height(rows),1); return;
end
[X,~] = robustDesignMatrix(rows,model,false);
prediction = X*model.beta;
end

function rows = selectionRowsForState(discrete,predicted,target,policy,fold,modelPolicy)
oracleValue = max(discrete.raw_delta_F);
oracleRows = discrete(abs(discrete.raw_delta_F-oracleValue) <= 1e-14* ...
    max(1,abs(oracleValue)),:);
oracleEta = tieBrokenEta(oracleRows.eta);
if modelPolicy
    [eta,prediction] = chooseEta(discrete.eta,predicted);
    chosen = discrete(abs(discrete.eta-eta) < 1e-12,:);
    rows = makeSelectionRow(chosen,target,policy,fold,prediction,oracleEta,oracleValue);
else
    rows = repmat(emptySelectionRow(),0,1);
    fixed = [0.25 0.50 0.75 1.00];
    for eta = fixed
        chosen = discrete(abs(discrete.eta-eta) < 1e-12,:);
        rows(end+1,1) = makeSelectionRow(chosen,target, ...
            sprintf('fixed_eta_%03d',round(100*eta)),fold,NaN,oracleEta,oracleValue); %#ok<AGROW>
    end
    chosen = oracleRows(abs(oracleRows.eta-oracleEta) < 1e-12,:);
    rows(end+1,1) = makeSelectionRow(chosen,target,'oracle_discrete',fold, ...
        NaN,oracleEta,oracleValue); %#ok<AGROW>
end
end

function row = makeSelectionRow(chosen,target,policy,fold,predicted,oracleEta,oracleValue)
row = emptySelectionRow();
row.target = target; row.SMD = chosen.SMD; row.run_no = chosen.run_no;
row.run_key = chosen.run_key{1}; row.state_index = chosen.state_index;
row.phase = chosen.phase{1}; row.fold = fold; row.policy = policy;
row.selected_eta = chosen.eta; row.predicted_response = predicted;
row.actual_response = chosen.(target); row.actual_raw_delta_F = chosen.raw_delta_F;
row.oracle_eta = oracleEta; row.oracle_raw_delta_F = oracleValue;
row.raw_regret_to_oracle = oracleValue-chosen.raw_delta_F;
row.selected_matches_oracle = double(abs(chosen.eta-oracleEta) < 1e-12);
end

function [eta,prediction] = chooseEta(etas,predictions)
maximum = max(predictions);
ties = etas(abs(predictions-maximum) <= 1e-12*max(1,abs(maximum)));
eta = tieBrokenEta(ties);
prediction = predictions(find(abs(etas-eta) < 1e-12,1,'first'));
end

function eta = tieBrokenEta(etas)
[~,order] = sortrows([abs(etas(:)-0.5),etas(:)],[1 2]);
eta = etas(order(1));
end

function row = predictionRow(observed,target,model,fold,split,prediction)
row = emptyPredictionRow(); row.target = target; row.SMD = observed.SMD;
row.run_no = observed.run_no; row.run_key = observed.run_key{1};
row.state_index = observed.state_index; row.phase = observed.phase{1};
row.eta = observed.eta; row.model = model; row.fold = fold; row.split = split;
row.observed_response = observed.(target); row.predicted_response = prediction;
row.raw_delta_F = observed.raw_delta_F;
row.absolute_error = abs(prediction-observed.(target));
row.squared_error = (prediction-observed.(target)).^2;
end

function [metrics,summary] = predictionMetrics(discrete,intermediate)
allRows = [discrete;intermediate]; rows = repmat(emptyMetricRow(),0,1);
keys = unique(allRows(:,{'target','model','split','SMD','run_no','run_key'}), ...
    'rows','stable');
for index = 1:height(keys)
    key = keys(index,:);
    part = allRows(strcmp(allRows.target,key.target{1}) & ...
        strcmp(allRows.model,key.model{1}) & strcmp(allRows.split,key.split{1}) & ...
        allRows.SMD == key.SMD & allRows.run_no == key.run_no,:);
    row = emptyMetricRow(); row.target = key.target{1}; row.model = key.model{1};
    row.split = key.split{1}; row.SMD = key.SMD; row.run_no = key.run_no;
    row.run_key = key.run_key{1}; row.state_count = height(part);
    row.MAE = mean(part.absolute_error,'omitnan');
    row.RMSE = sqrt(mean(part.squared_error,'omitnan'));
    rows(end+1,1) = row; %#ok<AGROW>
end
metrics = struct2table(rows); summaryRows = repmat(emptyMetricSummaryRow(),0,1);
keys = unique(metrics(:,{'target','model','split','SMD'}),'rows','stable');
for index = 1:height(keys)
    key = keys(index,:); part = metrics(strcmp(metrics.target,key.target{1}) & ...
        strcmp(metrics.model,key.model{1}) & strcmp(metrics.split,key.split{1}) & ...
        metrics.SMD == key.SMD,:);
    row = emptyMetricSummaryRow(); row.target = key.target{1}; row.model = key.model{1};
    row.split = key.split{1}; row.SMD = key.SMD; row.independent_runs = height(part);
    row.MAE_median = median(part.MAE,'omitnan'); row.MAE_IQR = iqrValue(part.MAE);
    row.RMSE_median = median(part.RMSE,'omitnan'); row.RMSE_IQR = iqrValue(part.RMSE);
    summaryRows(end+1,1) = row; %#ok<AGROW>
end
summary = struct2table(summaryRows);
end

function [runSummary,summary] = strategySummaries(selection)
rows = repmat(emptyRunStrategyRow(),0,1);
keys = unique(selection(:,{'target','policy','SMD','run_no','run_key'}),'rows','stable');
for index = 1:height(keys)
    key = keys(index,:); part = selection(strcmp(selection.target,key.target{1}) & ...
        strcmp(selection.policy,key.policy{1}) & selection.SMD == key.SMD & ...
        selection.run_no == key.run_no,:);
    row = emptyRunStrategyRow(); row.target = key.target{1}; row.policy = key.policy{1};
    row.SMD = key.SMD; row.run_no = key.run_no; row.run_key = key.run_key{1};
    row.state_count = height(part);
    row.raw_delta_F_mean = mean(part.actual_raw_delta_F,'omitnan');
    row.raw_delta_F_median = median(part.actual_raw_delta_F,'omitnan');
    row.response_mean = mean(part.actual_response,'omitnan');
    row.raw_regret_mean = mean(part.raw_regret_to_oracle,'omitnan');
    row.oracle_match_rate = mean(part.selected_matches_oracle,'omitnan');
    rows(end+1,1) = row; %#ok<AGROW>
end
runSummary = struct2table(rows); summaryRows = repmat(emptyStrategySummaryRow(),0,1);
keys = unique(runSummary(:,{'target','policy','SMD'}),'rows','stable');
for index = 1:height(keys)
    key = keys(index,:); part = runSummary(strcmp(runSummary.target,key.target{1}) & ...
        strcmp(runSummary.policy,key.policy{1}) & runSummary.SMD == key.SMD,:);
    row = emptyStrategySummaryRow(); row.target = key.target{1}; row.policy = key.policy{1};
    row.SMD = key.SMD; row.independent_runs = height(part);
    row.raw_delta_F_median = median(part.raw_delta_F_mean,'omitnan');
    row.raw_delta_F_IQR = iqrValue(part.raw_delta_F_mean);
    row.response_median = median(part.response_mean,'omitnan');
    row.response_IQR = iqrValue(part.response_mean);
    row.raw_regret_median = median(part.raw_regret_mean,'omitnan');
    row.oracle_match_rate_mean = mean(part.oracle_match_rate,'omitnan');
    summaryRows(end+1,1) = row; %#ok<AGROW>
end
summary = struct2table(summaryRows);
end

function [differences,tests] = pairedStrategyTests(runSummary)
comparisons = {'M1','M0';'M1','M1_d_only';'M1','M1_q_only'; ...
    'M1','fixed_eta_025';'M1','fixed_eta_050';'M1','fixed_eta_075'; ...
    'M1','fixed_eta_100';'M0','fixed_eta_025';'M0','fixed_eta_050'; ...
    'M0','fixed_eta_075';'M0','fixed_eta_100'};
rows = repmat(emptyDifferenceRow(),0,1); testRows = repmat(emptyTestRow(),0,1);
for targetCell = unique(runSummary.target)'
    target = targetCell{1};
    for smd = unique(runSummary.SMD)'
        part = runSummary(strcmp(runSummary.target,target) & runSummary.SMD == smd,:);
        for pairIndex = 1:size(comparisons,1)
            left = part(strcmp(part.policy,comparisons{pairIndex,1}),:);
            right = part(strcmp(part.policy,comparisons{pairIndex,2}),:);
            commonRuns = intersect(left.run_no,right.run_no);
            left = sortrows(left(ismember(left.run_no,commonRuns),:),'run_no');
            right = sortrows(right(ismember(right.run_no,commonRuns),:),'run_no');
            if isempty(commonRuns), continue; end
            for runIndex = 1:height(left)
                row = emptyDifferenceRow(); row.target = target; row.SMD = smd;
                row.run_no = left.run_no(runIndex); row.run_key = left.run_key{runIndex};
                row.policy_A = comparisons{pairIndex,1}; row.policy_B = comparisons{pairIndex,2};
                row.delta_raw_delta_F_A_minus_B = left.raw_delta_F_mean(runIndex)- ...
                    right.raw_delta_F_mean(runIndex);
                row.delta_response_A_minus_B = left.response_mean(runIndex)- ...
                    right.response_mean(runIndex);
                rows(end+1,1) = row; %#ok<AGROW>
            end
            test = pairedWilcoxonSignedRank(left.raw_delta_F_mean,right.raw_delta_F_mean);
            row = emptyTestRow(); row.target = target; row.SMD = smd;
            row.policy_A = comparisons{pairIndex,1}; row.policy_B = comparisons{pairIndex,2};
            row.independent_runs = height(left); row.nonzero_pairs = test.n;
            row.delta_raw_delta_F_median = median(left.raw_delta_F_mean-right.raw_delta_F_mean,'omitnan');
            row.delta_raw_delta_F_IQR = iqrValue(left.raw_delta_F_mean-right.raw_delta_F_mean);
            row.p_raw_delta_F = test.p; row.significant_0_05 = double(test.h);
            row.test_method = test.method;
            testRows(end+1,1) = row; %#ok<AGROW>
        end
    end
end
differences = struct2table(rows); tests = struct2table(testRows);
end

function value = iqrValue(values)
values = values(isfinite(values));
if numel(values) < 2, value = NaN; return; end
value = prctile(values,75)-prctile(values,25);
end

function name = alphaName(alpha)
name = sprintf('robust_y_a%03d',round(100*alpha));
end

function name = floorNameForAlpha(alpha)
name = sprintf('floor_a%03d',round(100*alpha));
end

function row = statusParameterRow(target,smd,model,status)
row = emptyParameterRow(); row.target = target; row.SMD = smd;
row.model = model; row.status = status;
end

function row = parameterRow(model,target,smd,fold,spec,runCount,stateCount)
row = emptyParameterRow(); row.target = target; row.SMD = smd; row.fold = fold;
row.model = spec.name; row.status = model.status; row.training_runs = runCount;
row.training_states = stateCount; row.lambda = model.lambda;
row.huber_delta = model.huber_delta; row.d_log_mean = model.d_mean;
row.d_log_std = model.d_std; row.q_log_mean = model.q_mean; row.q_log_std = model.q_std;
if numel(model.beta) >= 2, row.a1 = model.beta(1); row.a2 = model.beta(2); end
position = 3;
if spec.useD && numel(model.beta) >= position+1
    row.b_d_u = model.beta(position); row.c_d_u2 = model.beta(position+1);
    position = position+2;
end
if spec.useQ && numel(model.beta) >= position+1
    row.b_q_u = model.beta(position); row.c_q_u2 = model.beta(position+1);
end
end

function row = emptyStateScaleRow()
row = struct('SMD',NaN,'run_no',NaN,'state_index',NaN,'generation',NaN, ...
    'sF',NaN,'prefix_median_sF',NaN,'floor_a001',NaN,'floor_a005',NaN, ...
    'floor_a010',NaN);
end
function row = emptyAuditRow()
row = struct('SMD',NaN,'branch_rows',NaN,'independent_runs',NaN, ...
    'matched_state_repeats',NaN,'min_sF',NaN,'sF_le_1e_6',NaN, ...
    'max_abs_original_relative_y',NaN,'max_abs_raw_delta_F',NaN, ...
    'eta050_max_abs_raw_delta',NaN,'common_checkpoint_groups',NaN);
end
function row = emptyCollapsedRow()
row = struct('SMD',NaN,'run_no',NaN,'run_key','','state_index',NaN,'phase','', ...
    'eta',NaN,'is_training_eta',NaN,'is_intermediate_eta',NaN, ...
    'is_reference_eta',NaN,'d',NaN,'q',NaN,'sF',NaN,'prefix_median_sF',NaN, ...
    'repeat_count',NaN,'raw_delta_F',NaN,'raw_delta_F_median',NaN, ...
    'raw_delta_F_std',NaN,'F_end',NaN,'F_end_reference_eta050',NaN, ...
    'robust_y_a001',NaN,'robust_y_a005',NaN,'robust_y_a010',NaN);
end
function row = emptyParameterRow()
row = struct('target','','SMD',NaN,'fold',NaN,'model','','status','', ...
    'training_runs',NaN,'training_states',NaN,'lambda',NaN,'huber_delta',NaN, ...
    'd_log_mean',NaN,'d_log_std',NaN,'q_log_mean',NaN,'q_log_std',NaN, ...
    'a1',NaN,'a2',NaN,'b_d_u',NaN,'c_d_u2',NaN,'b_q_u',NaN,'c_q_u2',NaN);
end
function row = emptyPredictionRow()
row = struct('target','','SMD',NaN,'run_no',NaN,'run_key','','state_index',NaN, ...
    'phase','','eta',NaN,'model','','fold',NaN,'split','', ...
    'observed_response',NaN,'predicted_response',NaN,'raw_delta_F',NaN, ...
    'absolute_error',NaN,'squared_error',NaN);
end
function row = emptySelectionRow()
row = struct('target','','SMD',NaN,'run_no',NaN,'run_key','','state_index',NaN, ...
    'phase','','fold',NaN,'policy','','selected_eta',NaN, ...
    'predicted_response',NaN,'actual_response',NaN,'actual_raw_delta_F',NaN, ...
    'oracle_eta',NaN,'oracle_raw_delta_F',NaN,'raw_regret_to_oracle',NaN, ...
    'selected_matches_oracle',NaN);
end
function row = emptyMetricRow()
row = struct('target','','model','','split','','SMD',NaN,'run_no',NaN, ...
    'run_key','','state_count',NaN,'MAE',NaN,'RMSE',NaN);
end
function row = emptyMetricSummaryRow()
row = struct('target','','model','','split','','SMD',NaN,'independent_runs',NaN, ...
    'MAE_median',NaN,'MAE_IQR',NaN,'RMSE_median',NaN,'RMSE_IQR',NaN);
end
function row = emptyRunStrategyRow()
row = struct('target','','policy','','SMD',NaN,'run_no',NaN,'run_key','', ...
    'state_count',NaN,'raw_delta_F_mean',NaN,'raw_delta_F_median',NaN, ...
    'response_mean',NaN,'raw_regret_mean',NaN,'oracle_match_rate',NaN);
end
function row = emptyStrategySummaryRow()
row = struct('target','','policy','','SMD',NaN,'independent_runs',NaN, ...
    'raw_delta_F_median',NaN,'raw_delta_F_IQR',NaN,'response_median',NaN, ...
    'response_IQR',NaN,'raw_regret_median',NaN,'oracle_match_rate_mean',NaN);
end
function row = emptyDifferenceRow()
row = struct('target','','SMD',NaN,'run_no',NaN,'run_key','','policy_A','', ...
    'policy_B','','delta_raw_delta_F_A_minus_B',NaN, ...
    'delta_response_A_minus_B',NaN);
end
function row = emptyTestRow()
row = struct('target','','SMD',NaN,'policy_A','','policy_B','', ...
    'independent_runs',NaN,'nonzero_pairs',NaN,'delta_raw_delta_F_median',NaN, ...
    'delta_raw_delta_F_IQR',NaN,'p_raw_delta_F',NaN, ...
    'significant_0_05',NaN,'test_method','');
end
