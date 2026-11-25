function result = analyseNextGenerationScaleModels(outcomes,analysisOptions)
%ANALYSENEXTGENERATIONSCALEMODELS Grouped-CV M0/M1 next-generation study.
% The response is the matched, real t+1 normal-probe top-two UF difference
% R=(Q(eta)-Q(reference))/s_ref.  K3 scale branches alone train the models;
% structural controls (A0 or A1) never become an eta training observation.

if nargin < 2 || isempty(analysisOptions), analysisOptions = struct(); end
settings = normaliseAnalysisSettings(analysisOptions);

required = {'SMD','run_no','state_index','repeat_no','variant', ...
    'candidate_policy','eta','is_scale_branch','probe_completed','d','q', ...
    's_ref','Q_top2_UF','raw_delta_Q','R_stable'};
assert(istable(outcomes) && all(ismember(required,outcomes.Properties.VariableNames)), ...
    'Next-generation outcome table is missing required columns.');

scaleRows = outcomes(outcomes.is_scale_branch ~= 0 & ...
    outcomes.probe_completed ~= 0 & isfinite(outcomes.eta) & ...
    isfinite(outcomes.R_stable) & isfinite(outcomes.raw_delta_Q),:);
collapsed = collapseRepeats(scaleRows);
specs = {'M0',false,false; 'M1_d_only',true,false; ...
    'M1_q_only',false,true; 'M1',true,true};
parameterRows = repmat(emptyParameterRow(),0,1);
predictionRows = repmat(emptyPredictionRow(),0,1);
selectionRows = repmat(emptySelectionRow(),0,1);

for smd = unique(collapsed.SMD)'
    problemRows = collapsed(collapsed.SMD == smd,:);
    runs = unique(problemRows.run_no);
    if numel(runs) < 5
        for specIndex = 1:size(specs,1)
            parameterRows(end+1,1) = statusParameterRow(smd,specs{specIndex,1}, ...
                'insufficient_independent_runs_for_grouped_fivefold_cv'); %#ok<AGROW>
        end
        continue;
    end
    foldMap = assignRunFolds(runs,5);
    for fold = 1:5
        testRuns = foldMap.run_no(foldMap.fold == fold);
        trainRows = problemRows(~ismember(problemRows.run_no,testRuns),:);
        testRows = problemRows(ismember(problemRows.run_no,testRuns),:);
        for specIndex = 1:size(specs,1)
            spec = struct('name',specs{specIndex,1}, ...
                'useD',specs{specIndex,2},'useQ',specs{specIndex,3});
            model = fitModel(trainRows,spec,settings);
            parameterRows(end+1,1) = parameterRow(model,smd,fold,spec, ...
                numel(unique(trainRows.run_no)),stateCount(trainRows)); %#ok<AGROW>
            states = unique(testRows(:,{'run_no','state_index'}),'rows','stable');
            for i = 1:height(states)
                rows = sortrows(testRows(testRows.run_no == states.run_no(i) & ...
                    testRows.state_index == states.state_index(i),:),'eta');
                assertMatchedEtaSet(rows,settings.etas);
                predicted = predictModel(model,rows);
                for j = 1:height(rows)
                    predictionRows(end+1,1) = predictionRow(rows(j,:), ...
                        spec.name,fold,predicted(j)); %#ok<AGROW>
                end
                selectionRows = [selectionRows;modelSelectionRows(rows,predicted, ...
                    spec.name,fold,true,settings)]; %#ok<AGROW>
            end
        end
        states = unique(testRows(:,{'run_no','state_index'}),'rows','stable');
        for i = 1:height(states)
            rows = sortrows(testRows(testRows.run_no == states.run_no(i) & ...
                testRows.state_index == states.state_index(i),:),'eta');
            assertMatchedEtaSet(rows,settings.etas);
            selectionRows = [selectionRows;modelSelectionRows(rows,NaN(height(rows),1), ...
                'fixed',fold,false,settings)]; %#ok<AGROW>
        end
    end
end

parameterTable = struct2table(parameterRows);
predictionTable = struct2table(predictionRows);
selectionTable = struct2table(selectionRows);
metricTable = makeMetricTable(predictionTable);
strategySummary = makeStrategySummary(selectionTable);
pairedTests = makePairedTests(selectionTable,settings);
a1Summary = makeA1Summary(outcomes);
referenceTests = makeReferenceComparisonTests(collapsed,settings);
result = struct('collapsedOutcomes',collapsed,'parameterTable',parameterTable, ...
    'predictionTable',predictionTable,'selectionTable',selectionTable, ...
    'metricTable',metricTable,'strategySummary',strategySummary, ...
    'pairedTests',pairedTests,'a1Summary',a1Summary, ...
    'referenceTests',referenceTests,'analysisSettings',settings);
end

function settings = normaliseAnalysisSettings(requested)
defaults = struct('etas',[.25 .50 .75 1.00], ...
    'referenceVariant','eta_050','responseMode','anchored_eta', ...
    'fallbackEta',.50);
assert(isstruct(requested) && isscalar(requested), ...
    'analysisOptions must be one scalar structure.');
names = fieldnames(requested);
for i = 1:numel(names)
    assert(isfield(defaults,names{i}), ...
        'Unknown next-generation analysis option: %s.',names{i});
    defaults.(names{i}) = requested.(names{i});
end
settings = defaults;
settings.etas = sort(settings.etas(:)');
validateattributes(settings.etas,{'numeric'}, ...
    {'real','finite','positive','vector','nonempty'});
assert(numel(unique(settings.etas)) == numel(settings.etas), ...
    'Analysis eta values must be unique.');
settings.referenceVariant = char(settings.referenceVariant);
settings.responseMode = char(settings.responseMode);
validateattributes(settings.fallbackEta,{'numeric'}, ...
    {'real','finite','scalar','positive'});
assert(any(abs(settings.etas-settings.fallbackEta) < 1e-12), ...
    'fallbackEta must be one of the analysed scale values.');
assert(ismember(settings.responseMode,{'anchored_eta','baseline_relative'}), ...
    'responseMode must be anchored_eta or baseline_relative.');
if strcmp(settings.responseMode,'anchored_eta')
    assert(any(abs(settings.etas-.50) < 1e-12), ...
        'anchored_eta analysis requires eta=.50.');
end
end

function assertMatchedEtaSet(rows,etas)
assert(height(rows) == numel(etas) && ...
    all(abs(rows.eta(:)-etas(:)) < 1e-12), ...
    'A complete matched set of configured eta branches is required for every test state.');
end

function collapsed = collapseRepeats(rows)
keys = unique(rows(:,{'SMD','run_no','state_index','eta'}),'rows','stable');
items = repmat(emptyCollapsedRow(),0,1);
for i = 1:height(keys)
    part = rows(rows.SMD == keys.SMD(i) & rows.run_no == keys.run_no(i) & ...
        rows.state_index == keys.state_index(i) & abs(rows.eta-keys.eta(i)) < 1e-12,:);
    row = emptyCollapsedRow();
    row.SMD=keys.SMD(i); row.run_no=keys.run_no(i); row.state_index=keys.state_index(i);
    row.eta=keys.eta(i); row.repeat_count=height(part);
    row.d=mean(part.d,'omitnan'); row.q=mean(part.q,'omitnan');
    row.s_ref=mean(part.s_ref,'omitnan'); row.generation=mean(part.generation,'omitnan');
    row.Q_top2_UF=mean(part.Q_top2_UF,'omitnan');
    row.raw_delta_Q=mean(part.raw_delta_Q,'omitnan');
    row.R_stable=mean(part.R_stable,'omitnan');
    row.mean_upper_error=mean(part.probe_mean_upper_error,'omitnan');
    row.mean_lower_error=mean(part.probe_mean_lower_error,'omitnan');
    row.probe_UFE_delta=mean(part.probe_UFE_delta,'omitnan');
    row.probe_LFE_delta=mean(part.probe_LFE_delta,'omitnan');
    items(end+1,1)=row; %#ok<AGROW>
end
collapsed = sortrows(struct2table(items),{'SMD','run_no','state_index','eta'});
end

function model = fitModel(rows,spec,settings)
model = struct('name',spec.name,'useD',spec.useD,'useQ',spec.useQ, ...
    'beta',NaN(0,1),'lambda',NaN,'d_mean',NaN,'d_std',NaN, ...
    'q_mean',NaN,'q_std',NaN,'status','insufficient_data', ...
    'responseMode',settings.responseMode,'fallbackEta',settings.fallbackEta);
valid = isfinite(rows.R_stable) & isfinite(rows.d) & isfinite(rows.q);
rows = rows(valid,:);
if height(rows) < 8, return; end
[X,model] = designMatrix(rows,model,true);
y = rows.R_stable;
candidateLambda = [0,1e-6,1e-4,1e-2,1,10];
model.lambda = chooseLambda(rows,spec,candidateLambda,settings);
model.beta = (X'*X + model.lambda*eye(size(X,2))) \ (X'*y);
if all(isfinite(model.beta)), model.status='fitted'; else, model.status='nonfinite_solution'; end
end

function lambda = chooseLambda(rows,spec,candidates,settings)
runs = unique(rows.run_no);
if numel(runs) < 3, lambda = 1e-2; return; end
loss = Inf(size(candidates));
for l = 1:numel(candidates)
    errors = zeros(0,1);
    for r = runs(:)'
        train = rows(rows.run_no ~= r,:); test = rows(rows.run_no == r,:);
        model = struct('name',spec.name,'useD',spec.useD,'useQ',spec.useQ, ...
            'beta',NaN(0,1),'lambda',candidates(l),'d_mean',NaN,'d_std',NaN, ...
            'q_mean',NaN,'q_std',NaN,'status','', ...
            'responseMode',settings.responseMode,'fallbackEta',settings.fallbackEta);
        [Xtrain,model] = designMatrix(train,model,true);
        beta = (Xtrain'*Xtrain+candidates(l)*eye(size(Xtrain,2))) \ (Xtrain'*train.R_stable);
        model.beta = beta; prediction = predictModel(model,test);
        errors = [errors;(prediction-test.R_stable).^2]; %#ok<AGROW>
    end
    loss(l)=mean(errors,'omitnan');
end
[~,index]=min(loss); lambda=candidates(index);
end

function [X,model] = designMatrix(rows,model,fitNormalizer)
eta = rows.eta(:);
u = eta-.50; u2=u.^2;
d = log(1+max(rows.d(:),0)); q = log(1+max(rows.q(:),0));
if fitNormalizer
    model.d_mean=mean(d,'omitnan'); model.d_std=max(std(d,0,'omitnan'),sqrt(eps));
    model.q_mean=mean(q,'omitnan'); model.q_std=max(std(q,0,'omitnan'),sqrt(eps));
end
d=(d-model.d_mean)/model.d_std; q=(q-model.q_mean)/model.q_std;
if strcmp(model.responseMode,'anchored_eta')
    % The legacy response is exactly zero at eta=.50 by construction.
    X=[u,u2];
    if model.useD, X=[X,d.*u,d.*u2]; end %#ok<AGROW>
    if model.useQ, X=[X,q.*u,q.*u2]; end %#ok<AGROW>
else
    % For A0-referenced responses, every tested eta may depart from zero.
    % Retain an intercept and state main effects, then add the same
    % first-/second-order state-by-scale interactions used by the legacy M1.
    X=[ones(size(eta)),eta,eta.^2];
    if model.useD, X=[X,d,d.*eta,d.*eta.^2]; end %#ok<AGROW>
    if model.useQ, X=[X,q,q.*eta,q.*eta.^2]; end %#ok<AGROW>
end
end

function prediction = predictModel(model,rows)
if isempty(model.beta) || any(~isfinite(model.beta)), prediction=NaN(height(rows),1); return; end
[X,~]=designMatrix(rows,model,false); prediction=X*model.beta;
end

function rows = modelSelectionRows(discrete,predicted,policy,fold,isModel,settings)
[oracleValue,~] = max(discrete.R_stable);
oracle = discrete(abs(discrete.R_stable-oracleValue) <= 1e-12*max(1,abs(oracleValue)),:);
    oracleEta = tieEta(oracle.eta,settings.fallbackEta);
rows = repmat(emptySelectionRow(),0,1);
if isModel
    if any(~isfinite(predicted))
        eta = settings.fallbackEta; predicted(:) = NaN;
    else
        eta = tieEta(discrete.eta(abs(predicted-max(predicted)) <= ...
            1e-12*max(1,abs(max(predicted)))),settings.fallbackEta);
    end
    chosen = discrete(abs(discrete.eta-eta) < 1e-12,:);
    rows(end+1,1)=selectionRow(chosen,policy,fold,predicted(find(abs(discrete.eta-eta)<1e-12,1)),oracleEta,oracleValue); %#ok<AGROW>
else
    for eta = settings.etas
        chosen=discrete(abs(discrete.eta-eta)<1e-12,:);
        rows(end+1,1)=selectionRow(chosen,sprintf('fixed_eta_%03d',round(100*eta)),fold,NaN,oracleEta,oracleValue); %#ok<AGROW>
    end
end
end

function row = selectionRow(chosen,policy,fold,predicted,oracleEta,oracleValue)
row=emptySelectionRow(); row.SMD=chosen.SMD; row.run_no=chosen.run_no;
row.state_index=chosen.state_index; row.generation=chosen.generation; row.fold=fold;
row.policy=policy; row.selected_eta=chosen.eta; row.predicted_R=predicted;
row.actual_R=chosen.R_stable; row.actual_raw_delta_Q=chosen.raw_delta_Q;
row.oracle_eta=oracleEta; row.oracle_R=oracleValue;
row.raw_regret_to_oracle=oracleValue-chosen.R_stable;
row.selected_matches_oracle=double(abs(chosen.eta-oracleEta)<1e-12);
end

function value = tieEta(etas,target)
etas=etas(:); [~,order]=sortrows([abs(etas-target),etas],[1 2]); value=etas(order(1));
end

function tableData = makeMetricTable(predictions)
rows=repmat(emptyMetricRow(),0,1);
for smd=unique(predictions.SMD)'
    for modelCell=unique(predictions.model)'
        model=modelCell{1}; part=predictions(predictions.SMD==smd & strcmp(predictions.model,model),:);
        for run=unique(part.run_no)'
            subset=part(part.run_no==run,:); row=emptyMetricRow(); row.SMD=smd; row.model=model; row.run_no=run;
            residual=subset.predicted_R-subset.actual_R;
            row.MAE=mean(abs(residual),'omitnan'); row.RMSE=sqrt(mean(residual.^2,'omitnan'));
            rows(end+1,1)=row; %#ok<AGROW>
        end
    end
end
tableData=struct2table(rows);
end

function tableData = makeStrategySummary(selection)
rows=repmat(emptyStrategyRow(),0,1);
for smd=unique(selection.SMD)'
    for policyCell=unique(selection.policy)'
        policy=policyCell{1}; part=selection(selection.SMD==smd & strcmp(selection.policy,policy),:);
        runRows=repmat(struct('run_no',NaN,'R',NaN,'raw',NaN,'match',NaN),0,1);
        for run=unique(part.run_no)'
            sub=part(part.run_no==run,:);
            runRows(end+1,1)=struct('run_no',run,'R',mean(sub.actual_R,'omitnan'), ...
                'raw',mean(sub.actual_raw_delta_Q,'omitnan'),'match',mean(sub.selected_matches_oracle,'omitnan')); %#ok<AGROW>
        end
        row=emptyStrategyRow(); row.SMD=smd; row.policy=policy; row.independent_runs=numel(runRows);
        row.R_median=median([runRows.R],'omitnan'); row.R_IQR=iqrValue([runRows.R]);
        row.raw_delta_Q_median=median([runRows.raw],'omitnan'); row.raw_delta_Q_IQR=iqrValue([runRows.raw]);
        row.oracle_match_rate=mean([runRows.match],'omitnan'); rows(end+1,1)=row; %#ok<AGROW>
    end
end
tableData=struct2table(rows);
end

function tableData = makeA1Summary(outcomes)
rows=repmat(emptyA1Row(),0,1);
part=outcomes(strcmp(outcomes.candidate_policy,'A1') & outcomes.probe_completed~=0 & isfinite(outcomes.R_stable),:);
for smd=unique(part.SMD)'
    for run=unique(part.run_no)'
        subset=part(part.SMD==smd & part.run_no==run,:); row=emptyA1Row();
        row.SMD=smd; row.run_no=run; row.mean_R=mean(subset.R_stable,'omitnan');
        row.mean_raw_delta_Q=mean(subset.raw_delta_Q,'omitnan'); row.state_count=height(subset);
        rows(end+1,1)=row; %#ok<AGROW>
    end
end
tableData=struct2table(rows);
end

function tests = makeReferenceComparisonTests(collapsed,settings)
% Collapse repeated states inside each independent main run before testing.
% Thus a long run that produces many valid prediction generations cannot be
% mistaken for many statistically independent comparisons.
rows = repmat(emptyReferenceTestRow(),0,1);
for smd = unique(collapsed.SMD)'
    for eta = settings.etas
        part = collapsed(collapsed.SMD == smd & abs(collapsed.eta-eta) < 1e-12,:);
        runNumbers = unique(part.run_no);
        values = NaN(numel(runNumbers),1);
        rawValues = NaN(numel(runNumbers),1);
        for i = 1:numel(runNumbers)
            subset = part(part.run_no == runNumbers(i),:);
            values(i) = mean(subset.R_stable,'omitnan');
            rawValues(i) = mean(subset.raw_delta_Q,'omitnan');
        end
        finite = isfinite(values);
        result = pairedWilcoxonSignedRank(values(finite),zeros(sum(finite),1));
        row = emptyReferenceTestRow();
        row.SMD=smd; row.reference_variant=settings.referenceVariant; row.eta=eta;
        row.independent_runs=sum(finite); row.R_median=median(values(finite),'omitnan');
        row.R_IQR=iqrValue(values(finite));
        row.raw_delta_Q_median=median(rawValues(finite),'omitnan');
        row.raw_delta_Q_IQR=iqrValue(rawValues(finite));
        row.p_value=result.p; row.significant_0_05=double(result.h);
        row.method=result.method; row.positive_median_advantage=double(row.R_median > 0);
        rows(end+1,1)=row; %#ok<AGROW>
    end
end
tests = struct2table(rows);
for smd = unique(tests.SMD)'
    subset = tests.SMD == smd;
    tests.p_value_holm(subset) = holmAdjust(tests.p_value(subset));
    tests.significant_positive_advantage(subset) = ...
        double(tests.p_value_holm(subset) < .05 & ...
        tests.R_median(subset) > 0);
end
end

function adjusted = holmAdjust(values)
adjusted = NaN(size(values));
finite = find(isfinite(values));
if isempty(finite), return; end
[ordered,order] = sort(values(finite));
count = numel(ordered); corrected = zeros(size(ordered)); running = 0;
for i = 1:count
    running = max(running,(count-i+1)*ordered(i));
    corrected(i) = min(running,1);
end
positions = finite(order); adjusted(positions) = corrected;
end

function tableData = makePairedTests(selection,settings)
fixedComparisons = arrayfun(@(eta){'M1', ...
    sprintf('fixed_eta_%03d',round(100*eta))},settings.etas, ...
    'UniformOutput',false);
comparisons = [{'M1','M0'};{'M1','M1_d_only'};{'M1','M1_q_only'}; ...
    vertcat(fixedComparisons{:})];
rows = repmat(emptyPairedTestRow(),0,1);
for smd = unique(selection.SMD)'
    for i = 1:size(comparisons,1)
        left = perRunPolicyMeans(selection,smd,comparisons{i,1});
        right = perRunPolicyMeans(selection,smd,comparisons{i,2});
        common = intersect(left.run_no,right.run_no);
        x = left.R(ismember(left.run_no,common));
        y = right.R(ismember(right.run_no,common));
        test = pairedWilcoxonSignedRank(x,y);
        row = emptyPairedTestRow(); row.SMD=smd; row.policy_A=comparisons{i,1};
        row.policy_B=comparisons{i,2}; row.independent_runs=numel(common);
        row.delta_R_median=median(x-y,'omitnan'); row.p_value=test.p;
        row.significant_0_05=double(test.h); row.method=test.method;
        rows(end+1,1)=row; %#ok<AGROW>
    end
end
tableData=struct2table(rows);
end

function data = perRunPolicyMeans(selection,smd,policy)
part=selection(selection.SMD==smd & strcmp(selection.policy,policy),:);
runs=unique(part.run_no); data=table(runs,NaN(numel(runs),1),'VariableNames',{'run_no','R'});
for i=1:numel(runs), data.R(i)=mean(part.actual_R(part.run_no==runs(i)),'omitnan'); end
end

function map = assignRunFolds(runs,k)
rows=repmat(struct('run_no',NaN,'fold',NaN),numel(runs),1);
for i=1:numel(runs), rows(i).run_no=runs(i); rows(i).fold=mod(i-1,k)+1; end
map=struct2table(rows);
end
function count = stateCount(rows)
count=height(unique(rows(:,{'run_no','state_index'}),'rows'));
end
function value = iqrValue(x)
x=x(isfinite(x));
if isempty(x), value=NaN; else, value=iqr(x); end
end

function row=parameterRow(model,smd,fold,spec,runCount,stateCountValue)
row=statusParameterRow(smd,spec.name,model.status); row.fold=fold; row.training_runs=runCount; row.training_states=stateCountValue;
row.lambda=model.lambda; row.d_log_mean=model.d_mean; row.d_log_std=model.d_std; row.q_log_mean=model.q_mean; row.q_log_std=model.q_std;
b=model.beta;
if strcmp(model.responseMode,'anchored_eta')
    if numel(b)>=2,row.a1=b(1);row.a2=b(2);end
    position=3;
    if spec.useD && numel(b)>=position+1
        row.b_d_u=b(position);row.c_d_u2=b(position+1);position=position+2;
    end
    if spec.useQ && numel(b)>=position+1
        row.b_q_u=b(position);row.c_q_u2=b(position+1);
    end
else
    if numel(b)>=3, row.a0=b(1);row.a1=b(2);row.a2=b(3);end
    position=4;
    if spec.useD && numel(b)>=position+2
        row.b_d=b(position);row.b_d_u=b(position+1);row.c_d_u2=b(position+2);position=position+3;
    end
    if spec.useQ && numel(b)>=position+2
        row.b_q=b(position);row.b_q_u=b(position+1);row.c_q_u2=b(position+2);
    end
end
end
function row=statusParameterRow(smd,model,status)
row=emptyParameterRow();row.SMD=smd;row.model=model;row.status=status;
end
function row=predictionRow(observed,model,fold,predicted)
row=emptyPredictionRow();row.SMD=observed.SMD;row.run_no=observed.run_no;row.state_index=observed.state_index;row.generation=observed.generation;row.eta=observed.eta;row.model=model;row.fold=fold;row.d=observed.d;row.q=observed.q;row.actual_R=observed.R_stable;row.predicted_R=predicted;row.actual_raw_delta_Q=observed.raw_delta_Q;
end

function row=emptyCollapsedRow(), row=struct('SMD',NaN,'run_no',NaN,'state_index',NaN,'eta',NaN,'repeat_count',NaN,'d',NaN,'q',NaN,'s_ref',NaN,'generation',NaN,'Q_top2_UF',NaN,'raw_delta_Q',NaN,'R_stable',NaN,'mean_upper_error',NaN,'mean_lower_error',NaN,'probe_UFE_delta',NaN,'probe_LFE_delta',NaN);end
function row=emptyParameterRow(),row=struct('SMD',NaN,'fold',NaN,'model','','status','','training_runs',NaN,'training_states',NaN,'lambda',NaN,'d_log_mean',NaN,'d_log_std',NaN,'q_log_mean',NaN,'q_log_std',NaN,'a0',NaN,'a1',NaN,'a2',NaN,'b_d',NaN,'b_d_u',NaN,'c_d_u2',NaN,'b_q',NaN,'b_q_u',NaN,'c_q_u2',NaN);end
function row=emptyPredictionRow(),row=struct('SMD',NaN,'run_no',NaN,'state_index',NaN,'generation',NaN,'eta',NaN,'model','','fold',NaN,'d',NaN,'q',NaN,'actual_R',NaN,'predicted_R',NaN,'actual_raw_delta_Q',NaN);end
function row=emptySelectionRow(),row=struct('SMD',NaN,'run_no',NaN,'state_index',NaN,'generation',NaN,'fold',NaN,'policy','','selected_eta',NaN,'predicted_R',NaN,'actual_R',NaN,'actual_raw_delta_Q',NaN,'oracle_eta',NaN,'oracle_R',NaN,'raw_regret_to_oracle',NaN,'selected_matches_oracle',NaN);end
function row=emptyMetricRow(),row=struct('SMD',NaN,'model','','run_no',NaN,'MAE',NaN,'RMSE',NaN);end
function row=emptyStrategyRow(),row=struct('SMD',NaN,'policy','','independent_runs',NaN,'R_median',NaN,'R_IQR',NaN,'raw_delta_Q_median',NaN,'raw_delta_Q_IQR',NaN,'oracle_match_rate',NaN);end
function row=emptyA1Row(),row=struct('SMD',NaN,'run_no',NaN,'mean_R',NaN,'mean_raw_delta_Q',NaN,'state_count',NaN);end
function row=emptyReferenceTestRow(),row=struct('SMD',NaN,'reference_variant','','eta',NaN,'independent_runs',NaN,'R_median',NaN,'R_IQR',NaN,'raw_delta_Q_median',NaN,'raw_delta_Q_IQR',NaN,'p_value',NaN,'p_value_holm',NaN,'significant_0_05',NaN,'positive_median_advantage',NaN,'significant_positive_advantage',NaN,'method','');end
function row=emptyPairedTestRow(),row=struct('SMD',NaN,'policy_A','','policy_B','','independent_runs',NaN,'delta_R_median',NaN,'p_value',NaN,'significant_0_05',NaN,'method','');end
