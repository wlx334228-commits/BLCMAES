function report = testPredictionCorrection(referenceDir,outputDir)
%TESTPREDICTIONCORRECTION Numerical and fixed-seed integration verification.
% referenceDir contains the unmodified, pre-strategy BLCMAES.m snapshot.
projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 2
    outputDir = fullfile(projectRoot,'results_prediction_tests', ...
        datestr(now,'yyyymmdd_HHMMSS'));
end
assert(~exist(outputDir,'dir'),'Use a fresh test output directory.');
mkdir(outputDir);
BI = getBLOPinfo('SMD',1,5);
BI.enablePredictionCorrection = true;
options = predictionCorrectionOptions(BI);
% Nonzero cross-term; a full quadratic fit must recover the known maximum.
[x,y] = ndgrid(linspace(-1,1,7));
X = [x(:),y(:)];
pred = [-0.3,0.4];
Q = [2,0.7;0.7,1];
UF = 5-sum(((X-pred)*Q).*(X-pred),2);
history = [10*ones(49,1),(1:49)',X,UF];
augmented = [history;0,1,0,0,1e10;11,1,0,0,1e10];
rngBefore = rng;
model = fitLocalUpperQuadratic(augmented,[0,0],10,options);
assert(model.valid && norm(model.x_pred-pred) < 1e-10);
assert(model.fitTrainRMSE < 1e-10 && model.fitTrainR2 > 1-1e-12);
assert(model.fitLOOCVRMSE < 1e-9 && model.fitLOOCVR2 > 1-1e-10);
assert(model.zPredNorm > 0 && model.predictionInHull == 1);
assert(isequal(rngBefore,rng),'Fitting must not consume random numbers.');
assert(size(model.selectedSamples,1) == 40);
assert(all(model.selectedSamples(:,1) == 10),'Future/old samples were used.');
distances = sqrt(sum(X.^2,2));
[~,nearest] = sortrows([distances,(1:49)'],[1 2]);
assert(isequal(model.selectedSamples,history(nearest(1:40),:)));
shift = [2,-1];
shifted = history;
shifted(:,3:4) = 3*X + shift;
shiftModel = fitLocalUpperQuadratic(shifted,shift,10,options);
assert(shiftModel.valid && norm(shiftModel.x_pred-(3*pred+shift)) < 1e-10);
bad = history;
bad(:,5) = sum(X.^2,2);
m = fitLocalUpperQuadratic(bad,[0,0],10,options);
assert(~m.valid && strcmp(m.reason,'hessian_not_negative_definite'));
bad(:,5) = X(:,1).^2-X(:,2).^2;
m = fitLocalUpperQuadratic(bad,[0,0],10,options);
assert(~m.valid);
bad(:,5) = -X(:,1).^2-1e-12*X(:,2).^2;
m = fitLocalUpperQuadratic(bad,[0,0],10,options);
assert(~m.valid && strcmp(m.reason,'ill_conditioned_hessian'));
bad(:,3:4) = 0;
m = fitLocalUpperQuadratic(bad,[0,0],10,options);
assert(~m.valid && strcmp(m.reason,'degenerate_scale'));
bad = history; bad(:,4) = 0;
m = fitLocalUpperQuadratic(bad,[0,0],10,options);
assert(~m.valid && strcmp(m.reason,'ill_conditioned_design'));
m = fitLocalUpperQuadratic(history(1:39,:),[0,0],10,options);
assert(~m.valid && strcmp(m.reason,'insufficient_samples'));
m = fitLocalUpperQuadratic(history,[0,0],9,options);
assert(~m.valid && strcmp(m.reason,'warmup'));

% Signed-vector example, state invariance, boundary repair, optional cap.
for i = 1:49, POP(i) = struct('UX',X(i,:),'UF',UF(i)); end
CMA = struct('xmean',[0.2,-0.1,1,2,3],'sigma',0.5, ...
    'C',eye(5),'pc',ones(1,5),'ps',2*ones(1,5), ...
    'B',eye(5),'D',ones(1,5),'invsqrtC',eye(5));
global ulFunctionEvaluations llFunctionEvaluations;
ulFunctionEvaluations = 123; llFunctionEvaluations = 456;
rngBefore = rng;
[corrected,state] = applyUpperPredictionCorrection(CMA,[0,0],POP,10,BI, ...
    options,[],true,max(UF),max(UF),123,456);
assert(norm(corrected.xmean(1:2)-[-0.05,0.15]) < 1e-10);
assert(isequal(corrected.xmean(3:end),CMA.xmean(3:end)));
assert(isequal(rmfield(corrected,'xmean'),rmfield(CMA,'xmean')));
assert(isequal(rngBefore,rng));
assert(ulFunctionEvaluations == 123 && llFunctionEvaluations == 456);
assert(norm(state.log.d_new-[-0.05,0.15]) < 1e-10);
assert(abs(state.log.normal_move_norm - norm([0.2,-0.1])) < 1e-12);
assert(abs(state.log.predicted_move_norm - norm([-0.3,0.4])) < 1e-12);
assert(abs(state.log.correction_to_cma_ratio - norm([-0.25,0.25]) / ...
    norm([0.2,-0.1])) < 1e-10);
assert(abs(state.log.correction_to_sigma_ratio - norm([-0.25,0.25]) / ...
    (0.5*sqrt(2))) < 1e-10);
assert(abs(state.log.correction_mahalanobis - sqrt(0.5)) < 1e-12);
assert(abs(state.log.cos_pred_cma - ...
    dot([-0.3,0.4],[0.2,-0.1])/(0.5*norm([0.2,-0.1]))) < 1e-12);
[scheduled,scheduledState] = applyUpperPredictionCorrection(CMA,[0,0], ...
    POP,11,BI,options,state,true,max(UF),max(UF),123,456);
assert(~isequal(scheduled,CMA));
assert(scheduledState.log(end).correction_scheduled);
assert(strcmp(scheduledState.log(end).status,'applied'));
assert(size(scheduledState.history,1) == 2*numel(POP));
assert(scheduledState.log(1).next_generation == 11);
assert(scheduledState.log(1).next_UFE_delta == 0 && ...
    scheduledState.log(1).next_LFE_delta == 0);
limitedOptions = options;
limitedOptions.limitCorrection = true;
limitedOptions.maxCorrectionSigmaFactor = 0.01;
[~,limited] = applyUpperPredictionCorrection(CMA,[0,0],POP,10,BI, ...
    limitedOptions,[],true,max(UF),max(UF),123,456);
assert(limited.log.cap_applied);
assert(limited.log.correction_norm <= limited.log.correction_limit+1e-14);
boundaryBI = BI; boundaryBI.u_lb = [-1,-1]; boundaryBI.u_ub = [1,1];
for i = 1:49, POP(i).UF = -sum((X(i,:)-[100,100]).^2); end
[bounded,boundLog] = applyUpperPredictionCorrection(CMA,[0,0],POP,10, ...
    boundaryBI,options,[],true,0,0,123,456);
assert(boundLog.log.boundary_repaired);
assert(norm(bounded.xmean(1:2)-[0.6,0.45]) < 1e-10);
[terminal,terminalLog] = applyUpperPredictionCorrection(CMA,CMA.xmean(1:2), ...
    POP,11,BI,options,[],false,0,0,123,456);
assert(isequal(terminal,CMA) && ~terminalLog.log.mean_update_applied);

% Integration tests against an exact copy of the previous implementation.
assert(exist(fullfile(referenceDir,'BLCMAES.m'),'file') == 2, ...
    'Supply a directory containing the pre-change BLCMAES.m snapshot.');
report = struct('numericalChecksPassed',true,'problems',[],'changedProblems',[]);
for problem = [1,5,8]
    BI = getBLOPinfo('SMD',problem,5);
    BI.UmaxFEs = 400; BI.UmaxImprFEs = 350;
    BI.LmaxFEs = 250; BI.LmaxImprFEs = 25;
    BI.generationDiagnostic = struct('enabled',true);
    seed = (problem-1)*10+1;
    originalDir = pwd;
    cleanup = onCleanup(@()cd(originalDir));
    cd(referenceDir);
    clear BLCMAES;
    rng(seed,'twister');
    original = BLCMAES(BI);
    originalRNG = rng;
    cd(projectRoot);
    clear BLCMAES;
    BI.enablePredictionCorrection = false;
    BI.predictionCorrection = struct('logEnabled',true);
    rng(seed,'twister');
    baseline = BLCMAES(BI);
    baselineRNG = rng;
    assert(isequaln(original,rmfield(baseline,{'predictionLog','predictionOptions'})), ...
        'Disabled strategy differs from the pre-change implementation.');
    assert(isequal(originalRNG,baselineRNG));
    BI.enablePredictionCorrection = true;
    BI.predictionCorrection.weight = 0;
    rng(seed,'twister');
    zeroWeight = BLCMAES(BI);
    assert(isequaln(original,rmfield(zeroWeight,{'predictionLog','predictionOptions'})));
    assert(isequal(originalRNG,rng),'Fitting/logging changed RNG with zero weight.');
    BI.predictionCorrection.weight = 0.5;
    rng(seed,'twister');
    prediction = BLCMAES(BI);
    changed = any([prediction.predictionLog.correction_applied]);
    assert(isequaln(baseline.generationLog(1:10,:),prediction.generationLog(1:10,:)), ...
        'The trajectory changed before generation 10.');
    if changed
        first = find([prediction.predictionLog.correction_applied],1);
        assert(first >= 10);
        assert(mod(first-10,1) == 0);
        assert(norm(prediction.predictionLog(first).m_new - ...
                    baseline.predictionLog(first).m_new) > 0);
        assert(~isequaln(prediction.generationPopulationLog( ...
            prediction.generationPopulationLog(:,1) == first+1,:), ...
            baseline.generationPopulationLog( ...
            baseline.generationPopulationLog(:,1) == first+1,:)));
    end
    log = prediction.predictionLog;
    for j = 1:numel(log)
        if j > 1
            assert(isequal(log(j).m_before,log(j-1).m_new), ...
                'Next generation did not use the corrected sampling mean.');
        end
        if log(j).valid_fit
            samples = log(j).model.selectedSamples;
            assert(all(samples(:,1) <= j & samples(:,1) >= max(1,j-9)));
            assert(norm(log(j).d_pred-(log(j).x_pred-log(j).m_before)) < 1e-10);
            expectedMean = (1-log(j).weight_used)*log(j).m_cma + ...
                log(j).weight_used*log(j).x_pred;
            assert(norm(log(j).d_new-(expectedMean-log(j).m_before)) < 1e-10);
        end
        if log(j).correction_applied
            assert(log(j).correction_scheduled);
            assert(mod(log(j).generation-10,1) == 0);
        end
    end
    save(fullfile(outputDir,sprintf('SMD%d_smoke.mat',problem)), ...
        'original','baseline','zeroWeight','prediction','seed','BI');
    report.problems(end+1) = problem;
    if changed, report.changedProblems(end+1) = problem; end
    fprintf('Smoke SMD%d: baseline exact, weight=0 exact, corrections=%d\n', ...
        problem,sum([log.correction_applied]));
    clear cleanup;
end
assert(~isempty(report.changedProblems),'No real trajectory changes were observed.');
report.passed = true;
save(fullfile(outputDir,'test_report.mat'),'report');
disp(report);
end
