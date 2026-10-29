function report = testPredictionInjection(outputDir)
%TESTPREDICTIONINJECTION Validate K=0,1,2,4,8 injection bookkeeping.
% The test has no benchmark-result claim.  It checks candidate placement,
% private auxiliary RNG use, LL/UL input consistency and legacy B1/B2
% equivalence before a full K=0,...,8 experiment is launched.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(projectRoot,'results_prediction_injection_tests', ...
        datestr(now,'yyyymmdd_HHMMSS'));
end
assert(exist(outputDir,'dir') ~= 7,'Choose a new test directory.');
mkdir(outputDir);

BI = getBLOPinfo('SMD',1,5);
CMA = struct('xmean',[0,0,1,2,3],'sigma',0.5, ...
    'C',eye(5),'B',eye(5),'D',ones(1,5));
rawUpper = [1,-1;-2,2;0.5,0.5;-0.5,-0.5;1.5,-1.5; ...
    -1.5,1.5;0.25,-0.25;-0.25,0.25];
rawLower = zeros(8,BI.l_dim);
synthetic = struct();
for K = [0 1 2 4 8]
    options = predictionInjectionOptions(withCount(BI,K,true));
    state = initializePredictionInjectionState(BI,options);
    state.pending = pendingPrediction(BI);
    rngBefore = rng;
    [upper,lower,entry] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
        11,BI,options,state);
    assert(isequal(rngBefore,rng), ...
        'K=%d injection changed the global random sequence.',K);
    assert(sum(strcmp(entry.source_labels,'normal')) == 8-K);
    assert(sum(strcmp(entry.source_labels,'predicted_direct')) == (K > 0));
    assert(sum(strcmp(entry.source_labels,'predicted_auxiliary')) == max(K-1,0));
    assert(entry.injected_count == K && entry.effective_injected_count == K);
    if K == 0
        assert(isequal(upper,rawUpper) && isequal(lower,rawLower));
    else
        directIndex = 9-K;
        assert(norm(upper(directIndex,:)-[0.5,-0.5]) < 1e-12);
        assert(isequal(lower(directIndex,:),rawLower(directIndex,:)));
        assert(isequal(entry.auxiliary_indices,(directIndex+1):8));
    end
    synthetic.(sprintf('K%d',K)) = entry;
end

% K=1 and K=2 must keep the previously published B1/B2 candidate layouts.
legacyEquivalence = struct();
for K = [1 2]
    legacyMode = sprintf('B%d',K);
    legacyOptions = predictionInjectionOptions(withMode(BI,legacyMode,true));
    genericOptions = predictionInjectionOptions(withCount(BI,K,true));
    legacyState = initializePredictionInjectionState(BI,legacyOptions);
    genericState = initializePredictionInjectionState(BI,genericOptions);
    legacyState.pending = pendingPrediction(BI);
    genericState.pending = pendingPrediction(BI);
    [legacyUpper,legacyLower] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
        11,BI,legacyOptions,legacyState);
    [genericUpper,genericLower] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
        11,BI,genericOptions,genericState);
    assert(isequal(legacyUpper,genericUpper) && isequal(legacyLower,genericLower), ...
        'Legacy %s and generic K%d candidate layouts differ.',legacyMode,K);
    legacyEquivalence.(sprintf('K%d',K)) = true;
end

% Integration test: use actual SMD1 LL/UL calls with a small UFE cap.  It
% verifies that the exact evaluated coordinates become CMA-ES parent data.
baseBI = getBLOPinfo('SMD',1,5);
baseBI.UmaxFEs = 96;
baseBI.UmaxImprFEs = 350;
baseBI.LmaxFEs = 250;
baseBI.LmaxImprFEs = 25;
baseBI.u_ftol = 0;
integration = struct();
for K = [0 1 2 4 8]
    BIrun = withCount(baseBI,K,true);
    rng(1,'twister');
    ins = BLCMAES(BIrun);
    log = ins.predictionInjectionLog;
    assert(all([log.consistency_checked]) && all([log.consistency_passed]), ...
        'Consistency audit failed for K=%d.',K);
    assert(all(arrayfun(@(item) isequal(item.upper_entering_lower_search, ...
        item.upper_after_injection),log)), ...
        'Lower-level search did not receive actual K=%d candidate coordinates.',K);
    for generation = 1:min(10,numel(log))
        assert(all(strcmp(log(generation).source_labels,'normal')), ...
            'K=%d injected a prediction before generation 11.',K);
    end
    for generation = 1:numel(log)
        item = log(generation);
        if item.mean_update_applied
            assert(isequal(item.selected_parent_joint, ...
                item.joint_for_cma_update(item.selected_parent_indices,:)));
        end
    end
    integration.(sprintf('K%d',K)) = struct('UF',ins.UF,'LF',ins.LF, ...
        'UFE',ins.UFEs,'LFE',ins.LFEs, ...
        'injectedGenerations',sum([log.injected_count] > 0), ...
        'consistencyPassed',all([log.consistency_passed]));
end

% The full optimiser must also reproduce the prior B1/B2 trajectories for
% K=1/K=2, not merely their synthetic candidate layout.
legacyRunEquivalence = struct();
for K = [1 2]
    legacyMode = sprintf('B%d',K);
    legacyBI = withMode(baseBI,legacyMode,true);
    genericBI = withCount(baseBI,K,true);
    rng(17,'twister');
    legacyIns = BLCMAES(legacyBI);
    rng(17,'twister');
    genericIns = BLCMAES(genericBI);
    assertSameLegacyRun(legacyIns,genericIns,legacyMode,K);
    legacyRunEquivalence.(sprintf('K%d',K)) = true;
end

report = struct('passed',true,'synthetic',synthetic, ...
    'legacyEquivalence',legacyEquivalence, ...
    'legacyRunEquivalence',legacyRunEquivalence,'integration',integration);
save(fullfile(outputDir,'prediction_injection_test_report.mat'),'report');
disp(report);
end

function assertSameLegacyRun(legacyIns,genericIns,legacyMode,K)
assert(legacyIns.UF == genericIns.UF && legacyIns.LF == genericIns.LF && ...
    legacyIns.UFEs == genericIns.UFEs && legacyIns.LFEs == genericIns.LFEs && ...
    isequal(legacyIns.UX,genericIns.UX) && isequal(legacyIns.LX,genericIns.LX), ...
    'Legacy %s and K%d final outcomes differ.',legacyMode,K);
legacyLog = legacyIns.predictionInjectionLog;
genericLog = genericIns.predictionInjectionLog;
assert(numel(legacyLog) == numel(genericLog), ...
    'Legacy %s and K%d generation counts differ.',legacyMode,K);
for generation = 1:numel(legacyLog)
    assert(isequal(legacyLog(generation).joint_before_injection, ...
        genericLog(generation).joint_before_injection) && ...
        isequal(legacyLog(generation).joint_after_injection, ...
        genericLog(generation).joint_after_injection) && ...
        isequal(legacyLog(generation).joint_for_cma_update, ...
        genericLog(generation).joint_for_cma_update), ...
        'Legacy %s and K%d differ at generation %d.',legacyMode,K,generation);
end
end

function pending = pendingPrediction(BI)
pending = struct('valid',true,'x_pred',[0.5,-0.5], ...
    'source_generation',10,'model',struct());
assert(BI.u_dim == 2);
end

function BIout = withCount(BIin,K,consistencyCheckEnabled)
BIout = BIin;
BIout.enablePredictionCorrection = false;
BIout.enableIndividualPredictionCorrection = false;
BIout.predictionInjection = struct('mode',sprintf('K%d',K), ...
    'predictionInjectCount',K,'logEnabled',true,'startGeneration',10, ...
    'historyGenerations',10,'sampleCount',40,'auxiliaryRandomSeed',991, ...
    'consistencyCheckEnabled',consistencyCheckEnabled);
end

function BIout = withMode(BIin,mode,consistencyCheckEnabled)
BIout = BIin;
BIout.enablePredictionCorrection = false;
BIout.enableIndividualPredictionCorrection = false;
BIout.predictionInjection = struct('mode',mode,'logEnabled',true, ...
    'startGeneration',10,'historyGenerations',10,'sampleCount',40, ...
    'auxiliaryRandomSeed',991,'consistencyCheckEnabled', ...
    consistencyCheckEnabled);
end
