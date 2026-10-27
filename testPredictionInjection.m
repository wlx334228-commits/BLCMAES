function report = testPredictionInjection(outputDir)
%TESTPREDICTIONINJECTION Test B1/B2 sources, evaluation and CMA updates.
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
rawUpper = [1,-1;-2,2;0.5,0.5;-0.5,-0.5];
rawLower = zeros(4,BI.l_dim);
synthetic = struct();
for mode = {'A0','B1','B2'}
    options = predictionInjectionOptions(withMode(BI,mode{1},true));
    state = initializePredictionInjectionState(BI,options);
    state.pending = struct('valid',true,'x_pred',[0.5,-0.5], ...
        'source_generation',10,'model',struct());
    rngBefore = rng;
    [upper,lower,entry] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
        11,BI,options,state);
    assert(isequal(rngBefore,rng), ...
        'Injection helper changed the global random sequence.');
    switch mode{1}
        case 'A0'
            assert(isequal(upper,rawUpper) && isequal(lower,rawLower));
            assert(all(strcmp(entry.source_labels,'normal')));
        case 'B1'
            assert(isequal(upper(1:3,:),rawUpper(1:3,:)) && ...
                norm(upper(4,:)-[0.5,-0.5]) < 1e-12);
            assert(isequal(lower,rawLower));
            assert(isequal(entry.source_labels, ...
                {'normal';'normal';'normal';'predicted_direct'}));
        case 'B2'
            assert(isequal(upper(1:2,:),rawUpper(1:2,:)) && ...
                norm(upper(3,:)-[0.5,-0.5]) < 1e-12);
            assert(isequal(entry.source_labels, ...
                {'normal';'normal';'predicted_direct';'predicted_auxiliary'}));
            assert(~isequal([upper(4,:),lower(4,:)], ...
                [rawUpper(4,:),rawLower(4,:)]), ...
                'B2 did not create an auxiliary sample.');
    end
    synthetic.(mode{1}) = entry;
end

% A0/B1 must have identical normal samples during the ten-generation warmup;
% the B1/B2 candidate sources then must remain consistent through LL/UL/CMA.
baseBI = getBLOPinfo('SMD',1,5);
baseBI.UmaxFEs = 144;
baseBI.UmaxImprFEs = 350;
baseBI.LmaxFEs = 250;
baseBI.LmaxImprFEs = 25;
baseBI.u_ftol = 0;
integration = struct();
for mode = {'A0','B1','B2'}
    BIrun = withMode(baseBI,mode{1},true);
    rng(1,'twister');
    ins = BLCMAES(BIrun);
    log = ins.predictionInjectionLog;
    assert(all([log.consistency_checked]) && all([log.consistency_passed]), ...
        'Consistency audit failed for %s.',mode{1});
    assert(all(arrayfun(@(item) isequal(item.upper_entering_lower_search, ...
        item.upper_after_injection),log)), ...
        'Lower-level search did not receive injection coordinates.');
    for generation = 1:min(10,numel(log))
        assert(all(strcmp(log(generation).source_labels,'normal')), ...
            'Prediction was injected before generation 11.');
    end
    for generation = 1:numel(log)
        item = log(generation);
        if item.mean_update_applied
            assert(isequal(item.selected_parent_joint, ...
                item.joint_for_cma_update(item.selected_parent_indices,:)));
            assert(numel(item.parent_step_scales) == ...
                numel(item.selected_parent_indices));
            assert(all(isfinite(item.parent_step_scales)));
        end
    end
    integration.(mode{1}) = struct('UF',ins.UF,'LF',ins.LF, ...
        'UFE',ins.UFEs,'LFE',ins.LFEs, ...
        'injectedGenerations',sum([log.injected_count] > 0), ...
        'consistencyPassed',all([log.consistency_passed]));
    integration.(mode{1}).log = log;
end
for generation = 1:min(10,numel(integration.A0.log))
    assert(isequal(integration.A0.log(generation).joint_before_injection, ...
        integration.B1.log(generation).joint_before_injection), ...
        'B1 changed the normal sampling sequence during warmup.');
end

report = struct('passed',true,'synthetic',synthetic,'integration',integration);
save(fullfile(outputDir,'prediction_injection_test_report.mat'),'report');
disp(report);
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
