function report = testPredictionAblation(outputDir)
%TESTPREDICTIONABLATION Verify A0--A4 transforms and audit relationships.
projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(projectRoot,'results_prediction_ablation_tests', ...
        datestr(now,'yyyymmdd_HHMMSS'));
end
assert(exist(outputDir,'dir') ~= 7, ...
    'Use a new test directory; results must not be overwritten.');
mkdir(outputDir);

BI = getBLOPinfo('SMD',1,5);
CMA = struct('xmean',[0,0,1,2,3],'sigma',0.5, ...
    'C',eye(5),'pc',ones(1,5),'ps',2*ones(1,5), ...
    'B',eye(5),'D',ones(1,5),'invsqrtC',eye(5));
rawUpper = [1,-1;-2,2;0.5,0.5;-0.5,-0.5];
rawLower = zeros(4,BI.l_dim);
expected = struct('A0',rawUpper, ...
    'A1',[0.95,-0.95;-1.75,1.75;0.5,0.4;-0.4,-0.5], ...
    'A2',[1.05,-1.05;-1.95,1.95;0.55,0.45;-0.45,-0.55], ...
    'A3',[0.9,-0.9;-1.8,1.8;0.45,0.45;-0.45,-0.45]);
global ulFunctionEvaluations llFunctionEvaluations;
ulFunctionEvaluations = 123;
llFunctionEvaluations = 456;
synthetic = struct();
for mode = {'A0','A1','A2','A3','A4'}
    options = individualPredictionCorrectionOptions(withMode(BI,mode{1},true));
    state = initializeIndividualPredictionState(BI,options);
    state.pending = struct('valid',true,'x_pred',[0.5,-0.5], ...
        'source_generation',10,'model',struct());
    rngBefore = rng;
    [actual,entry] = applyIndividualPredictionCorrection(CMA,rawUpper,rawLower, ...
        11,BI,options,state);
    assert(isequal(rngBefore,rng),'Ablation helper changed global RNG.');
    assert(ulFunctionEvaluations == 123 && llFunctionEvaluations == 456, ...
        'Ablation helper changed function-evaluation counters.');
    assert(isequal(entry.joint_after_correction,[actual,rawLower]));
    switch mode{1}
        case {'A0','A1','A2','A3'}
            assert(norm(actual-expected.(mode{1}),'fro') < 1e-12, ...
                'Incorrect %s sample transform.',mode{1});
        case 'A4'
            assert(sum(entry.transformation_selected) == 2, ...
                'A4 did not select exactly two points.');
            changed = entry.transformation_selected;
            assert(norm(actual(~changed,:)-rawUpper(~changed,:),'fro') < 1e-12);
            assert(norm(actual(changed,:) - ...
                (0.9*rawUpper(changed,:) + 0.1*[0.5,-0.5]),'fro') < 1e-12);
    end
    synthetic.(mode{1}) = entry;
end

% Integration audit: normal sampled coordinates, true evaluations, and the
% exact parents consumed by updateCMAES must be the same logged positions.
baseBI = getBLOPinfo('SMD',1,5);
baseBI.UmaxFEs = 144;
baseBI.UmaxImprFEs = 350;
baseBI.LmaxFEs = 250;
baseBI.LmaxImprFEs = 25;
baseBI.u_N = 50;
baseBI.l_N = 50;
baseBI.u_ftol = 0;
integration = struct();
for mode = {'A0','A1','A2','A3','A4'}
    BIrun = withMode(baseBI,mode{1},true);
    rng(1,'twister');
    ins = BLCMAES(BIrun);
    log = ins.individualPredictionLog;
    assert(all([log.consistency_checked]) && all([log.consistency_passed]), ...
        'Consistency audit failed for %s.',mode{1});
    assert(all(arrayfun(@(item) isequal(item.upper_entering_lower_search, ...
        item.upper_after),log)), ...
        'The lower search did not receive the corrected upper coordinate.');
    assert(~any([log.mean_directly_corrected]), ...
        'Ablation mode directly modified CMA.xmean.');
    firstSelected = find([log.correction_selected_count] > 0,1);
    if strcmp(mode{1},'A0')
        assert(isempty(firstSelected),'A0 transformed an individual.');
    elseif strcmp(mode{1},'A3')
        assert(firstSelected == 11,'A3 must first act in generation 11.');
    elseif ~isempty(firstSelected)
        assert(firstSelected >= 11,'Prediction was used before generation 11.');
    end
    integration.(mode{1}) = struct('UF',ins.UF,'UFE',ins.UFEs, ...
        'LFE',ins.LFEs,'firstSelectedGeneration',firstSelected, ...
        'consistencyPassed',all([log.consistency_passed]));
end

report = struct('passed',true,'synthetic',synthetic,'integration',integration);
save(fullfile(outputDir,'prediction_ablation_test_report.mat'),'report');
disp(report);
end

function BIout = withMode(BIin,mode,consistencyCheckEnabled)
BIout = BIin;
BIout.enablePredictionCorrection = false;
BIout.enableIndividualPredictionCorrection = ~strcmp(mode,'A0');
BIout.individualPredictionCorrection = struct('mode',mode,'logEnabled',true, ...
    'weight',0.1,'startGeneration',10,'partialCount',2, ...
    'partialSelectionSeed',777,'consistencyCheckEnabled', ...
    consistencyCheckEnabled);
end
