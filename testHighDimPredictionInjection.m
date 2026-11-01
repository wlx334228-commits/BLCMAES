function report = testHighDimPredictionInjection(outputDir,totalDimensions)
%TESTHIGHDIMPREDICTIONINJECTION Small non-evaluating high-dimensional test.
% It verifies complete 5-D/10-D quadratic recovery, archive sample counts,
% candidate composition, and upper/lower joint-vector dimensions.  It does
% not make an SMD performance claim or consume benchmark evaluations.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(projectRoot,'results_prediction_injection_tests', ...
        datestr(now,'yyyymmdd_HHMMSS'));
end
if nargin < 2 || isempty(totalDimensions)
    totalDimensions = [10,20];
end
validateattributes(totalDimensions,{'numeric'},{'vector','integer'});
assert(all(ismember(totalDimensions,[10,20])), ...
    'Only total dimensions 10 (DU=5, DL=5) and 20 (DU=10, DL=10) are supported.');
% This small test is called automatically by the formal runner.  If a
% previous attempt stopped before the report was saved, allow it to be
% rerun in the same smoke-test directory.
if exist(outputDir,'dir') ~= 7
    mkdir(outputDir);
end

specifications = struct( ...
    'totalDim',{10,20},'upperDim',{5,10},'lowerDim',{5,10}, ...
    'populationSize',{10,12},'injectCount',{5,6}, ...
    'sampleCount',{100,300});
specifications = specifications(ismember([specifications.totalDim],totalDimensions));
% DU=5 and DU=10 contain vectors/matrices of different widths.  MATLAB
% cannot place such nested structures in one structure *array*, so keep
% each dimension case in a named field of one scalar structure instead.
report = struct('DU5_DL5',struct(),'DU10_DL10',struct());
for specIndex = 1:numel(specifications)
    spec = specifications(specIndex);
    BI = getBLOPinfo('SMD',1,spec.totalDim);
    assert(BI.u_dim == spec.upperDim && BI.l_dim == spec.lowerDim);
    assert(4 + floor(3*log(BI.dim)) == spec.populationSize, ...
        'Original CMA-ES lambda is not the expected population size.');
    options = highDimOptions(BI,spec);
    [model,expectedPrediction] = knownCompleteQuadratic(BI,options,specIndex);
    assert(model.valid && model.coefficientCount == ...
        (BI.u_dim+1)*(BI.u_dim+2)/2, ...
        'Complete %d-D quadratic fit failed.',BI.u_dim);
    assert(size(model.H,1) == BI.u_dim && ...
        size(model.quadraticPairs,1) == BI.u_dim*(BI.u_dim+1)/2, ...
        'Hessian or quadratic-pair count is incorrect for DU=%d.',BI.u_dim);
    assert(norm(model.x_pred-expectedPrediction) < 1e-8, ...
        'Predicted stationary point is incorrect for DU=%d.',BI.u_dim);

    rawUpper = reshape(1:(spec.populationSize*BI.u_dim), ...
        BI.u_dim,spec.populationSize)' / 100;
    rawLower = zeros(spec.populationSize,BI.l_dim);
    CMA = struct('xmean',zeros(1,BI.dim),'sigma',0.25, ...
        'C',eye(BI.dim),'B',eye(BI.dim),'D',ones(1,BI.dim));
    state = initializePredictionInjectionState(BI,options);
    state.pending = struct('valid',true,'x_pred',expectedPrediction, ...
        'source_generation',10,'model',model);
    rngBefore = rng;
    [upper,lower,entry] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
        11,BI,options,state);
    assert(isequal(rngBefore,rng), ...
        'High-dimensional injection changed the global random sequence.');
    assert(size(upper,1) == spec.populationSize && ...
        size(upper,2) == BI.u_dim && size(lower,2) == BI.l_dim, ...
        'Joint candidate dimensions are incorrect.');
    assert(sum(strcmp(entry.source_labels,'normal')) == spec.populationSize/2);
    assert(sum(strcmp(entry.source_labels,'predicted_direct')) == 1);
    assert(sum(strcmp(entry.source_labels,'predicted_auxiliary')) == ...
        spec.populationSize/2-1);
    directIndex = spec.populationSize-spec.injectCount+1;
    assert(norm(upper(directIndex,:)-expectedPrediction) < 1e-8, ...
        'Direct prediction was not inserted at the expected candidate slot.');
    assert(size(entry.auxiliary_actual_joint,1) == spec.injectCount-1 && ...
        size(entry.auxiliary_actual_joint,2) == BI.dim, ...
        'Auxiliary joint samples have incorrect dimensions.');

    insufficient = model.selectedSamples(1:spec.sampleCount-1,:);
    insufficientModel = fitLocalUpperQuadratic(insufficient, ...
        zeros(1,BI.u_dim),10,options);
    assert(~insufficientModel.valid && ...
        strcmp(insufficientModel.reason,'insufficient_samples'), ...
        'Insufficient high-dimensional history did not disable prediction.');
    caseReport = struct('totalDim',BI.dim,'upperDim',BI.u_dim, ...
        'lowerDim',BI.l_dim,'populationSize',spec.populationSize, ...
        'injectCount',spec.injectCount,'sampleCount',spec.sampleCount, ...
        'coefficientCount',model.coefficientCount, ...
        'quadraticPairCount',size(model.quadraticPairs,1), ...
        'prediction',model.x_pred,'candidateEntry',entry);
    report.(sprintf('DU%d_DL%d',BI.u_dim,BI.l_dim)) = caseReport;
end
save(fullfile(outputDir,'highdim_prediction_injection_test_report.mat'),'report');
disp(report);
end

function options = highDimOptions(BI,spec)
BI.predictionInjection = struct('mode',sprintf('K%d',spec.injectCount), ...
    'predictionInjectCount',spec.injectCount, ...
    'candidatePolicy','direct_plus_auxiliary','logEnabled',true, ...
    'startGeneration',10,'historyGenerations',10,'useAllHistory',true, ...
    'sampleCount',spec.sampleCount,'auxiliaryRandomSeed',1991+spec.totalDim, ...
    'maxPredictionNormalizedDistance',Inf,'consistencyCheckEnabled',true);
options = predictionInjectionOptions(BI);
end

function [model,expectedPrediction] = knownCompleteQuadratic(BI,options,seed)
% Create a strictly concave complete quadratic with all cross terms nonzero.
rng(1000+seed,'twister');
upperDim = BI.u_dim;
sampleCount = options.sampleCount;
historyCount = sampleCount + 60;
meanBefore = zeros(1,upperDim);
randomMatrix = randn(upperDim);
H = -(randomMatrix'*randomMatrix + 0.5*eye(upperDim));
linear = linspace(-0.3,0.4,upperDim)';
expectedZ = -(H \ linear);
expectedPrediction = expectedZ';
z = randn(historyCount,upperDim);
UF = 2 + z*linear + 0.5*sum((z*H).*z,2);
% A full archive can contain many evaluated individuals per generation.
% Keep these synthetic samples available at generation 10, as in the real
% 5-D/10-D archive after enough normal CMA-ES populations have completed.
history = [10*ones(historyCount,1),(1:historyCount)',z,UF];
model = fitLocalUpperQuadratic(history,meanBefore,10,options);
end
