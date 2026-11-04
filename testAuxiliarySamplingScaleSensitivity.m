function report = testAuxiliarySamplingScaleSensitivity(outputDir,replications)
%TESTAUXILIARYSAMPLINGSCALESENSITIVITY Read-only sampling checks for eta.
% Verifies A0/A1 composition, the five-plus-one-plus-two layout, exact
% eta scaling of the auxiliary upper perturbations, and empirical standard
% deviations. No SMD objective or lower-level solver is called.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(projectRoot,'results_auxiliary_sampling_scale_tests', ...
        datestr(now,'yyyymmdd_HHMMSS'));
end
if nargin < 2 || isempty(replications), replications = 2000; end
validateattributes(replications,{'numeric'},{'scalar','integer','>=',1000});
if exist(outputDir,'dir') ~= 7, mkdir(outputDir); end

BI = getBLOPinfo('SMD',1,5);
assert(BI.u_dim == 2 && BI.l_dim == 3 && BI.dim == 5, ...
    'This smoke test requires SMD1 at DU=2, DL=3.');
populationSize = 4 + floor(3*log(BI.dim));
assert(populationSize == 8,'The original (2+3) population size must be 8.');

base = [1.0 0.2 0.1 0.0 0.0; ...
        0.4 1.1 0.0 0.1 0.0; ...
        0.2 0.1 1.0 0.0 0.1; ...
        0.0 0.2 0.1 0.9 0.0; ...
        0.1 0.0 0.2 0.1 0.8];
C = base*base';
[B,eigenvalues] = eig(C);
CMA = struct('xmean',(BI.xrange(1,:)+BI.xrange(2,:))/2, ...
    'sigma',0.02,'C',C,'B',B,'D',sqrt(diag(eigenvalues))');
centre = CMA.xmean(1:BI.u_dim);
rawUpper = repmat(centre,populationSize,1);
rawLower = repmat(CMA.xmean(BI.u_dim+1:end),populationSize,1);
etas = [0.25,0.5,0.75,1.0];
report = struct('passed',true,'replications',replications,'etas',etas);

% A0 and A1 must not depend on eta at all.
for modeIndex = 1:2
    K = modeIndex-1;
    reference = [];
    for eta = etas
        [upper,lower,entry] = onePopulation(CMA,rawUpper,rawLower,BI,K,eta);
        if isempty(reference)
            reference = struct('upper',upper,'lower',lower);
        else
            assert(isequal(upper,reference.upper) && isequal(lower,reference.lower), ...
                'A%d changed when only auxiliarySamplingScale changed.',K);
        end
        assert(all(strcmp(entry.source_labels(1:populationSize-K),'normal')), ...
            'A%d normal candidates were changed.',K);
    end
end

% A failed fit must never create a partially injected population, whatever
% eta has been configured for a later valid prediction.
for eta = etas
    options = etaOptions(BI,3,eta);
    unavailableState = initializePredictionInjectionState(BI,options);
    [upper,lower,entry] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
        11,BI,options,unavailableState);
    assert(isequal(upper,rawUpper) && isequal(lower,rawLower) && ...
        all(strcmp(entry.source_labels,'normal')) && ...
        entry.effective_injected_count == 0, ...
        'An unavailable prediction did not fall back to eight normal candidates.');
end

% Same stream seed gives a pointwise scale relation, stronger than a
% distributional check: d_eta must equal eta*d_eta1 before repair.
[~,~,entryOne] = onePopulation(CMA,rawUpper,rawLower,BI,3,1.0);
oneOffsets = entryOne.auxiliary_upper_raw_offsets;
assert(all(strcmp(entryOne.source_labels(1:5),'normal')) && ...
    strcmp(entryOne.source_labels{6},'predicted_direct') && ...
    all(strcmp(entryOne.source_labels(7:8),'predicted_auxiliary')), ...
    'The 5 normal + 1 direct + 2 auxiliary composition is invalid.');
assert(norm(entryOne.upper_after_injection(6,:)-centre) < 1e-12, ...
    'The direct prediction point moved when eta changed.');

targetOne = CMA.sigma^2*CMA.C(1:BI.u_dim,1:BI.u_dim);
for eta = etas
    [upper,lower,entry] = onePopulation(CMA,rawUpper,rawLower,BI,3,eta);
    assert(isequal(upper(1:5,:),rawUpper(1:5,:)) && ...
        isequal(lower(1:5,:),rawLower(1:5,:)), ...
        'Normal CMA-ES candidates changed for eta=%.2g.',eta);
    assert(norm(upper(6,:)-centre) < 1e-12, ...
        'The direct prediction changed for eta=%.2g.',eta);
    assert(norm(entry.auxiliary_upper_raw_offsets-eta*oneOffsets,'fro') < 1e-12, ...
        'Auxiliary perturbations do not scale exactly by eta=%.2g.',eta);
    assert(norm(entry.auxiliary_upper_target_covariance-eta^2*targetOne,'fro') < ...
        1e-12,'The logged covariance is incorrect for eta=%.2g.',eta);
    report.(etaField(eta)) = struct('targetCovariance', ...
        entry.auxiliary_upper_target_covariance, ...
        'samplingScale',entry.auxiliary_sampling_scale, ...
        'rawOffsets',entry.auxiliary_upper_raw_offsets);
end

% Empirical deviation check with the same private random-stream policy as
% the formal batch. It uses raw samples, so boundary repair cannot mask eta.
standardDeviations = zeros(numel(etas),1);
for etaIndex = 1:numel(etas)
    eta = etas(etaIndex);
    options = etaOptions(BI,3,eta);
    state = initializePredictionInjectionState(BI,options);
    state.pending = pendingPrediction(centre);
    samples = NaN(replications*2,BI.u_dim);
    rngBefore = rng;
    for i = 1:replications
        [~,~,entry,state] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
            11,BI,options,state);
        samples((2*i-1):(2*i),:) = entry.auxiliary_raw_joint(:,1:BI.u_dim);
    end
    assert(isequal(rngBefore,rng), ...
        'Auxiliary eta=%.2g changed the global random sequence.',eta);
    empiricalCovariance = cov(samples,1);
    standardDeviations(etaIndex) = sqrt(trace(empiricalCovariance)/BI.u_dim);
    report.(etaField(eta)).empiricalCovariance = empiricalCovariance;
    report.(etaField(eta)).empiricalRmsStd = standardDeviations(etaIndex);
end
ratios = standardDeviations / standardDeviations(end);
assert(max(abs(ratios-etas')) < 0.06, ...
    'Empirical auxiliary standard-deviation ratios do not match eta.');
report.empiricalRmsStdRatios = ratios;
save(fullfile(outputDir,'auxiliary_sampling_scale_test_report.mat'), ...
    'report');
disp(report);
end

function [upper,lower,entry] = onePopulation(CMA,rawUpper,rawLower,BI,K,eta)
options = etaOptions(BI,K,eta);
state = initializePredictionInjectionState(BI,options);
state.pending = pendingPrediction(CMA.xmean(1:BI.u_dim));
[upper,lower,entry] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
    11,BI,options,state);
end

function options = etaOptions(BI,K,eta)
BIrun = BI;
BIrun.predictionInjection = struct('mode',sprintf('K%d',K), ...
    'predictionInjectCount',K,'candidatePolicy','direct_plus_auxiliary', ...
    'auxiliarySamplingDistribution','cma_gaussian', ...
    'auxiliarySamplingScale',eta,'logEnabled',true, ...
    'startGeneration',10,'historyGenerations',10,'sampleCount',40, ...
    'auxiliaryRandomSeed',741852,'consistencyCheckEnabled',true);
options = predictionInjectionOptions(BIrun);
end

function pending = pendingPrediction(centre)
pending = struct('valid',true,'x_pred',centre, ...
    'source_generation',10,'model',struct());
end

function field = etaField(eta)
field = sprintf('eta_%03d',round(100*eta));
end
