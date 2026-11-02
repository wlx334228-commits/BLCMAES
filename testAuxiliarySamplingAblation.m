function report = testAuxiliarySamplingAblation(outputDir,replications)
%TESTAUXILIARYSAMPLINGABLATION Non-evaluating check for A0--A4 sampling.
% It validates candidate composition, auxiliary centres, target covariance,
% uniform-ellipsoid support, and preservation of the global RNG.  No SMD
% upper/lower objective is called in this test.

projectRoot = fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(projectRoot,'results_auxiliary_sampling_tests', ...
        datestr(now,'yyyymmdd_HHMMSS'));
end
if nargin < 2 || isempty(replications), replications = 4000; end
validateattributes(replications,{'numeric'},{'scalar','integer','>=',1000});
if exist(outputDir,'dir') ~= 7, mkdir(outputDir); end

BI = getBLOPinfo('SMD',1,5);
assert(BI.u_dim == 2 && BI.l_dim == 3 && BI.dim == 5, ...
    'This test requires SMD1 at DU=2, DL=3.');
populationSize = 4 + floor(3*log(BI.dim));
assert(populationSize == 8,'The original (2+3) population size must be 8.');

% A non-diagonal covariance gives A2/A4 an orientation that A3 must not
% reuse.  sigma is deliberately small so all synthetic samples are interior.
base = [1.0 0.2 0.1 0.0 0.0; ...
        0.4 1.1 0.0 0.1 0.0; ...
        0.2 0.1 1.0 0.0 0.1; ...
        0.0 0.2 0.1 0.9 0.0; ...
        0.1 0.0 0.2 0.1 0.8];
C = base * base';
[B,eigenvalues] = eig(C);
CMA = struct('xmean',(BI.xrange(1,:)+BI.xrange(2,:))/2, ...
    'sigma',0.02,'C',C,'B',B,'D',sqrt(diag(eigenvalues))');
upperCentre = CMA.xmean(1:BI.u_dim);
rawUpper = repmat(upperCentre,populationSize,1);
rawLower = repmat(CMA.xmean(BI.u_dim+1:end),populationSize,1);
specifications = samplingSpecifications();
report = struct('passed',true,'replications',replications);

for specIndex = 1:numel(specifications)
    spec = specifications(specIndex);
    options = ablationOptions(BI,spec,9000+specIndex);
    % A failed/not-yet-available quadratic fit must leave the full normal
    % population untouched, regardless of the configured injection count.
    unavailableState = initializePredictionInjectionState(BI,options);
    [unavailableUpper,unavailableLower,unavailableEntry] = ...
        applyPredictionInjection(CMA,rawUpper,rawLower,11,BI,options, ...
        unavailableState);
    assert(isequal(unavailableUpper,rawUpper) && ...
        isequal(unavailableLower,rawLower) && ...
        all(strcmp(unavailableEntry.source_labels,'normal')) && ...
        unavailableEntry.effective_injected_count == 0, ...
        '%s did not fall back to eight normal candidates after prediction failure.', ...
        spec.id);
    state = initializePredictionInjectionState(BI,options);
    state.pending = struct('valid',true,'x_pred',upperCentre, ...
        'source_generation',10,'model',struct());
    rngBefore = rng;
    [upper,lower,entry] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
        11,BI,options,state);
    assert(isequal(rngBefore,rng), ...
        '%s changed the global random sequence.',spec.id);
    assert(sum(strcmp(entry.source_labels,'normal')) == spec.normalCount && ...
        sum(strcmp(entry.source_labels,'predicted_direct')) == spec.directCount && ...
        sum(strcmp(entry.source_labels,'predicted_auxiliary')) == spec.auxiliaryCount, ...
        '%s candidate composition is invalid.',spec.id);
    assert(isequal(size(upper),[populationSize,BI.u_dim]) && ...
        isequal(size(lower),[populationSize,BI.l_dim]), ...
        '%s candidate dimensions are invalid.',spec.id);
    if spec.directCount == 1
        directIndex = populationSize-spec.injectCount+1;
        assert(norm(upper(directIndex,:)-upperCentre) < 1e-12, ...
            '%s did not insert the direct prediction point.',spec.id);
    end

    caseReport = struct('predictionFailureFallbackPassed',true, ...
        'candidateEntry',entry, ...
        'normalCandidateCount',spec.normalCount, ...
        'directCandidateCount',spec.directCount, ...
        'auxiliaryCandidateCount',spec.auxiliaryCount, ...
        'auxiliarySamplingDistribution',spec.distribution);
    if spec.auxiliaryCount > 0
        assert(norm(entry.auxiliary_upper_center-upperCentre) < 1e-12, ...
            '%s has an incorrect auxiliary upper centre.',spec.id);
        [empiricalMean,empiricalCovariance,maximumWhitenedRadius, ...
            allInterior] = empiricalAuxiliaryStatistics(CMA,BI,options, ...
            upperCentre,rawUpper,rawLower,replications);
        target = expectedUpperCovariance(CMA,BI.u_dim,spec.distribution);
        meanRelativeError = norm(empiricalMean-upperCentre) / ...
            max(sqrt(trace(target)),eps);
        covarianceRelativeError = norm(empiricalCovariance-target,'fro') / ...
            max(norm(target,'fro'),eps);
        assert(meanRelativeError < 0.08, ...
            '%s auxiliary mean test failed (relative error %.3g).', ...
            spec.id,meanRelativeError);
        assert(covarianceRelativeError < 0.13, ...
            '%s auxiliary covariance test failed (relative error %.3g).', ...
            spec.id,covarianceRelativeError);
        assert(allInterior, ...
            '%s synthetic auxiliary samples unexpectedly hit a boundary.',spec.id);
        if strcmp(spec.distribution,'uniform_ellipsoid')
            assert(maximumWhitenedRadius <= sqrt(BI.u_dim+2)+1e-10, ...
                'A4 sampled outside its prescribed uniform ellipsoid.');
        end
        assert(norm(entry.auxiliary_upper_target_covariance-target,'fro') < ...
            1e-12,'Logged target covariance is incorrect for %s.',spec.id);
        caseReport.empiricalMean = empiricalMean;
        caseReport.empiricalCovariance = empiricalCovariance;
        caseReport.targetCovariance = target;
        caseReport.meanRelativeError = meanRelativeError;
        caseReport.covarianceRelativeError = covarianceRelativeError;
        caseReport.maximumWhitenedRadius = maximumWhitenedRadius;
    end
    report.(spec.id) = caseReport;
end
% A small real SMD1 run verifies the same candidates enter lower search,
% receive real upper evaluations, and are the exact coordinates used by
% the later CMA-ES parent update.  This is intentionally separate from the
% 20-run formal experiment and uses a small evaluation cap only.
report.integration = integrationCheck(BI,specifications);
save(fullfile(outputDir,'auxiliary_sampling_ablation_test_report.mat'), ...
    'report','specifications');
disp(report);
end

function specifications = samplingSpecifications()
specifications = [ ...
    makeSpec('A0',0,8,0,0,'cma_gaussian'); ...
    makeSpec('A1',1,7,1,0,'cma_gaussian'); ...
    makeSpec('A2',3,5,1,2,'cma_gaussian'); ...
    makeSpec('A3',3,5,1,2,'isotropic_gaussian'); ...
    makeSpec('A4',3,5,1,2,'uniform_ellipsoid')];
end

function spec = makeSpec(id,injectCount,normalCount,directCount, ...
    auxiliaryCount,distribution)
assert(normalCount+directCount+auxiliaryCount == 8);
spec = struct('id',id,'injectCount',injectCount, ...
    'normalCount',normalCount,'directCount',directCount, ...
    'auxiliaryCount',auxiliaryCount,'distribution',distribution);
end

function options = ablationOptions(BI,spec,seed)
options = predictionInjectionOptions(ablationBI(BI,spec,seed));
end

function BIout = ablationBI(BI,spec,seed)
BIout = BI;
BIout.enablePredictionCorrection = false;
BIout.enableIndividualPredictionCorrection = false;
BIout.predictionInjection = struct('mode',sprintf('K%d',spec.injectCount), ...
    'predictionInjectCount',spec.injectCount, ...
    'candidatePolicy','direct_plus_auxiliary', ...
    'auxiliarySamplingDistribution',spec.distribution, ...
    'logEnabled',true,'startGeneration',10,'historyGenerations',10, ...
    'sampleCount',40,'auxiliaryRandomSeed',seed, ...
    'consistencyCheckEnabled',true);
end

function integration = integrationCheck(BI,specifications)
integration = struct();
for specIndex = 1:numel(specifications)
    spec = specifications(specIndex);
    BIrun = ablationBI(BI,spec,10000+specIndex);
    BIrun.UmaxFEs = 128;
    BIrun.UmaxImprFEs = 350;
    % getBLOPinfo supplies tolerances and bounds, whereas the normal batch
    % runner supplies these original (2+3) lower-level budgets explicitly.
    % The real-integration smoke test must do the same.
    BIrun.LmaxFEs = 250;
    BIrun.LmaxImprFEs = 25;
    BIrun.u_ftol = 0;
    rng(123,'twister');
    ins = BLCMAES(BIrun);
    log = ins.predictionInjectionLog;
    assert(all(arrayfun(@(item) isequal(item.upper_entering_lower_search, ...
        item.upper_after_injection),log)), ...
        '%s lower-level search did not receive the actual injected points.',spec.id);
    for generation = 1:numel(log)
        item = log(generation);
        if item.mean_update_applied
            assert(item.consistency_checked && item.consistency_passed, ...
                '%s CMA-ES update consistency check failed.',spec.id);
            assert(isequal(item.selected_parent_joint, ...
                item.joint_for_cma_update(item.selected_parent_indices,:)), ...
                '%s used a position different from the evaluated parent.',spec.id);
        end
        if item.effective_injected_count > 0
            assert(strcmp(item.auxiliary_sampling_distribution, ...
                spec.distribution), ...
                '%s logged an incorrect auxiliary distribution.',spec.id);
            assert(sum(strcmp(item.source_labels,'normal')) == spec.normalCount && ...
                sum(strcmp(item.source_labels,'predicted_direct')) == ...
                    spec.directCount && ...
                sum(strcmp(item.source_labels,'predicted_auxiliary')) == ...
                    spec.auxiliaryCount, ...
                '%s injection composition changed during real SMD1 testing.',spec.id);
        end
    end
    integration.(spec.id) = struct('UF',ins.UF,'LF',ins.LF, ...
        'UFE',ins.UFEs,'LFE',ins.LFEs, ...
        'injectedGenerations',sum([log.effective_injected_count] > 0));
end
end

function [average,covariance,maximumWhitenedRadius,allInterior] = ...
    empiricalAuxiliaryStatistics(CMA,BI,options,upperCentre,rawUpper, ...
    rawLower,replications)
state = initializePredictionInjectionState(BI,options);
state.pending = struct('valid',true,'x_pred',upperCentre, ...
    'source_generation',10,'model',struct());
samples = NaN(replications*2,BI.u_dim);
allInterior = true;
for i = 1:replications
    [~,~,entry,state] = applyPredictionInjection(CMA,rawUpper,rawLower, ...
        11,BI,options,state);
    rows = (2*i-1):(2*i);
    samples(rows,:) = entry.auxiliary_raw_joint(:,1:BI.u_dim);
    allInterior = allInterior && ~any(entry.auxiliary_boundary_repaired(:));
end
average = mean(samples,1);
covariance = cov(samples,1);
if strcmp(options.auxiliarySamplingDistribution,'uniform_ellipsoid')
    Cuu = CMA.C(1:BI.u_dim,1:BI.u_dim);
    factor = symmetricSquareRootForTest(Cuu);
    whitened = bsxfun(@minus,samples,upperCentre) / (CMA.sigma*factor');
    maximumWhitenedRadius = max(sqrt(sum(whitened.^2,2)));
else
    maximumWhitenedRadius = NaN;
end
end

function target = expectedUpperCovariance(CMA,upperDim,distribution)
Cuu = CMA.C(1:upperDim,1:upperDim);
switch distribution
    case {'cma_gaussian','uniform_ellipsoid'}
        target = CMA.sigma^2 * Cuu;
    case 'isotropic_gaussian'
        target = CMA.sigma^2 * trace(Cuu) / upperDim * eye(upperDim);
    otherwise
        error('Unknown test distribution: %s',distribution);
end
end

function factor = symmetricSquareRootForTest(covariance)
[vectors,values] = eig((covariance+covariance')/2);
factor = real(vectors*diag(sqrt(max(real(diag(values)),0))));
end
