function [upper,lower,entry,state] = applyPredictionInjection( ...
    CMA,rawUpper,rawLower,generation,BI,options,state)
%APPLYPREDICTIONINJECTION Construct K=0,...,lambda evaluated candidates.
% All main-distribution samples are first drawn exactly as in the original
% code. When a pending prediction is valid, the final K slots are replaced
% before any lower-level or upper-level evaluation. The legacy
% direct-plus-auxiliary policy uses one direct prediction and K-1 auxiliary
% joint samples. The auxiliary-only policy uses K auxiliary joint samples,
% with the prediction only as their upper sampling centre.

populationSize = size(rawUpper,1);
assert(size(rawUpper,2) == BI.u_dim && size(rawLower,1) == populationSize, ...
    'BLCMAES:InvalidSample','Unexpected joint-sample dimensions.');
if isempty(state)
    state = initializePredictionInjectionState(BI,options);
end
if options.predictionInjectCount > populationSize
    error('BLCMAES:PredictionInjectionCount', ...
        'predictionInjectCount=%d exceeds population size %d.', ...
        options.predictionInjectCount,populationSize);
end
upper = rawUpper;
lower = rawLower;
entry = createEntry(CMA,rawUpper,rawLower,generation,BI,options);

if ~options.enabled || generation <= options.startGeneration || ...
        ~state.pending.valid
    entry.injection_reason = injectionReason(options,generation,state);
    entry.upper_after_injection = upper;
    entry.lower_after_injection = lower;
    entry.joint_after_injection = [upper,lower];
    return;
end

xPredRequested = state.pending.x_pred;
if ~isreal(xPredRequested) || numel(xPredRequested) ~= BI.u_dim || ...
        any(~isfinite(xPredRequested))
    entry.injection_reason = 'invalid_pending_prediction';
    entry.upper_after_injection = upper;
    entry.lower_after_injection = lower;
    entry.joint_after_injection = [upper,lower];
    return;
end

injectCount = options.predictionInjectCount;
injectedStart = populationSize-injectCount+1;
if options.usesDirectPrediction
    directIndex = injectedStart;
    auxiliaryIndices = (directIndex+1):populationSize;
    predictionReference = rawUpper(directIndex,:);
else
    directIndex = NaN;
    auxiliaryIndices = injectedStart:populationSize;
    % The prediction itself is not evaluated in auxiliary-only mode.  This
    % reference is used only by the existing boundary-repair rule to form a
    % feasible auxiliary-distribution centre.
    predictionReference = rawUpper(auxiliaryIndices(1),:);
end
[xPredActual,predictionCentreRepair] = repairUpperCandidate( ...
    xPredRequested,predictionReference,BI);
entry.prediction_valid_used = true;
entry.prediction_source_generation = state.pending.source_generation;
entry.x_pred_requested = xPredRequested;
entry.x_pred_actual = xPredActual;
entry.prediction_center_requested_step = xPredRequested - ...
    entry.cma_mean_before_sampling(1:BI.u_dim);
entry.prediction_center_actual_step = xPredActual - ...
    entry.cma_mean_before_sampling(1:BI.u_dim);
entry.injection_reason = 'prediction_injected';

% Draw all auxiliary members before changing any population slot.  If the
% requested covariance decomposition is numerically unusable, this entire
% generation deliberately falls back to the already-drawn normal population
% instead of evaluating a partially injected population.
auxiliaryRaw = NaN(0,BI.dim);
auxiliaryActual = NaN(0,BI.dim);
auxiliaryRepair = false(0,BI.dim);
samplingInfo = struct();
if ~isempty(auxiliaryIndices)
    if isempty(state.auxiliaryStream)
        error('BLCMAES:AuxiliaryRandomStream', ...
            'This policy requires its private auxiliary sampling stream.');
    end
    auxiliaryMean = CMA.xmean;
    % Use the bounded prediction only as the auxiliary upper centre. The
    % lower centre remains the main CMA-ES lower mean. In auxiliary-only
    % mode this is the sole use of xPredActual: no population member is set
    % equal to it directly.
    auxiliaryMean(1:BI.u_dim) = xPredActual;
    try
        [auxiliaryRaw,samplingInfo] = sampleAuxiliaryJointCandidates( ...
            CMA,auxiliaryMean,numel(auxiliaryIndices),BI,options, ...
            state.auxiliaryStream);
    catch samplingException
        if strncmp(samplingException.identifier,'BLCMAES:Auxiliary', ...
                numel('BLCMAES:Auxiliary'))
            entry.prediction_valid_used = false;
            entry.injection_reason = 'auxiliary_sampling_failed';
            entry.upper_after_injection = upper;
            entry.lower_after_injection = lower;
            entry.joint_after_injection = [upper,lower];
            return;
        end
        rethrow(samplingException);
    end
end

if options.usesDirectPrediction
    upper(directIndex,:) = xPredActual;
    entry.source_labels{directIndex} = 'predicted_direct';
    entry.direct_index = directIndex;
    entry.direct_boundary_repaired = predictionCentreRepair;
    entry.direct_requested_step = entry.prediction_center_requested_step;
    entry.direct_actual_step = entry.prediction_center_actual_step;
else
    % Kept separate from direct_boundary_repaired so boundary-repair totals
    % remain interpretable when no direct candidate exists.
    entry.prediction_center_boundary_repaired = predictionCentreRepair;
end
if ~isempty(auxiliaryIndices)
    auxiliaryActual = NaN(numel(auxiliaryIndices),BI.dim);
    auxiliaryRepair = false(numel(auxiliaryIndices),BI.dim);
    for j = 1:numel(auxiliaryIndices)
        [auxiliaryActual(j,:),auxiliaryRepair(j,:)] = repairJointCandidate( ...
            auxiliaryRaw(j,:),auxiliaryMean,BI);
        index = auxiliaryIndices(j);
        upper(index,:) = auxiliaryActual(j,1:BI.u_dim);
        lower(index,:) = auxiliaryActual(j,BI.u_dim+1:end);
        entry.source_labels{index} = 'predicted_auxiliary';
    end
    entry.auxiliary_index = auxiliaryIndices(1);
    entry.auxiliary_indices = auxiliaryIndices;
    entry.auxiliary_mean = repmat(auxiliaryMean,numel(auxiliaryIndices),1);
    entry.auxiliary_raw_joint = auxiliaryRaw;
    entry.auxiliary_actual_joint = auxiliaryActual;
    entry.auxiliary_boundary_repaired = auxiliaryRepair;
    entry.auxiliary_upper_center = samplingInfo.upperCenter;
    entry.auxiliary_upper_target_covariance = samplingInfo.upperTargetCovariance;
    entry.auxiliary_lower_conditional_covariance = ...
        samplingInfo.lowerConditionalCovariance;
    entry.auxiliary_isotropic_variance = samplingInfo.isotropicVariance;
end

entry.upper_after_injection = upper;
entry.lower_after_injection = lower;
entry.joint_after_injection = [upper,lower];
entry.injected_indices = find(~strcmp(entry.source_labels,'normal'))';
entry.injected_count = numel(entry.injected_indices);
entry.effective_injected_count = entry.injected_count;
entry.normal_sample_count = populationSize-entry.injected_count;
end

function [jointSamples,info] = sampleAuxiliaryJointCandidates( ...
    CMA,auxiliaryMean,sampleCount,BI,options,stream)
% Construct the auxiliary joint population without touching the global RNG.
% For the two alternative upper distributions, lower variables are sampled
% from the original CMA-ES Gaussian conditional on the new upper location.
% Thus the lower marginal/conditional search relationship is retained and
% the ablation changes only the upper distribution around x_pred.

upperDim = BI.u_dim;
lowerDim = BI.l_dim;
jointDim = BI.dim;
C = (CMA.C + CMA.C') / 2;
Cuu = C(1:upperDim,1:upperDim);
Cul = C(1:upperDim,upperDim+1:end);
Cll = C(upperDim+1:end,upperDim+1:end);
if rcond(Cuu) <= eps
    error('BLCMAES:AuxiliaryCovariance', ...
        'The upper CMA-ES covariance block is numerically singular.');
end
conditionalMap = Cuu \ Cul;
conditionalLowerCovariance = CMA.sigma^2 * (Cll - Cul' * conditionalMap);
conditionalLowerCovariance = ...
    (conditionalLowerCovariance + conditionalLowerCovariance') / 2;
upperTargetCovariance = CMA.sigma^2 * Cuu;
upperCenter = auxiliaryMean(1:upperDim);
lowerCenter = auxiliaryMean(upperDim+1:end);
isotropicVariance = CMA.sigma^2 * trace(Cuu) / upperDim;
jointSamples = NaN(sampleCount,jointDim);

switch options.auxiliarySamplingDistribution
    case 'cma_gaussian'
        % Preserve the historic B2/K implementation exactly: one private
        % row-vector normal draw per auxiliary member, transformed through
        % the full joint CMA-ES eigensystem.
        for j = 1:sampleCount
            jointSamples(j,:) = auxiliaryMean + CMA.sigma * ...
                randn(stream,1,jointDim) .* CMA.D * CMA.B';
        end
    case {'isotropic_gaussian','uniform_ellipsoid'}
        upperFactor = symmetricSquareRoot(Cuu,'upper CMA-ES covariance');
        lowerFactor = symmetricSquareRoot( ...
            conditionalLowerCovariance,'conditional lower covariance');
        for j = 1:sampleCount
            normalDraw = randn(stream,1,jointDim);
            switch options.auxiliarySamplingDistribution
                case 'isotropic_gaussian'
                    upper = upperCenter + sqrt(isotropicVariance) * ...
                        normalDraw(1:upperDim);
                case 'uniform_ellipsoid'
                    directionNorm = norm(normalDraw(1:upperDim));
                    if directionNorm <= 0
                        % A zero normal vector has probability zero, but the
                        % deterministic fallback keeps the routine total.
                        direction = [1,zeros(1,upperDim-1)];
                    else
                        direction = normalDraw(1:upperDim) / directionNorm;
                    end
                    radius = rand(stream,1)^(1/upperDim);
                    upper = upperCenter + sqrt(upperDim+2) * CMA.sigma * ...
                        radius * direction * upperFactor';
            end
            upperOffset = upper - upperCenter;
            lowerConditionalMean = lowerCenter + upperOffset * conditionalMap;
            lower = lowerConditionalMean + normalDraw(upperDim+1:end) * ...
                lowerFactor';
            jointSamples(j,:) = [upper,lower];
        end
        if strcmp(options.auxiliarySamplingDistribution,'isotropic_gaussian')
            upperTargetCovariance = isotropicVariance * eye(upperDim);
        end
    otherwise
        error('BLCMAES:AuxiliaryDistribution', ...
            'Unsupported auxiliary sampling distribution: %s', ...
            options.auxiliarySamplingDistribution);
end

info = struct('upperCenter',upperCenter, ...
    'upperTargetCovariance',upperTargetCovariance, ...
    'lowerConditionalCovariance',conditionalLowerCovariance, ...
    'isotropicVariance',isotropicVariance);
end

function factor = symmetricSquareRoot(covariance,label)
covariance = (covariance + covariance') / 2;
[vectors,values] = eig(covariance);
eigenvalues = real(diag(values));
scale = max(1,max(abs(eigenvalues)));
if any(~isfinite(eigenvalues)) || min(eigenvalues) < -1e-10 * scale
    error('BLCMAES:AuxiliaryCovariance', ...
        '%s is not positive semidefinite.',label);
end
factor = real(vectors * diag(sqrt(max(eigenvalues,0))));
end

function reason = injectionReason(options,generation,state)
if ~options.enabled
    reason = 'A0';
elseif generation <= options.startGeneration
    reason = 'warmup';
elseif ~isfield(state,'pending') || ~state.pending.valid
    reason = 'invalid_prediction';
else
    reason = 'not_injected';
end
end

function [actual,repaired] = repairUpperCandidate(requested,reference,BI)
actual = requested;
high = actual > BI.u_ub;
low = actual < BI.u_lb;
actual(high) = (reference(high)+BI.u_ub(high))/2;
actual(low) = (reference(low)+BI.u_lb(low))/2;
repaired = high | low;
end

function [actual,repaired] = repairJointCandidate(requested,reference,BI)
actual = requested;
upper = actual(1:BI.u_dim);
lower = actual(BI.u_dim+1:end);
upperHigh = upper > BI.u_ub;
upperLow = upper < BI.u_lb;
lowerHigh = lower > BI.l_ub;
lowerLow = lower < BI.l_lb;
upper(upperHigh) = (reference(upperHigh)+BI.u_ub(upperHigh))/2;
upper(upperLow) = (reference(upperLow)+BI.u_lb(upperLow))/2;
lowerReference = reference(BI.u_dim+1:end);
lower(lowerHigh) = (lowerReference(lowerHigh)+BI.l_ub(lowerHigh))/2;
lower(lowerLow) = (lowerReference(lowerLow)+BI.l_lb(lowerLow))/2;
actual = [upper,lower];
repaired = [upperHigh | upperLow,lowerHigh | lowerLow];
end

function entry = createEntry(CMA,rawUpper,rawLower,generation,BI,options)
populationSize = size(rawUpper,1);
upperDim = BI.u_dim;
jointDim = BI.dim;
entry = struct('generation',generation,'mode',options.mode, ...
    'candidate_policy',options.candidatePolicy, ...
    'auxiliary_sampling_distribution',options.auxiliarySamplingDistribution, ...
    'prediction_inject_count',options.predictionInjectCount, ...
    'random_seed',options.randomSeed, ...
    'population_size',populationSize, ...
    'cma_mean_before_sampling',CMA.xmean, ...
    'cma_mean_after_update',NaN(1,jointDim), ...
    'sigma_before_sampling',CMA.sigma,'sigma_after_update',NaN, ...
    'C_before_sampling',CMA.C,'C_after_update',NaN(jointDim), ...
    'cond_C_before_sampling',cond(CMA.C), ...
    'cond_C_after_update',NaN,'eig_min_C_after_update',NaN, ...
    'eig_max_C_after_update',NaN, ...
    'prediction_valid_used',false,'prediction_source_generation',NaN, ...
    'x_pred_requested',NaN(1,upperDim),'x_pred_actual',NaN(1,upperDim), ...
    'direct_index',NaN,'auxiliary_index',NaN,'auxiliary_indices',zeros(1,0), ...
    'source_labels',{repmat({'normal'},populationSize,1)}, ...
    'normal_sample_count',populationSize,'requested_injected_count', ...
    options.predictionInjectCount,'effective_injected_count',0, ...
    'injected_count',0, ...
    'injected_indices',zeros(1,0),'injection_reason','not_checked', ...
    'direct_boundary_repaired',false(1,upperDim), ...
    'prediction_center_boundary_repaired',false(1,upperDim), ...
    'prediction_center_requested_step',NaN(1,upperDim), ...
    'prediction_center_actual_step',NaN(1,upperDim), ...
    'direct_requested_step',NaN(1,upperDim), ...
    'direct_actual_step',NaN(1,upperDim), ...
    'auxiliary_mean',NaN(0,jointDim), ...
    'auxiliary_raw_joint',NaN(0,jointDim), ...
    'auxiliary_actual_joint',NaN(0,jointDim), ...
    'auxiliary_boundary_repaired',false(0,jointDim), ...
    'auxiliary_upper_center',NaN(1,upperDim), ...
    'auxiliary_upper_target_covariance',NaN(upperDim), ...
    'auxiliary_lower_conditional_covariance',NaN(BI.l_dim), ...
    'auxiliary_isotropic_variance',NaN, ...
    'upper_before_injection',rawUpper,'lower_before_injection',rawLower, ...
    'joint_before_injection',[rawUpper,rawLower], ...
    'upper_after_injection',rawUpper,'lower_after_injection',rawLower, ...
    'joint_after_injection',[rawUpper,rawLower], ...
    'upper_entering_lower_search',NaN(populationSize,upperDim), ...
    'joint_after_true_evaluation',NaN(populationSize,jointDim), ...
    'upper_objective_after_true_evaluation',NaN(populationSize,1), ...
    'elite_after_refinement_joint',NaN(1,jointDim), ...
    'elite_after_refinement_UF',NaN, ...
    'external_refinement_upper_evaluated',false, ...
    'external_refinement_joint',NaN(1,jointDim), ...
    'external_refinement_UF',NaN, ...
    'external_refinement_is_new_best',false, ...
    'joint_for_cma_update',NaN(populationSize,jointDim), ...
    'population_UF',NaN(populationSize,1),'population_fit',NaN(populationSize,1), ...
    'fitness_rank',NaN(populationSize,1), ...
    'generation_best_index',NaN,'generation_best_source','', ...
    'direct_is_generation_best',false,'auxiliary_is_generation_best',false, ...
    'direct_fitness_ranks',NaN(0,1),'auxiliary_fitness_ranks',NaN(0,1), ...
    'new_best_so_far',false,'best_so_far_UF',NaN, ...
    'selected_parent_indices',zeros(1,0), ...
    'selected_parent_joint',NaN(0,jointDim), ...
    'selected_parent_sources',{{}}, ...
    'direct_parent_selected',false,'auxiliary_parent_selected',false, ...
    'direct_parent_count',0,'auxiliary_parent_count',0, ...
    'normal_parent_count',0, ...
    'parent_raw_steps',NaN(0,jointDim), ...
    'parent_clipped_steps',NaN(0,jointDim), ...
    'parent_whitened_norms',NaN(0,1), ...
    'parent_step_scales',NaN(0,1), ...
    'injected_step_limited',false(1,0), ...
    'UFE',NaN,'LFE_total',NaN,'mean_update_applied',false, ...
    'termination_reason','','consistency_checked',false, ...
    'consistency_passed',NaN,'consistency_message','not_checked', ...
    'next_prediction_valid',false,'next_prediction_reason','not_fitted', ...
    'next_x_pred',NaN(1,upperDim),'next_prediction_model',struct());
end
