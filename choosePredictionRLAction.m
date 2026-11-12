function [rl,decision] = choosePredictionRLAction(rl,CMA,xPred,history,model,BI,options)
%CHOOSEPREDICTIONRLACTION Select one shared eta/delta action before sampling.
% No true upper/lower evaluation is performed here.  Every state feature is
% available before the two auxiliary candidates are generated.

decision = emptyPredictionRLDecision(BI.u_dim);
if isempty(rl) || ~isfield(rl,'enabled') || ~rl.enabled
    decision.reason = 'rl_not_initialized';
    return;
end
try
    [directions,eigenvalues] = upperEigenSystem(CMA.C,BI.u_dim, ...
        options.rlCovarianceEigenvalueFloor);
    cmaCovariance = CMA.sigma^2 * (CMA.C(1:BI.u_dim,1:BI.u_dim) + ...
        CMA.C(1:BI.u_dim,1:BI.u_dim)')/2;
    cmaScale = sqrt(max(trace(cmaCovariance)/BI.u_dim,eps));
    meanUpper = CMA.xmean(1:BI.u_dim);
    distanceToMean = norm(xPred-meanUpper) / cmaScale;
    nearestDistance = predictionNearestDistance(xPred,history,model) / cmaScale;
    eigenRatio = log(max(eigenvalues(1),eps)/max(eigenvalues(2),eps));
    curvatureContrast = hessianContrast(model,directions);
    rawState = [clipFinite(distanceToMean,10); ...
        clipFinite(nearestDistance,10); ...
        clipFinite(2*rl.previousDirectPairRate-1,1); ...
        clipFinite(eigenRatio,8); ...
        clipFinite(curvatureContrast,8); ...
        clipFinite(rl.previousAuxDirectionPerformance(1),1); ...
        clipFinite(rl.previousAuxDirectionPerformance(2),1); ...
        (rl.previousEta-options.rlFixedEta) / ...
            max(options.rlEtaMax-options.rlEtaMin,eps); ...
        rl.previousDelta/max(options.rlDeltaMax,eps); ...
        clipFinite(rl.previousReward/options.rlRewardClip,1)];
    [normalized,rl] = normaliseAndAccumulateState(rawState,rl);
    [hidden,output] = actorForward(rl.actor,normalized);
    mu = output(1:2);
    logStd = min(max(output(3:4),log(options.rlMinActionStd)), ...
        log(options.rlMaxActionStd));
    standardDeviation = exp(logStd);
    latent = mu + standardDeviation .* randn(rl.actorStream,2,1);
    [eta,delta,etaLearned,deltaLearned] = actionFromLatent(latent,options);
    rlCovariance = adjustedRLCovariance(directions,eigenvalues,CMA.sigma, ...
        eta,delta,options);
    rlEigenvalues = sort(real(eig(rlCovariance)),'descend');
    decision = struct('valid',true,'reason','valid','mode',options.rlMode, ...
        'state_raw',rawState,'state_normalized',normalized, ...
        'actor_hidden',hidden,'actor_mean',mu,'actor_log_std',logStd, ...
        'actor_std',standardDeviation,'action_latent',latent, ...
        'eta',eta,'delta',delta,'eta_learned',etaLearned, ...
        'delta_learned',deltaLearned,'upperCovariance',rlCovariance, ...
        'cmaEigenvalues',eigenvalues,'rlEigenvalues',rlEigenvalues, ...
        'trace_ratio',trace(rlCovariance)/max(trace(cmaCovariance),eps), ...
        'covariance_condition',max(rlEigenvalues)/max(min(rlEigenvalues),eps), ...
        'covariance_regularized',false,'principalDirections',directions);
    rl.pendingDecision = decision;
catch exception
    % Keep the exact safe-fallback cause in the MAT log and smoke-test
    % message.  It never changes sampling or invokes an evaluation.
    decision.reason = sprintf('action_failed:%s:%s', ...
        exception.identifier,exception.message);
    rl.pendingDecision = emptyPredictionRLDecision(BI.u_dim);
end
end

function [directions,eigenvalues] = upperEigenSystem(C,upperDim,eigenFloor)
Cuu = (C(1:upperDim,1:upperDim)+C(1:upperDim,1:upperDim)')/2;
[directions,diagonal] = eig(Cuu);
[eigenvalues,order] = sort(real(diag(diagonal)),'descend');
directions = real(directions(:,order));
if any(~isfinite(eigenvalues)) || min(eigenvalues) <= eigenFloor
    error('BLCMAES:RLCovariance','Invalid CMA-ES upper covariance eigensystem.');
end
end

function distance = predictionNearestDistance(xPred,history,model)
distance = NaN;
if isstruct(model) && isfield(model,'predictionToNearestSample') && ...
        isfinite(model.predictionToNearestSample)
    distance = model.predictionToNearestSample;
elseif ~isempty(history)
    upperDim = numel(xPred);
    positions = history(:,3:(upperDim+2));
    distance = min(sqrt(sum(bsxfun(@minus,positions,xPred).^2,2)));
end
if ~isfinite(distance), distance = 0; end
end

function contrast = hessianContrast(model,directions)
contrast = 0;
if ~isstruct(model) || ~isfield(model,'H') || ...
        any(~isfinite(model.H(:)))
    return;
end
curvature = real(diag(directions' * (-model.H) * directions));
if numel(curvature) >= 2 && all(curvature > 0) && all(isfinite(curvature))
    contrast = log(curvature(1)/curvature(2));
end
end

function [normalised,rl] = normaliseAndAccumulateState(raw,rl)
if rl.stateCount < 2
    normalised = raw;
else
    variance = rl.stateM2 / max(rl.stateCount-1,1);
    normalised = (raw-rl.stateMean) ./ sqrt(max(variance,1e-8));
    normalised = min(max(normalised,-5),5);
end
rl.stateCount = rl.stateCount + 1;
delta = raw-rl.stateMean;
rl.stateMean = rl.stateMean + delta/rl.stateCount;
rl.stateM2 = rl.stateM2 + delta .* (raw-rl.stateMean);
end

function [hidden,output] = actorForward(actor,state)
hidden = tanh(actor.W1*state + actor.b1);
output = actor.W2*hidden + actor.b2;
end

function [eta,delta,etaLearned,deltaLearned] = actionFromLatent(latent,options)
etaLearned = any(strcmp(options.rlMode,{'eta_only','joint'}));
deltaLearned = any(strcmp(options.rlMode,{'delta_only','joint'}));
if etaLearned
    eta = options.rlEtaMin + (options.rlEtaMax-options.rlEtaMin) / ...
        (1+exp(-min(max(latent(1),-20),20)));
else
    eta = options.rlFixedEta;
end
if deltaLearned
    delta = options.rlDeltaMax*tanh(latent(2));
else
    delta = 0;
end
end

function covariance = adjustedRLCovariance(directions,eigenvalues,sigma,eta,delta,options)
rawEigenvalues = eigenvalues(:) .* [exp(delta); exp(-delta)];
rawEigenvalues = max(rawEigenvalues, ...
    options.rlCovarianceEigenvalueFloor*max(eigenvalues));
% The bounded delta already controls anisotropy.  This final cap prevents a
% pathological CMA state from producing a non-positive numerical covariance.
largest = max(rawEigenvalues);
rawEigenvalues = max(rawEigenvalues,largest/options.rlCovarianceMaxCondition);
shape = directions*diag(rawEigenvalues)*directions';
baseTrace = sum(eigenvalues);
shape = baseTrace/max(trace(shape),eps)*shape;
covariance = eta^2*sigma^2*(shape+shape')/2;
if any(~isfinite(covariance(:))) || min(real(eig(covariance))) <= 0
    error('BLCMAES:RLCovariance','The RL auxiliary covariance is invalid.');
end
end

function value = clipFinite(value,bound)
if ~isfinite(value), value = 0; end
value = min(max(value,-bound),bound);
end
