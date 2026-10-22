function model = fitLocalUpperQuadratic(history,meanBefore,generation,options)
%FITLOCALUPPERQUADRATIC Fit the nearest historical UF samples, without FEs.
% history columns: generation, individual_index, xu1, xu2, UF.
% The reference mean is the mean that generated the current population.
% The current population has already been evaluated when this is called.
model = struct('valid',false,'reason','warmup','selectedSamples',zeros(0,5), ...
    'distances',zeros(0,1),'scale',NaN,'coefficients',NaN(6,1), ...
    'H',NaN(2),'hessianEigenvalues',NaN(2,1),'hessianRcond',NaN, ...
    'designCondition',NaN,'fitTrainRMSE',NaN,'fitTrainR2',NaN, ...
    'fitLOOCVRMSE',NaN,'fitLOOCVR2',NaN, ...
    'selectedSpreadEigenvalues',NaN(2,1),'selectedSpreadCondition',NaN, ...
    'z_pred',NaN(1,2),'zPredNorm',NaN,'x_pred',NaN(1,2), ...
    'predictionToNearestSample',NaN, ...
    'predictionToNearestSampleNormalized',NaN, ...
    'predictionInHull',NaN);
if generation < options.startGeneration
    return;
end
firstGeneration = max(1,generation-options.historyGenerations+1);
history = history(history(:,1) >= firstGeneration & ...
                  history(:,1) <= generation,:);
history = history(all(isfinite(history),2),:);
if size(history,1) < options.sampleCount
    model.reason = 'insufficient_samples';
    return;
end
offsets = bsxfun(@minus,history(:,3:4),meanBefore);
distances = sqrt(sum(offsets.^2,2));
% Break equal-distance ties by archive order, never by UF.
[~,order] = sortrows([distances,(1:numel(distances))'],[1 2]);
selected = order(1:options.sampleCount);
model.selectedSamples = history(selected,:);
model.distances = distances(selected);
model.scale = max(model.distances);
if ~isfinite(model.scale) || ...
        model.scale <= options.minScaleRelative * max(1,norm(meanBefore))
    model.reason = 'degenerate_scale';
    return;
end
z = offsets(selected,:) / model.scale;
selectedSpread = cov(z,1);
model.selectedSpreadEigenvalues = eig(selectedSpread);
if min(abs(model.selectedSpreadEigenvalues)) > 0
    model.selectedSpreadCondition = max(abs(model.selectedSpreadEigenvalues)) / ...
        min(abs(model.selectedSpreadEigenvalues));
end
design = [ones(size(z,1),1),z(:,1),z(:,2),z(:,1).^2, ...
          z(:,1).*z(:,2),z(:,2).^2];
singularValues = svd(design,0);
model.designCondition = singularValues(1)/singularValues(end);
if ~isfinite(model.designCondition) || ...
        model.designCondition > options.maxDesignCondition
    model.reason = 'ill_conditioned_design';
    return;
end
a = design \ history(selected,5);
model.coefficients = a;
if any(~isfinite(a))
    model.reason = 'invalid_coefficients';
    return;
end
fitted = design * a;
residual = history(selected,5) - fitted;
sumSquaredResidual = sum(residual.^2);
totalSumSquares = sum((history(selected,5) - mean(history(selected,5))).^2);
model.fitTrainRMSE = sqrt(mean(residual.^2));
if totalSumSquares > 0
    model.fitTrainR2 = 1 - sumSquaredResidual / totalSumSquares;
end
% Exact leave-one-out residuals for a full-rank least-squares model.  This
% reuses evaluated historical UF values only; it makes no objective call.
[Q,~] = qr(design,0);
leverage = sum(Q.^2,2);
leaveOneOutDenominator = 1 - leverage;
if all(isfinite(leaveOneOutDenominator)) && ...
        all(leaveOneOutDenominator > sqrt(eps))
    leaveOneOutResidual = residual ./ leaveOneOutDenominator;
    model.fitLOOCVRMSE = sqrt(mean(leaveOneOutResidual.^2));
    if totalSumSquares > 0
        model.fitLOOCVR2 = 1 - sum(leaveOneOutResidual.^2) / totalSumSquares;
    end
end
H = [2*a(4),a(5);a(5),2*a(6)];
model.H = H;
model.hessianEigenvalues = eig(H);
model.hessianRcond = rcond(H);
if ~isfinite(model.hessianRcond) || ...
        model.hessianRcond < options.minHessianRcond
    model.reason = 'ill_conditioned_hessian';
    return;
end
% ulTestProblem returns -F_original, and BL-CMA-ES sorts UF descending.
% Consequently this fit must have a strict MAXIMUM, not a minimum.
[~,notNegativeDefinite] = chol(-H);
if notNegativeDefinite
    model.reason = 'hessian_not_negative_definite';
    return;
end
model.z_pred = (-(H \ a(2:3)))';
model.x_pred = meanBefore + model.scale * model.z_pred;
if ~isreal(model.x_pred) || any(~isfinite(model.x_pred))
    model.reason = 'invalid_prediction';
    return;
end
model.zPredNorm = norm(model.z_pred);
predictionOffsets = bsxfun(@minus,model.selectedSamples(:,3:4),model.x_pred);
model.predictionToNearestSample = min(sqrt(sum(predictionOffsets.^2,2)));
model.predictionToNearestSampleNormalized = ...
    model.predictionToNearestSample / model.scale;
try
    hullIndices = convhull(model.selectedSamples(:,3),model.selectedSamples(:,4));
    model.predictionInHull = double(inpolygon(model.x_pred(1),model.x_pred(2), ...
        model.selectedSamples(hullIndices,3),model.selectedSamples(hullIndices,4)));
catch
    % A valid quadratic can still be fitted even if a hull diagnostic is
    % unavailable.  Keep the diagnostic missing instead of changing search.
    model.predictionInHull = NaN;
end
model.valid = true;
model.reason = 'valid';
end
