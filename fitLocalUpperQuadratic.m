function model = fitLocalUpperQuadratic(history,meanBefore,generation,options)
%FITLOCALUPPERQUADRATIC Fit a complete local upper-level quadratic model.
% The archive has rows [generation, individual_index, xu(1:DU), UF].  Only
% already evaluated entries are used.  The quadratic contains the constant,
% every linear term, every squared term, and every pairwise cross term.

upperDim = numel(meanBefore);
% The older mean/individual correction experiments share this fitting
% helper.  Missing high-dimensional controls retain their historic values.
if ~isfield(options,'useAllHistory'), options.useAllHistory = false; end
if ~isfield(options,'maxPredictionNormalizedDistance')
    options.maxPredictionNormalizedDistance = Inf;
end
coefficientCount = (upperDim + 1) * (upperDim + 2) / 2;
historyColumns = upperDim + 3;
quadraticPairs = completeQuadraticPairs(upperDim);
model = struct('valid',false,'reason','warmup', ...
    'upperDimension',upperDim,'coefficientCount',coefficientCount, ...
    'quadraticPairs',quadraticPairs, ...
    'selectedSamples',zeros(0,historyColumns),'distances',zeros(0,1), ...
    'scale',NaN,'coefficients',NaN(coefficientCount,1), ...
    'H',NaN(upperDim),'hessianEigenvalues',NaN(upperDim,1), ...
    'hessianRcond',NaN,'designRank',NaN,'designCondition',NaN, ...
    'fitTrainRMSE',NaN,'fitTrainR2',NaN, ...
    'fitLOOCVRMSE',NaN,'fitLOOCVR2',NaN, ...
    'selectedSpreadEigenvalues',NaN(upperDim,1), ...
    'selectedSpreadCondition',NaN, ...
    'z_pred',NaN(1,upperDim),'zPredNorm',NaN, ...
    'x_pred',NaN(1,upperDim), ...
    'predictionToNearestSample',NaN, ...
    'predictionToNearestSampleNormalized',NaN, ...
    'predictionInHull',NaN);

if generation < options.startGeneration
    return;
end
if size(history,2) ~= historyColumns
    model.reason = 'invalid_history_shape';
    return;
end
if ~options.useAllHistory
    firstGeneration = max(1,generation-options.historyGenerations+1);
    history = history(history(:,1) >= firstGeneration & ...
                      history(:,1) <= generation,:);
else
    history = history(history(:,1) <= generation,:);
end
history = history(all(isfinite(history),2),:);
if size(history,1) < options.sampleCount
    model.reason = 'insufficient_samples';
    return;
end

upperColumns = 3:(upperDim+2);
objectiveColumn = upperDim + 3;
offsets = bsxfun(@minus,history(:,upperColumns),meanBefore);
distances = sqrt(sum(offsets.^2,2));
% Equal distances are resolved by archive order, never by UF quality.
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

design = buildCompleteQuadraticDesign(z,quadraticPairs);
model.designRank = rank(design);
if model.designRank < coefficientCount
    model.reason = 'rank_deficient_design';
    return;
end
singularValues = svd(design,0);
model.designCondition = singularValues(1) / singularValues(end);
if ~isfinite(model.designCondition) || ...
        model.designCondition > options.maxDesignCondition
    model.reason = 'ill_conditioned_design';
    return;
end

coefficients = design \ history(selected,objectiveColumn);
model.coefficients = coefficients;
if any(~isfinite(coefficients))
    model.reason = 'invalid_coefficients';
    return;
end
fitted = design * coefficients;
residual = history(selected,objectiveColumn) - fitted;
sumSquaredResidual = sum(residual.^2);
totalSumSquares = sum((history(selected,objectiveColumn) - ...
    mean(history(selected,objectiveColumn))).^2);
model.fitTrainRMSE = sqrt(mean(residual.^2));
if totalSumSquares > 0
    model.fitTrainR2 = 1 - sumSquaredResidual / totalSumSquares;
end

% This diagnostic reuses the same fitted matrix and evaluated values.  It
% never evaluates an objective or executes a lower-level optimization.
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

H = completeQuadraticHessian(coefficients,upperDim,quadraticPairs);
model.H = H;
model.hessianEigenvalues = eig(H);
model.hessianRcond = rcond(H);
if ~isfinite(model.hessianRcond) || ...
        model.hessianRcond < options.minHessianRcond
    model.reason = 'ill_conditioned_hessian';
    return;
end
% UF is -F_original and BL-CMA-ES ranks it in descending order.  Therefore
% an acceptable stationary point is a strict quadratic maximum.
[~,notNegativeDefinite] = chol(-H);
if notNegativeDefinite
    model.reason = 'hessian_not_negative_definite';
    return;
end

linearCoefficients = coefficients(2:(upperDim+1));
model.z_pred = (-(H \ linearCoefficients))';
model.zPredNorm = norm(model.z_pred);
if ~isfinite(model.zPredNorm) || ...
        model.zPredNorm > options.maxPredictionNormalizedDistance
    model.reason = 'extrapolated_prediction';
    return;
end
model.x_pred = meanBefore + model.scale * model.z_pred;
if ~isreal(model.x_pred) || any(~isfinite(model.x_pred))
    model.reason = 'invalid_prediction';
    return;
end

predictionOffsets = bsxfun(@minus, ...
    model.selectedSamples(:,upperColumns),model.x_pred);
model.predictionToNearestSample = min(sqrt(sum(predictionOffsets.^2,2)));
model.predictionToNearestSampleNormalized = ...
    model.predictionToNearestSample / model.scale;
if upperDim == 2
    try
        hullIndices = convhull(model.selectedSamples(:,upperColumns(1)), ...
            model.selectedSamples(:,upperColumns(2)));
        model.predictionInHull = double(inpolygon(model.x_pred(1), ...
            model.x_pred(2),model.selectedSamples(hullIndices,upperColumns(1)), ...
            model.selectedSamples(hullIndices,upperColumns(2))));
    catch
        % The hull is diagnostic only.  Its absence must not change search.
        model.predictionInHull = NaN;
    end
end
model.valid = true;
model.reason = 'valid';
end

function design = buildCompleteQuadraticDesign(z,quadraticPairs)
% For DU=2 this retains the historic monomial order:
% 1, z1, z2, z1^2, z1*z2, z2^2.
sampleCount = size(z,1);
design = [ones(sampleCount,1),z];
for pairIndex = 1:size(quadraticPairs,1)
    i = quadraticPairs(pairIndex,1);
    j = quadraticPairs(pairIndex,2);
    design(:,end+1) = z(:,i) .* z(:,j); %#ok<AGROW>
end
end

function H = completeQuadraticHessian(coefficients,upperDim,quadraticPairs)
H = zeros(upperDim);
quadraticOffset = upperDim + 1;
for pairIndex = 1:size(quadraticPairs,1)
    i = quadraticPairs(pairIndex,1);
    j = quadraticPairs(pairIndex,2);
    value = coefficients(quadraticOffset + pairIndex);
    if i == j
        H(i,i) = 2 * value;
    else
        H(i,j) = value;
        H(j,i) = value;
    end
end
end

function pairs = completeQuadraticPairs(upperDim)
pairs = zeros(upperDim*(upperDim+1)/2,2);
index = 0;
for i = 1:upperDim
    for j = i:upperDim
        index = index + 1;
        pairs(index,:) = [i,j];
    end
end
end
