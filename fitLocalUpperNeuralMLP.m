function model = fitLocalUpperNeuralMLP( ...
    history,meanBefore,generation,BI,options,stream)
%FITLOCALUPPERNEURALMLP Local 2-6-1 tanh surrogate and bounded maximiser.
% The model uses only archive entries evaluated no later than GENERATION.
% It never calls an upper objective or a lower-level solver.  Random MLP
% initialisation comes solely from the private STREAM created for this run.

upperDim = numel(meanBefore);
historyColumns = upperDim + 3;
upperColumns = 3:(upperDim+2);
objectiveColumn = upperDim + 3;
hiddenUnits = options.neuralHiddenUnits;
model = emptyNeuralModel(upperDim,historyColumns,hiddenUnits);

if upperDim ~= 2
    model.reason = 'neural_model_requires_DU2';
    return;
end
if isempty(stream) || ~isa(stream,'RandStream')
    model.reason = 'missing_private_neural_stream';
    return;
end
if generation < options.startGeneration
    return;
end
if size(history,2) ~= historyColumns
    model.reason = 'invalid_history_shape';
    return;
end

% This is intentionally the same local-archive rule used by the original
% quadratic model: same time window, same Euclidean distance to m_t, same
% archive-order tie break, and no objective-based sample selection.
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
offsets = bsxfun(@minus,history(:,upperColumns),meanBefore);
distances = sqrt(sum(offsets.^2,2));
[~,order] = sortrows([distances,(1:numel(distances))'],[1 2]);
selected = order(1:options.sampleCount);
samples = history(selected,:);
selectedDistances = distances(selected);
scale = max(selectedDistances);
model.selectedSamples = samples;
model.distances = selectedDistances;
model.scale = scale;
if ~isfinite(scale) || ...
        scale <= options.minScaleRelative * max(1,norm(meanBefore))
    model.reason = 'degenerate_scale';
    return;
end
z = offsets(selected,:) / scale;
model.selectedSpreadEigenvalues = eig(cov(z,1));
if min(abs(model.selectedSpreadEigenvalues)) > 0
    model.selectedSpreadCondition = ...
        max(abs(model.selectedSpreadEigenvalues)) / ...
        min(abs(model.selectedSpreadEigenvalues));
end
yRaw = samples(:,objectiveColumn);
yMean = mean(yRaw);
yScale = std(yRaw,0);
minimumScale = sqrt(eps) * max(1,abs(yMean));
if ~isfinite(yScale) || yScale <= minimumScale
    model.reason = 'degenerate_target';
    return;
end
y = (yRaw-yMean) / yScale;
if any(~isfinite(z(:))) || any(~isfinite(y))
    model.reason = 'invalid_normalised_training_data';
    return;
end
model.targetMean = yMean;
model.targetScale = yScale;

% A deterministic 80/20 split is intentional.  It avoids a global random
% draw yet still provides eight validation samples when the fixed archive
% size is 40.  The archive itself is already deterministically ordered by
% distance then insertion order.
sampleIndices = (1:options.sampleCount)';
validationMask = floor(sampleIndices*options.neuralValidationFraction) > ...
    floor((sampleIndices-1)*options.neuralValidationFraction);
if sum(validationMask) < 1 || sum(~validationMask) < 1
    model.reason = 'invalid_validation_split';
    return;
end
trainMask = ~validationMask;
model.trainingSampleCount = sum(trainMask);
model.validationSampleCount = sum(validationMask);
try
    [network,training] = trainMLP(z(trainMask,:),y(trainMask), ...
        z(validationMask,:),y(validationMask),options,stream,upperDim);
catch exception
    model.reason = ['training_exception:' exception.identifier];
    return;
end
if ~training.valid
    model.reason = training.reason;
    return;
end
model.W1 = network.W1;
model.b1 = network.b1;
model.W2 = network.W2;
model.b2 = network.b2;
model.trainMSE = training.trainMSE;
model.validationMSE = training.validationMSE;
model.trainingEpochs = training.trainingEpochs;
model.bestEpoch = training.bestEpoch;
model.trainingConverged = training.converged;
model.fitTrainRMSE = sqrt(training.trainMSE) * yScale;
model.fitValidationRMSE = sqrt(training.validationMSE) * yScale;

zLower = (BI.u_lb(:)' - meanBefore) / scale;
zUpper = (BI.u_ub(:)' - meanBefore) / scale;
if any(~isfinite(zLower)) || any(~isfinite(zUpper)) || ...
        any(zLower > zUpper)
    model.reason = 'invalid_search_bounds';
    return;
end
model.z_lower = zLower;
model.z_upper = zUpper;
[zPred,optimisation] = maximiseNetwork(network,z,yRaw,zLower,zUpper,options);
model.optimizerConverged = optimisation.converged;
model.optimizerIterations = optimisation.iterations;
model.optimizerStartCount = optimisation.startCount;
model.optimizerSuccessfulStarts = optimisation.successfulStarts;
model.optimizerFailureReason = optimisation.reason;
if ~optimisation.valid
    model.reason = optimisation.reason;
    return;
end
model.z_pred = zPred;
model.zPredNorm = norm(zPred);
[predictedNormalisedUF,~] = mlpValueGradient(network,zPred);
model.predictedNormalisedUF = predictedNormalisedUF;
model.predictedUF = yMean + yScale * predictedNormalisedUF;
model.x_pred = meanBefore + scale * zPred;
if ~isreal(model.x_pred) || any(~isfinite(model.x_pred)) || ...
        any(model.x_pred < BI.u_lb-1e-10) || any(model.x_pred > BI.u_ub+1e-10)
    model.reason = 'invalid_prediction';
    return;
end
model.predictionToNearestSample = min(sqrt(sum(bsxfun(@minus, ...
    samples(:,upperColumns),model.x_pred).^2,2)));
model.predictionToNearestSampleNormalized = ...
    model.predictionToNearestSample / scale;
model.predictionInHull = inSelectedHull(model.x_pred,samples(:,upperColumns));
model.predictionAtSearchBoundary = isAtBoundary( ...
    zPred,zLower,zUpper,options.maxPredictionNormalizedDistance);
model.valid = true;
model.reason = 'valid';
end

function model = emptyNeuralModel(upperDim,historyColumns,hiddenUnits)
% H-related fields are deliberately NaN: this first NN version has no
% quadratic curvature and therefore cannot be paired with H1 sampling.
model = struct('modelType','neural_mlp','valid',false,'reason','warmup', ...
    'upperDimension',upperDim,'hiddenUnits',hiddenUnits, ...
    'selectedSamples',zeros(0,historyColumns),'distances',zeros(0,1), ...
    'scale',NaN,'selectedSpreadEigenvalues',NaN(upperDim,1), ...
    'selectedSpreadCondition',NaN,'targetMean',NaN,'targetScale',NaN, ...
    'trainingSampleCount',0,'validationSampleCount',0, ...
    'W1',NaN(hiddenUnits,upperDim),'b1',NaN(hiddenUnits,1), ...
    'W2',NaN(1,hiddenUnits),'b2',NaN, ...
    'trainMSE',NaN,'validationMSE',NaN,'trainingEpochs',0, ...
    'bestEpoch',NaN,'trainingConverged',false, ...
    'fitTrainRMSE',NaN,'fitValidationRMSE',NaN, ...
    'z_lower',NaN(1,upperDim),'z_upper',NaN(1,upperDim), ...
    'z_pred',NaN(1,upperDim),'zPredNorm',NaN, ...
    'x_pred',NaN(1,upperDim),'predictedNormalisedUF',NaN, ...
    'predictedUF',NaN,'optimizerConverged',false, ...
    'optimizerIterations',0,'optimizerStartCount',0, ...
    'optimizerSuccessfulStarts',0,'optimizerFailureReason','not_run', ...
    'predictionToNearestSample',NaN, ...
    'predictionToNearestSampleNormalized',NaN, ...
    'predictionInHull',NaN,'predictionAtSearchBoundary',false, ...
    'H',NaN(upperDim),'hessianEigenvalues',NaN(upperDim,1), ...
    'hessianRcond',NaN);
end

function [network,info] = trainMLP(trainZ,trainY,validationZ,validationY, ...
    options,stream,inputDim)
hiddenUnits = options.neuralHiddenUnits;
network = struct('W1',randn(stream,hiddenUnits,inputDim) * sqrt(1/inputDim), ...
    'b1',zeros(hiddenUnits,1), ...
    'W2',randn(stream,1,hiddenUnits) * sqrt(1/hiddenUnits), ...
    'b2',0);
bestNetwork = network;
bestValidation = Inf;
bestEpoch = NaN;
badEpochs = 0;
firstMoment = zeroMoments(network);
secondMoment = zeroMoments(network);
beta1 = 0.9; beta2 = 0.999; adamEpsilon = 1e-8;
info = struct('valid',false,'reason','not_run','trainMSE',NaN, ...
    'validationMSE',NaN,'trainingEpochs',0,'bestEpoch',NaN, ...
    'converged',false);
for epoch = 1:options.neuralMaxEpochs
    [prediction,cache] = mlpForward(network,trainZ);
    residual = prediction-trainY;
    [dW1,db1,dW2,db2] = mlpGradients(network,cache,trainZ,residual, ...
        options.neuralL2);
    gradients = struct('W1',dW1,'b1',db1,'W2',dW2,'b2',db2);
    [firstMoment,secondMoment,network] = adamStep(network,gradients, ...
        firstMoment,secondMoment,epoch,options.neuralLearningRate, ...
        beta1,beta2,adamEpsilon);
    [validationPrediction,~] = mlpForward(network,validationZ);
    validationMSE = mean((validationPrediction-validationY).^2);
    if ~isfinite(validationMSE) || any(~isfinite(network.W1(:))) || ...
            any(~isfinite(network.W2(:)))
        info.reason = 'nonfinite_training_state';
        return;
    end
    if validationMSE < bestValidation-options.neuralMinImprovement
        bestNetwork = network;
        bestValidation = validationMSE;
        bestEpoch = epoch;
        badEpochs = 0;
    else
        badEpochs = badEpochs+1;
    end
    if badEpochs >= options.neuralPatience
        info.converged = true;
        break;
    end
end
network = bestNetwork;
[trainPrediction,~] = mlpForward(network,trainZ);
[validationPrediction,~] = mlpForward(network,validationZ);
info.valid = isfinite(bestValidation) && isfinite(mean((trainPrediction-trainY).^2));
info.reason = ternary(info.valid,'valid','invalid_training_result');
info.trainMSE = mean((trainPrediction-trainY).^2);
info.validationMSE = mean((validationPrediction-validationY).^2);
info.trainingEpochs = epoch;
info.bestEpoch = bestEpoch;
end

function [first,second,network] = adamStep(network,gradient,first,second, ...
    iteration,learningRate,beta1,beta2,epsilonValue)
fields = {'W1','b1','W2','b2'};
for i = 1:numel(fields)
    field = fields{i};
    first.(field) = beta1*first.(field) + (1-beta1)*gradient.(field);
    second.(field) = beta2*second.(field) + ...
        (1-beta2)*(gradient.(field).^2);
    firstHat = first.(field)/(1-beta1^iteration);
    secondHat = second.(field)/(1-beta2^iteration);
    network.(field) = network.(field) - learningRate * firstHat ./ ...
        (sqrt(secondHat)+epsilonValue);
end
end

function moments = zeroMoments(network)
moments = struct('W1',zeros(size(network.W1)),'b1',zeros(size(network.b1)), ...
    'W2',zeros(size(network.W2)),'b2',0);
end

function [output,cache] = mlpForward(network,z)
cache.hiddenPre = bsxfun(@plus,z*network.W1',network.b1');
cache.hidden = tanh(cache.hiddenPre);
output = cache.hidden*network.W2' + network.b2;
end

function [dW1,db1,dW2,db2] = mlpGradients(network,cache,z,residual,l2)
sampleCount = size(z,1);
dW2 = (2/sampleCount) * (residual' * cache.hidden) + 2*l2*network.W2;
db2 = 2*mean(residual);
hiddenDelta = (2/sampleCount) * (residual*network.W2) .* ...
    (1-cache.hidden.^2);
dW1 = hiddenDelta' * z + 2*l2*network.W1;
db1 = sum(hiddenDelta,1)';
end

function [zBest,info] = maximiseNetwork(network,z,rawUF,zLower,zUpper,options)
% Projected gradient ascent avoids an Optimization Toolbox dependency.  It
% is a bounded multistart numerical optimiser over the trained surrogate;
% it never obtains a true fitness value during the search.
[~,rank] = sort(rawUF,'descend');
startCount = min(options.neuralOptimizationStartCount,numel(rank)+1);
starts = [z(rank(1:min(numel(rank),startCount-1)),:);zeros(1,size(z,2))];
if isempty(starts), starts = zeros(1,size(z,2)); end
starts = unique(starts,'rows','stable');
zBest = NaN(1,size(z,2));
bestValue = -Inf;
totalIterations = 0;
successfulStarts = 0;
anyConverged = false;
for startIndex = 1:size(starts,1)
    current = projectPoint(starts(startIndex,:),zLower,zUpper, ...
        options.maxPredictionNormalizedDistance);
    [currentValue,gradient] = mlpValueGradient(network,current);
    if ~isfinite(currentValue) || any(~isfinite(gradient))
        continue;
    end
    converged = false;
    for iteration = 1:options.neuralOptimizationMaxIterations
        totalIterations = totalIterations+1;
        gradientNorm = norm(gradient);
        if gradientNorm <= options.neuralOptimizationGradientTolerance
            converged = true;
            break;
        end
        step = options.neuralOptimizationInitialStep;
        accepted = false;
        while step >= options.neuralOptimizationMinStep
            candidate = projectPoint(current+step*gradient, ...
                zLower,zUpper,options.maxPredictionNormalizedDistance);
            [candidateValue,candidateGradient] = mlpValueGradient(network,candidate);
            if isfinite(candidateValue) && all(isfinite(candidateGradient)) && ...
                    candidateValue >= currentValue + 1e-12
                current = candidate;
                currentValue = candidateValue;
                gradient = candidateGradient;
                accepted = true;
                break;
            end
            step = step/2;
        end
        if ~accepted
            converged = true;
            break;
        end
    end
    if isfinite(currentValue)
        successfulStarts = successfulStarts+1;
        if currentValue > bestValue
            bestValue = currentValue;
            zBest = current;
        end
    end
    anyConverged = anyConverged || converged;
end
info = struct('valid',isfinite(bestValue) && all(isfinite(zBest)), ...
    'reason','valid','converged',anyConverged, ...
    'iterations',totalIterations,'startCount',size(starts,1), ...
    'successfulStarts',successfulStarts);
if ~info.valid
    info.reason = 'network_optimisation_failed';
end
end

function [value,gradient] = mlpValueGradient(network,z)
[value,cache] = mlpForward(network,z);
gradient = (network.W2 .* (1-cache.hidden.^2)) * network.W1;
end

function z = projectPoint(z,zLower,zUpper,maximumNorm)
z = min(max(z,zLower),zUpper);
if isfinite(maximumNorm) && norm(z) > maximumNorm
    z = z * (maximumNorm/norm(z));
    z = min(max(z,zLower),zUpper);
end
end

function atBoundary = isAtBoundary(z,zLower,zUpper,maximumNorm)
tolerance = 1e-8 * max(1,max(abs([zLower,zUpper])));
atBoundary = any(abs(z-zLower) <= tolerance) || ...
    any(abs(z-zUpper) <= tolerance);
if isfinite(maximumNorm)
    atBoundary = atBoundary || abs(norm(z)-maximumNorm) <= tolerance;
end
end

function inside = inSelectedHull(x,samples)
try
    indices = convhull(samples(:,1),samples(:,2));
    inside = double(inpolygon(x(1),x(2),samples(indices,1),samples(indices,2)));
catch
    inside = NaN;
end
end

function output = ternary(condition,trueValue,falseValue)
if condition, output = trueValue; else, output = falseValue; end
end
