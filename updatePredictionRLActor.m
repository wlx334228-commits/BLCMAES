function [rl,update] = updatePredictionRLActor(rl,entry,previousBestUF,BI,options)
%UPDATEPREDICTIONRLACTOR One REINFORCE update after real evaluations.
% The log probability is that of the single eta/delta action selected before
% sampling.  It is intentionally NOT the probability of either auxiliary
% coordinate, because both candidates share one distribution action.

update = emptyPredictionRLUpdate(BI.u_dim);
if isempty(rl) || ~isfield(rl,'pendingDecision') || ...
        ~rl.pendingDecision.valid
    update.reason = 'no_valid_rl_action';
    return;
end
decision = rl.pendingDecision;
auxiliaryIndices = find(strcmp(entry.source_labels,'predicted_auxiliary'));
normalIndices = find(strcmp(entry.source_labels,'normal'));
directIndices = find(strcmp(entry.source_labels,'predicted_direct'));
trueUF = entry.upper_objective_after_true_evaluation(:);
if numel(auxiliaryIndices) ~= 2 || any(~isfinite(trueUF(auxiliaryIndices)))
    update.reason = 'missing_auxiliary_true_evaluations';
    rl.pendingDecision = emptyPredictionRLDecision(BI.u_dim);
    return;
end
auxUF = trueUF(auxiliaryIndices);
normalUF = trueUF(normalIndices);
directUF = trueUF(directIndices);
if isempty(normalUF) || any(~isfinite(normalUF))
    update.reason = 'missing_normal_true_evaluations';
    rl.pendingDecision = emptyPredictionRLDecision(BI.u_dim);
    return;
end
if isempty(directUF) || any(~isfinite(directUF))
    directComparison = 0.5;
else
    directComparison = mean(auxUF > directUF(1));
end
normalComparison = mean(mean(bsxfun(@gt,auxUF,normalUF')));
allUF = trueUF(isfinite(trueUF));
spread = max(std(allUF),1e-12);
if isfinite(previousBestUF)
    historicalImprovement = tanh((max(auxUF)-previousBestUF)/spread);
else
    historicalImprovement = 0;
end
reward = 0.50*(2*normalComparison-1) + ...
    0.25*(2*directComparison-1) + 0.25*historicalImprovement;
reward = min(max(reward,-options.rlRewardClip),options.rlRewardClip);
if rl.rewardBaselineInitialized
    baselineBefore = rl.rewardBaseline;
else
    baselineBefore = 0;
end
advantage = min(max(reward-baselineBefore,-options.rlRewardClip), ...
    options.rlRewardClip);
baselineAfter = (1-options.rlRewardBaselineRate)*baselineBefore + ...
    options.rlRewardBaselineRate*reward;

actorBefore = actorSnapshot(rl.actor);
[actor,gradientNormBefore,gradientNormAfter] = reinforceActorStep( ...
    rl.actor,decision,advantage,options);
rl.actor = actor;
rl.rewardBaseline = baselineAfter;
rl.rewardBaselineInitialized = true;
rl.previousEta = decision.eta;
rl.previousDelta = decision.delta;
rl.previousReward = reward;
if isfinite(entry.direct_better_than_normal_pair_rate)
    rl.previousDirectPairRate = entry.direct_better_than_normal_pair_rate;
end
rl.previousAuxDirectionPerformance = auxiliaryDirectionScores( ...
    entry,auxiliaryIndices,auxUF,trueUF,decision,BI,spread);
rl.pendingDecision = emptyPredictionRLDecision(BI.u_dim);

update = struct('updated',true,'reason','reinforce_updated','reward',reward, ...
    'reward_aux_vs_normal',normalComparison, ...
    'reward_aux_vs_direct',directComparison, ...
    'reward_history_improvement',historicalImprovement, ...
    'reward_baseline_before',baselineBefore,'reward_baseline_after',baselineAfter, ...
    'advantage',advantage,'auxiliary_UF',auxUF, ...
    'auxiliary_direction_performance',rl.previousAuxDirectionPerformance, ...
    'gradient_norm_before_clip',gradientNormBefore, ...
    'gradient_norm_after_clip',gradientNormAfter, ...
    'actor_step_before',actorBefore.step,'actor_step_after',rl.actor.step, ...
    'actor_parameter_norm_before',actorParameterNorm(actorBefore), ...
    'actor_parameter_norm_after',actorParameterNorm(actorSnapshot(rl.actor)), ...
    'actor_parameters_before',actorBefore, ...
    'actor_parameters_after',actorSnapshot(rl.actor));
rl.lastUpdate = update;
end

function scores = auxiliaryDirectionScores(entry,indices,auxUF,trueUF,decision,BI,spread)
scores = [0,0];
counts = [0,0];
if isempty(entry.auxiliary_actual_joint) || ...
        any(~isfinite(decision.principalDirections(:))) || ...
        any(~isfinite(entry.x_pred_actual))
    return;
end
offsets = entry.auxiliary_actual_joint(:,1:BI.u_dim) - entry.x_pred_actual;
populationMedian = median(trueUF(isfinite(trueUF)));
for i = 1:min(numel(indices),size(offsets,1))
    projection = abs(offsets(i,:)*decision.principalDirections);
    [~,direction] = max(projection);
    scores(direction) = scores(direction) + ...
        tanh((auxUF(i)-populationMedian)/spread);
    counts(direction) = counts(direction)+1;
end
nonzero = counts > 0;
scores(nonzero) = scores(nonzero)./counts(nonzero);
end

function [actor,normBefore,normAfter] = reinforceActorStep(actor,decision,advantage,options)
standardDeviation = max(decision.actor_std,options.rlMinActionStd);
standardized = (decision.action_latent-decision.actor_mean)./standardDeviation;
dMean = standardized./standardDeviation;
dLogStd = standardized.^2-1;
if ~decision.eta_learned
    dMean(1) = 0; dLogStd(1) = 0;
end
if ~decision.delta_learned
    dMean(2) = 0; dLogStd(2) = 0;
end
gradientOutput = advantage*[dMean;dLogStd];
gradientW2 = gradientOutput*decision.actor_hidden' - ...
    options.rlActorL2*actor.W2;
gradientB2 = gradientOutput;
gradientHidden = (actor.W2'*gradientOutput).*(1-decision.actor_hidden.^2);
gradientW1 = gradientHidden*decision.state_normalized' - ...
    options.rlActorL2*actor.W1;
gradientB1 = gradientHidden;
normBefore = sqrt(sum(gradientW1(:).^2)+sum(gradientB1.^2)+ ...
    sum(gradientW2(:).^2)+sum(gradientB2.^2));
scale = min(1,options.rlGradientClip/max(normBefore,eps));
gradientW1 = gradientW1*scale; gradientB1 = gradientB1*scale;
gradientW2 = gradientW2*scale; gradientB2 = gradientB2*scale;
normAfter = normBefore*scale;
actor.step = actor.step+1;
[actor.W1,actor.mW1,actor.vW1] = adamAscent(actor.W1,actor.mW1,actor.vW1, ...
    gradientW1,actor.step,options.rlLearningRate);
[actor.b1,actor.mb1,actor.vb1] = adamAscent(actor.b1,actor.mb1,actor.vb1, ...
    gradientB1,actor.step,options.rlLearningRate);
[actor.W2,actor.mW2,actor.vW2] = adamAscent(actor.W2,actor.mW2,actor.vW2, ...
    gradientW2,actor.step,options.rlLearningRate);
[actor.b2,actor.mb2,actor.vb2] = adamAscent(actor.b2,actor.mb2,actor.vb2, ...
    gradientB2,actor.step,options.rlLearningRate);
% Keep sampled action standard deviations within the documented bounds.
actor.b2(3:4) = min(max(actor.b2(3:4),log(options.rlMinActionStd)), ...
    log(options.rlMaxActionStd));
end

function [parameter,moment,variance] = adamAscent(parameter,moment,variance,gradient,step,learningRate)
beta1 = 0.9; beta2 = 0.999;
moment = beta1*moment+(1-beta1)*gradient;
variance = beta2*variance+(1-beta2)*(gradient.^2);
correctedMoment = moment/(1-beta1^step);
correctedVariance = variance/(1-beta2^step);
parameter = parameter+learningRate*correctedMoment./(sqrt(correctedVariance)+1e-8);
end

function snapshot = actorSnapshot(actor)
snapshot = struct('W1',actor.W1,'b1',actor.b1,'W2',actor.W2, ...
    'b2',actor.b2,'step',actor.step);
end

function value = actorParameterNorm(snapshot)
value = sqrt(sum(snapshot.W1(:).^2)+sum(snapshot.b1.^2)+ ...
    sum(snapshot.W2(:).^2)+sum(snapshot.b2.^2));
end
