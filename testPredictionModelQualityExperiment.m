function report = testPredictionModelQualityExperiment(outputRoot)
%TESTPREDICTIONMODELQUALITYEXPERIMENT Synthetic + full SMD1 isolation checks.
root=fileparts(mfilename('fullpath')); addpath(root,fullfile(root,'BLEA_D'));
if nargin<1||isempty(outputRoot)
    outputRoot=fullfile(root,'results_prediction_model_quality_tests','SMD1_1run');
end
savedRng=rng; cleanup=onCleanup(@() rng(savedRng)); %#ok<NASGU>
BI=getBLOPinfo('SMD',1,5); options=predictionModelQualityOptions;
rng(771,'twister'); X=randn(40,2)*.3; target=[.1 -.05];
UF=-sum((X-target).^2,2);
event=struct('generation',10,'history',[repmat(10,40,1),(1:40)',X,UF], ...
    'CMA',struct('xmean',zeros(1,5)));
before=rng;
[models,shared]=fitPredictionQualityModels(event,BI,options,123, ...
    repmat(struct('UX',[0 0]),8,1));
assert(isequal(before,rng),'Model fitting/search leaked global RNG.');
assert(all([models.fitValid])&&all([models.valid]), ...
    'A model failed the well-conditioned synthetic peak test.');
assert(norm(models(1).x-target)<1e-5&&norm(models(5).x-target)<1e-8, ...
    'QP maximum direction/stationary solution is incorrect.');
assert(norm(models(2).x-target)<1e-3,'Complete cubic fit/search is incorrect.');
assert(size(shared.candidateZ,1)==200&&all(vecnorm(shared.candidateZ,2,2)<=1+1e-10));
assert(all([models(1:4).successfulStarts]>0));
assert(models(4).gpMeanCheckError<1e-7&&isfinite(models(4).predictedStd));
assert(numel(models(2).payload.coefficients)==10,'Cubic mixed terms are missing.');
% Model-specific Hessian rejection must apply only to the analytic reference.
event.history(:,5)=X(:,1).^2+2*X(:,2).^2+.3*X(:,1);
[positive,~]=fitPredictionQualityModels(event,BI,options,124,repmat(struct('UX',[0 0]),8,1));
assert(~positive(5).valid&&positive(1).valid&&positive(2).valid, ...
    'A quadratic-only Hessian sign test leaked into the shared model optimizer.');
% Insufficient and degenerate histories must produce explicit, zero-eval failure.
few=event; few.history=few.history(1:20,:);
failed=fitPredictionQualityModels(few,BI,options,125,repmat(struct('UX',[0 0]),8,1));
assert(~any([failed.valid])&&all(strcmp({failed.reason},'insufficient_samples')));
flat=event; flat.history(:,5)=0;
failed=fitPredictionQualityModels(flat,BI,options,126,repmat(struct('UX',[0 0]),8,1));
assert(~any([failed.valid])&&all(strcmp({failed.reason},'degenerate_response_scale')));

experiment=runSMDPredictionModelQualityExperiment(outputRoot,1,1);
D=experiment.details; fair=D(D.model~="QP_analytic",:);
assert(~isempty(D)&&all(experiment.audit.isolation_verified)&& ...
    all(experiment.audit.main_trajectory_equal)&&all(experiment.audit.main_rng_equal));
assert(all(D.generation>=10&D.target_generation==D.generation+1));
assert(all(D.history_sample_count==40));
assert(all(D.extra_UFE(D.evaluation_valid)==1));
assert(all(D.extra_UFE(~D.prediction_valid)==0));
assert(all(fair.inside_common_region(fair.prediction_valid)));
valid=fair.prediction_valid&fair.evaluation_valid;
assert(all(fair.rank(valid)>=1&fair.rank(valid)<=9));
assert(all(fair.potential_parent(valid)==(fair.rank(valid)<=fair.mu(valid))));
for id=["QP","CP","RBF","GPR"]
    assert(any(D.model==id&D.prediction_valid&D.evaluation_valid),'No real valid %s prediction.',id);
end
file=fullfile(outputRoot,'SMD1','run001','model_quality_run.mat');
loaded=load(file,'stateResults');
for j=1:numel(loaded.stateResults)
    result=loaded.stateResults{j}; pool=result.shared.candidateZ;
    assert(size(pool,1)==200);
    for k=1:4
        m=result.models(k);
        if m.fitValid
            assert(isequal(m.startZ,pool(m.startIndices,:)), ...
                'A model did not select starts from the shared 200 locations.');
            assert(numel(m.exitFlags)==5,'A model did not use five optimizer starts.');
        end
    end
end
for file={'prediction_model_details.csv','prediction_model_run_summary.csv', ...
        'prediction_model_summary.csv','prediction_model_stage_summary.csv', ...
        'prediction_model_pairwise.csv','prediction_model_results.mat'}
    assert(isfile(fullfile(outputRoot,file{1})),'Missing required output.');
end
report=struct('passed',true,'outputRoot',outputRoot, ...
    'opportunities',height(D)/5,'validFourModelPredictions',sum(valid), ...
    'mainTrajectoryIdentical',true,'modelFittingAndEvaluationRngIsolated',true, ...
    'matchedSamplesAndCandidatePool',true,'extraUFE',sum(D.extra_UFE), ...
    'extraLFE',sum(D.extra_LFE));
save(fullfile(outputRoot,'prediction_model_quality_test_report.mat'),'report');
disp(report);
end
