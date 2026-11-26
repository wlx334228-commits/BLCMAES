function experiment = runSMDPredictionModelQualityExperiment( ...
        outputRoot,numberOfRuns,problemNumbers,settings)
%RUNSMDPREDICTIONMODELQUALITYEXPERIMENT Offline QP/CP/RBF/GPR quality study.
% Defaults: SMD1--8, (2+3), ten independent A0 runs per problem.
% No prediction is injected. Each t>=10 with a real t+1 population is tested.
% Every model shares 40 samples, normalization, 200 candidates and 5 starts.
% QP_analytic is a separately labelled historical reference, not a fifth
% member of the fair four-model search comparison.
%
%   testPredictionModelQualityExperiment    % correctness and isolation
%   runSMDPredictionModelQualityExperiment  % 8 problems x 10 runs
%   runSMDPredictionModelQualityExperiment([],1,1)
%
% Results are saved after each state and run. Re-running skips completed work.
projectRoot=fileparts(mfilename('fullpath'));
addpath(projectRoot,fullfile(projectRoot,'BLEA_D'));
if nargin<1||isempty(outputRoot)
    outputRoot=fullfile(projectRoot,'results_prediction_model_quality_2D','SMD1_to_SMD8_10runs');
end
if nargin<2||isempty(numberOfRuns), numberOfRuns=10; end
if nargin<3||isempty(problemNumbers), problemNumbers=1:8; end
if nargin<4, settings=struct(); end
options=predictionModelQualityOptions(settings);
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'},{'vector','integer','>=',1,'<=',8});
assert(exist('fitrgp','file')==2 && license('test','Statistics_Toolbox'), ...
    'Statistics and Machine Learning Toolbox (fitrgp) is required.');
assert(exist('fmincon','file')==2 && license('test','Optimization_Toolbox'), ...
    'Optimization Toolbox (fmincon) is required.');
if ~isfolder(outputRoot), mkdir(outputRoot); end
config=struct('options',options,'numberOfRuns',numberOfRuns, ...
    'problemNumbers',problemNumbers(:)','totalDim',5,'upperDim',2,'lowerDim',3, ...
    'seedRule','seed=(SMD-1)*numberOfRuns+run_no','UmaxFEs',2500, ...
    'UmaxImprFEs',350,'LmaxFEs',250,'LmaxImprFEs',25, ...
    'searchRegion','bounds intersect normalized radius <= 1', ...
    'GPRegularization','fixed normalized Sigma=.01; kernel parameters fitted', ...
    'notes','True evaluations offline; no injection or auxiliary sampling.');
configFile=fullfile(outputRoot,'prediction_model_config.mat');
if isfile(configFile)
    old=load(configFile,'config');
    assert(isequaln(old.config,config),'Output directory has a different model-quality configuration.');
else
    save(configFile,'config');
end
snapshotSources(projectRoot,outputRoot);
global ulFunctionEvaluations llFunctionEvaluations;
savedU=ulFunctionEvaluations; savedL=llFunctionEvaluations; savedRng=rng;
cleanup=onCleanup(@() restoreGlobals(savedU,savedL,savedRng)); %#ok<NASGU>
allDetails=table(); runFiles=cell(0,1); auditRows=cell(0,11);
for smdNo=problemNumbers(:)'
    for runNo=1:numberOfRuns
        seed=(smdNo-1)*numberOfRuns+runNo;
        leaf=fullfile(outputRoot,sprintf('SMD%d',smdNo),sprintf('run%03d',runNo));
        if ~isfolder(leaf), mkdir(leaf); end
        mainFile=fullfile(leaf,'main_run.mat');
        BI=makeBI(smdNo);
        if isfile(mainFile)
            loaded=load(mainFile,'mainIns','mainRngAfter','audit');
            mainIns=loaded.mainIns; mainRngAfter=loaded.mainRngAfter; audit=loaded.audit;
        else
            rng(seed,'twister'); mainIns=BLCMAES(BI); mainRngAfter=rng;
            audit=struct('verified',false,'rng_equal',false,'normal_trajectory_equal',false);
            if options.verifyMainTrajectory
                plainBI=BI; plainBI.predictionModelQualityDiagnostic.enabled=false;
                rng(seed,'twister'); plain=BLCMAES(plainBI); plainRngAfter=rng;
                copy=rmfield(mainIns,'predictionModelQualityCapture');
                audit.normal_trajectory_equal=isequaln(copy,plain);
                audit.rng_equal=isequaln(mainRngAfter,plainRngAfter);
                assert(audit.normal_trajectory_equal && audit.rng_equal, ...
                    'BLCMAES:ModelQualityIsolation','Snapshot capture changed A0 trajectory.');
                audit.verified=true;
            end
            save(mainFile,'mainIns','BI','mainRngAfter','audit','-v7.3');
        end
        runFile=fullfile(leaf,'model_quality_run.mat');
        if isfile(runFile)
            loaded=load(runFile,'details','complete');
            if loaded.complete
                details=loaded.details;
                fprintf('Reusing SMD%d run %d (%d model records).\n',smdNo,runNo,height(details));
            else
                error('Unexpected incomplete final run file.');
            end
        else
            events=mainIns.predictionModelQualityCapture.events;
            eligible=find(cellfun(@(e)e.generation>=options.startGeneration&&e.updateApplied,events));
            eligible=eligible(eligible<numel(events));
            details=table(); stateResults=cell(numel(eligible),1);
            for j=1:numel(eligible)
                eventIndex=eligible(j); event=events{eventIndex}; next=events{eventIndex+1};
                stateFile=fullfile(leaf,sprintf('state_gen%04d.mat',event.generation));
                if isfile(stateFile)
                    loaded=load(stateFile,'part','stateResult');
                    part=loaded.part; stateResult=loaded.stateResult;
                else
                    beforeRng=rng; beforeU=ulFunctionEvaluations; beforeL=llFunctionEvaluations;
                    [part,stateResult]=evaluatePredictionModelQualityState( ...
                        event,next,BI,options,smdNo,runNo,seed,j,mainIns.UFEs,events{eligible(1)}.UFE);
                    assert(isequal(beforeRng,rng)&&isequaln(beforeU,ulFunctionEvaluations)&& ...
                        isequaln(beforeL,llFunctionEvaluations),'Offline evaluation leaked RNG/counters.');
                    save(stateFile,'part','stateResult','-v7');
                end
                details=[details;part]; stateResults{j}=stateResult; %#ok<AGROW>
                if mod(j,10)==0||j==numel(eligible)
                    fprintf('SMD%d run %d: state %d/%d, generation %d, valid %d/4.\n', ...
                        smdNo,runNo,j,numel(eligible),event.generation,sum(part.prediction_valid(1:4)));
                end
            end
            complete=true;
            save(runFile,'details','stateResults','complete','config','-v7.3');
            writetable(details,fullfile(leaf,'prediction_model_details.csv'));
        end
        runFiles{end+1,1}=relativePath(runFile,outputRoot); %#ok<AGROW>
        allDetails=[allDetails;details]; %#ok<AGROW>
        auditRows(end+1,:)={smdNo,runNo,seed,mainIns.UFEs,mainIns.LFEs, ...
            abs(mainIns.UF-BI.u_fopt),abs(mainIns.LF-BI.l_fopt), ...
            audit.verified,audit.normal_trajectory_equal,audit.rng_equal,height(details)/5}; %#ok<AGROW>
        % Keep aggregate CSV current even if the next run is interrupted.
        writetable(allDetails,fullfile(outputRoot,'prediction_model_details.csv'));
    end
end
auditTable=cell2table(auditRows,'VariableNames',{'SMD','run_no','seed','main_UFE', ...
    'main_LFE','final_upper_error','final_lower_error','isolation_verified', ...
    'main_trajectory_equal','main_rng_equal','prediction_opportunities'});
writetable(auditTable,fullfile(outputRoot,'prediction_model_main_run_audit.csv'));
analysis=summarisePredictionModelQuality(allDetails,outputRoot);
figures=plotPredictionModelQuality(analysis,outputRoot);
experiment=struct('config',config,'outputRoot',outputRoot,'details',allDetails, ...
    'analysis',analysis,'audit',auditTable,'runFiles',{runFiles},'figureFiles',{figures});
save(fullfile(outputRoot,'prediction_model_results.mat'),'experiment','-v7.3');
writePredictionModelQualityReport(experiment);
fprintf('Model-quality diagnostic complete: %s\n',outputRoot);
end

function BI=makeBI(smd)
BI=getBLOPinfo('SMD',smd,5);
BI.UmaxFEs=2500; BI.UmaxImprFEs=350; BI.LmaxFEs=250; BI.LmaxImprFEs=25;
BI.enablePredictionCorrection=false; BI.predictionCorrection=struct('logEnabled',false);
BI.enableIndividualPredictionCorrection=false;
BI.individualPredictionCorrection=struct('mode','A0','logEnabled',false);
BI.predictionInjection=struct('mode','A0','logEnabled',false);
BI.predictionModelQualityDiagnostic=struct('enabled',true);
BI.generationDiagnostic=struct('enabled',true);
end

function snapshotSources(root,out)
files={'BLCMAES.m','predictionModelQualityCapture.m','predictionModelQualityOptions.m', ...
    'fitPredictionQualityModels.m','evaluatePredictionModelQualityState.m', ...
    'evaluatePredictionScaleDiagnosticCandidate.m','fitLocalUpperQuadratic.m', ...
    'runSMDPredictionModelQualityExperiment.m','summarisePredictionModelQuality.m', ...
    'testPredictionModelQualityExperiment.m'};
destination=fullfile(out,'source_snapshot');
if ~isfolder(destination), mkdir(destination); end
for i=1:numel(files)
    target=fullfile(destination,files{i});
    if isfile(target)
        assert(strcmp(fileread(target),fileread(fullfile(root,files{i}))), ...
            'Source changed since this experiment began: %s. Use a new output directory.',files{i});
    else
        copyfile(fullfile(root,files{i}),target);
    end
end
end

function p=relativePath(file,root)
p=strrep(file(numel(root)+2:end),filesep,'/');
end

function restoreGlobals(U,L,state)
global ulFunctionEvaluations llFunctionEvaluations;
ulFunctionEvaluations=U; llFunctionEvaluations=L; rng(state);
end
