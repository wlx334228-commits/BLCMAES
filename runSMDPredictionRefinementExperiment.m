function experiment=runSMDPredictionRefinementExperiment(outputRoot,numberOfRuns,problemNumbers,settings)
%RUNSMDPREDICTIONREFINEMENTEXPERIMENT P1-centred historical knowledge reuse.
% Default: (2+3) SMD1--8 x ten A0 runs, S0 and S1 only, eight labels:
% P1/P2, G1/G2, Q1/Q2, R1/R2. No injection or auxiliary random sampling.
% S2 is separately enabled with struct('modes',{{'S0','S1','S2'}}).
%   testPredictionRefinementExperiment
%   runSMDPredictionRefinementExperiment
root=fileparts(mfilename('fullpath')); addpath(root,fullfile(root,'BLEA_D'));
if nargin<1||isempty(outputRoot)
    outputRoot=fullfile(root,'results_prediction_refinement_2D','S0_S1_SMD1_to_SMD8_10runs');
end
if nargin<2||isempty(numberOfRuns), numberOfRuns=10; end
if nargin<3||isempty(problemNumbers), problemNumbers=1:8; end
if nargin<4, settings=struct(); end
options=predictionRefinementOptions(settings); qualityOptions=predictionModelQualityOptions;
validateattributes(numberOfRuns,{'numeric'},{'scalar','integer','positive'});
validateattributes(problemNumbers,{'numeric'},{'vector','integer','>=',1,'<=',8});
assert(exist('fitrgp','file')==2&&exist('fmincon','file')==2&&exist('quadprog','file')==2, ...
    'fitrgp, fmincon and quadprog require Statistics and Optimization toolboxes.');
assert(license('test','Statistics_Toolbox')&&license('test','Optimization_Toolbox'), ...
    'Statistics and Optimization toolbox licenses are required.');
if ~isfolder(outputRoot), mkdir(outputRoot); end
config=struct('options',options,'qualityOptions',qualityOptions,'numberOfRuns',numberOfRuns, ...
    'problemNumbers',problemNumbers(:)','DU',2,'DL',3, ...
    'seedRule','seed=(SMD-1)*numberOfRuns+run_no', ...
    'samplePolicy','full true archive, same last ten generations, nearest 40, Euclidean, stable ties', ...
    'S0_S1_regions','original analytic rules for P; unchanged previous local region for G/R/Q', ...
    'S2_region','P1 Mahalanobis radius intersect each model original allowed region', ...
    'curvatureModel','diagonal strictly convex original F=-UF; five coefficients, no cross term');
file=fullfile(outputRoot,'prediction_refinement_config.mat');
if isfile(file)
    prior=load(file,'config'); assert(isequaln(prior.config,config),'Output configuration differs; choose a new directory.');
else, save(file,'config'); end
snapshotSources(root,outputRoot);
global ulFunctionEvaluations llFunctionEvaluations;
U=ulFunctionEvaluations; L=llFunctionEvaluations; randomState=rng;
cleanup=onCleanup(@() restore(U,L,randomState)); %#ok<NASGU>
allDetails=table(); allSamples=table(); audits=cell(0,16); runFiles=cell(0,1);
for smd=problemNumbers(:)'
    for runNo=1:numberOfRuns
        seed=(smd-1)*numberOfRuns+runNo;
        leaf=fullfile(outputRoot,sprintf('SMD%d',smd),sprintf('run%03d',runNo));
        if ~isfolder(leaf), mkdir(leaf); end
        [mainIns,BI,audit]=mainRun(root,leaf,smd,runNo,seed,options);
        runFile=fullfile(leaf,'refinement_run.mat');
        if isfile(runFile)
            saved=load(runFile,'details','sampleTable','complete');
            assert(saved.complete); details=saved.details; sampleTable=saved.sampleTable;
            fprintf('Reusing refinement SMD%d run %d.\n',smd,runNo);
        else
            events=mainIns.predictionModelQualityCapture.events;
            first=find(cellfun(@(e)e.generation>=10&&e.updateApplied,events),1);
            details=table(); sampleRows=struct([]); stateFiles=cell(0,1); fullArchive=zeros(0,5); index=0;
            for j=1:numel(events)
                event=events{j}; rows=event.history(event.history(:,1)==event.generation,:);
                fullArchive=[fullArchive;rows]; %#ok<AGROW>
                window=fullArchive(fullArchive(:,1)>=max(1,event.generation-9),:);
                assert(isequal(window,event.history),'Full archive reconstruction differs from original history window.');
                if event.generation<10||~event.updateApplied||j==numel(events), continue; end
                index=index+1; event.history=fullArchive; nextEvent=events{j+1};
                stateFile=fullfile(leaf,sprintf('state_gen%04d.mat',event.generation));
                if isfile(stateFile)
                    saved=load(stateFile,'part','sample'); part=saved.part; sample=saved.sample;
                else
                    beforeRng=rng; beforeU=ulFunctionEvaluations; beforeL=llFunctionEvaluations;
                    [part,stateResult,sample]=evaluatePredictionRefinementState(event,nextEvent,BI, ...
                        qualityOptions,options,smd,runNo,seed,index,mainIns.UFEs,events{first}.UFE);
                    assert(isequal(beforeRng,rng)&&isequaln(beforeU,ulFunctionEvaluations)&& ...
                        isequaln(beforeL,llFunctionEvaluations),'Refinement diagnostics leaked RNG/counters.');
                    sample.state_total_seconds=stateResult.state_total_seconds;
                    % Models live once, in result.models, not twice in prepared.
                    stateResult.prepared=rmfield(stateResult.prepared,{'models','modelShared'});
                    save(stateFile,'part','sample','stateResult','-v7');
                end
                details=[details;part]; %#ok<AGROW>
                if isempty(sampleRows), sampleRows=sample; else, sampleRows(end+1,1)=sample; end %#ok<AGROW>
                stateFiles{end+1,1}=relative(stateFile,outputRoot); %#ok<AGROW>
                if mod(index,20)==0
                    fprintf('SMD%d run%d generation%d: P1 valid %d, group valid %d/%d.\n', ...
                        smd,runNo,event.generation,sample.P1_valid,sum(part.prediction_valid),height(part));
                end
            end
            sampleTable=struct2table(sampleRows); complete=true;
            save(runFile,'details','sampleTable','stateFiles','complete','config','-v7.3');
            writetable(details,fullfile(leaf,'prediction_refinement_details.csv'));
        end
        allDetails=[allDetails;details]; allSamples=[allSamples;sampleTable]; %#ok<AGROW>
        runFiles{end+1,1}=relative(runFile,outputRoot); %#ok<AGROW>
        audits(end+1,:)={smd,runNo,seed,mainIns.UFEs,mainIns.LFEs, ...
            abs(mainIns.UF-BI.u_fopt),abs(mainIns.LF-BI.l_fopt),audit.verified, ...
            audit.normal_trajectory_equal,audit.rng_equal,audit.reused_main, ...
            height(sampleTable),sum(sampleTable.P1_valid),sum(details.extra_UFE), ...
            sum(details.extra_LFE),sum(sampleTable.state_total_seconds)}; %#ok<AGROW>
        fprintf('Completed SMD%d run%d: %d eligible P1 states / %d opportunities.\n', ...
            smd,runNo,sum(sampleTable.P1_valid),height(sampleTable));
    end
    writetable(allDetails,fullfile(outputRoot,'prediction_refinement_details.csv'));
    writetable(allSamples,fullfile(outputRoot,'prediction_refinement_sample_analysis.csv'));
end
auditTable=cell2table(audits,'VariableNames',{'SMD','run_no','seed','main_UFE','main_LFE', ...
    'final_upper_error','final_lower_error','isolation_verified','main_trajectory_equal', ...
    'main_rng_equal','reused_main','all_prediction_opportunities','eligible_P1_opportunities', ...
    'diagnostic_UFE','diagnostic_LFE','diagnostic_seconds'});
writetable(auditTable,fullfile(outputRoot,'prediction_refinement_main_run_audit.csv'));
analysis=summarisePredictionRefinement(allDetails,allSamples,auditTable,options,outputRoot);
experiment=struct('config',config,'outputRoot',outputRoot,'details',allDetails,'samples',allSamples, ...
    'audit',auditTable,'analysis',analysis,'runFiles',{runFiles});
save(fullfile(outputRoot,'prediction_refinement_results.mat'),'experiment','-v7.3');
writePredictionRefinementReport(experiment);
fprintf('Prediction refinement complete: %s\n',outputRoot);
end

function [ins,BI,audit]=mainRun(root,leaf,smd,runNo,seed,options)
file=fullfile(leaf,'main_run.mat');
if isfile(file)
    saved=load(file,'mainIns','BI','audit'); ins=saved.mainIns; BI=saved.BI; audit=saved.audit; return;
end
audit=struct('verified',false,'normal_trajectory_equal',false,'rng_equal',false,'reused_main',false);
priorRoot=fullfile(root,'results_prediction_model_quality_2D','SMD1_to_SMD8_10runs');
priorFile=fullfile(priorRoot,sprintf('SMD%d',smd),sprintf('run%03d',runNo),'main_run.mat');
if options.reuseMainResults&&isfile(priorFile)&&seed==(smd-1)*10+runNo
    prior=load(priorFile,'mainIns','BI','audit','mainRngAfter');
    sourcesMatch=true;
    for source={'BLCMAES.m','predictionModelQualityCapture.m'}
        currentText=fileread(fullfile(root,source{1}));
        priorSource=fullfile(priorRoot,'source_snapshot',source{1});
        sourcesMatch=sourcesMatch&&isfile(priorSource)&&strcmp(currentText,fileread(priorSource));
    end
    if sourcesMatch&&all(isfield(prior,{'audit','BI','mainIns','mainRngAfter'}))&&prior.audit.verified && ...
            prior.audit.normal_trajectory_equal&&prior.audit.rng_equal && ...
            prior.BI.u_dim==2&&prior.BI.l_dim==3&&strcmpi(prior.BI.fn,sprintf('smd%d',smd)) && ...
            prior.BI.UmaxFEs==2500&&prior.BI.LmaxFEs==250 && ...
            prior.BI.UmaxImprFEs==350&&prior.BI.LmaxImprFEs==25
        ins=prior.mainIns; BI=prior.BI; mainRngAfter=prior.mainRngAfter;
        audit=prior.audit; audit.reused_main=true;
    else
        fprintf('Prior main SMD%d/run%d is incompatible; running fresh A0.\n',smd,runNo);
    end
end
if ~audit.reused_main
    BI=getBLOPinfo('SMD',smd,5);
    BI.UmaxFEs=2500; BI.UmaxImprFEs=350; BI.LmaxFEs=250; BI.LmaxImprFEs=25;
    BI.enablePredictionCorrection=false; BI.predictionCorrection=struct('logEnabled',false);
    BI.enableIndividualPredictionCorrection=false;
    BI.individualPredictionCorrection=struct('mode','A0','logEnabled',false);
    BI.predictionInjection=struct('mode','A0','logEnabled',false);
    BI.predictionModelQualityDiagnostic=struct('enabled',true); BI.generationDiagnostic=struct('enabled',true);
    rng(seed,'twister'); ins=BLCMAES(BI); mainRngAfter=rng;
    if options.verifyMainTrajectory
        plainBI=BI; plainBI.predictionModelQualityDiagnostic.enabled=false;
        rng(seed,'twister'); plain=BLCMAES(plainBI); plainRng=rng;
        audit.normal_trajectory_equal=isequaln(rmfield(ins,'predictionModelQualityCapture'),plain);
        audit.rng_equal=isequaln(mainRngAfter,plainRng);
        assert(audit.normal_trajectory_equal&&audit.rng_equal,'A0 main trajectory changed.');
        audit.verified=true;
    end
end
mainIns=ins; save(file,'mainIns','BI','audit','mainRngAfter','-v7.3');
end

function snapshotSources(root,out)
files={'BLCMAES.m','predictionModelQualityCapture.m','fitLocalUpperQuadratic.m', ...
    'predictionModelQualityOptions.m','predictionInjectionOptions.m', ...
    'BLEA_D/ulTestProblem.m','BLEA_D/llTestProblem.m','BLEA_D/getBLOPinfo.m', ...
    'fitPredictionQualityModels.m','evaluatePredictionModelQualityState.m', ...
    'summarisePredictionModelQuality.m','evaluatePredictionScaleDiagnosticCandidate.m', ...
    'predictionRefinementOptions.m','preparePredictionRefinementState.m', ...
    'evaluatePredictionRefinementState.m','predictionRefinementSampleAnalysis.m', ...
    'predictionRefinementRelations.m','runSMDPredictionRefinementExperiment.m', ...
    'summarisePredictionRefinement.m','writePredictionRefinementReport.m', ...
    'testPredictionRefinementExperiment.m'};
destination=fullfile(out,'source_snapshot'); if ~isfolder(destination), mkdir(destination); end
for i=1:numel(files)
    target=fullfile(destination,files{i}); source=fullfile(root,files{i});
    if ~isfolder(fileparts(target)), mkdir(fileparts(target)); end
    if isfile(target)
        assert(strcmp(fileread(target),fileread(source)),'Source changed: %s. Use a fresh output directory.',files{i});
    else, copyfile(source,target); end
end
end
function path=relative(file,root)
path=strrep(file(numel(root)+2:end),filesep,'/');
end
function restore(U,L,state)
global ulFunctionEvaluations llFunctionEvaluations;
ulFunctionEvaluations=U; llFunctionEvaluations=L; rng(state);
end
