function results = runSMDMeanTrajectory(outputDir)
%RUNSMDMEANTRAJECTORY Save one upper-mean trajectory for SMD1 through SMD12.
% Every run uses the (2+3)-dimensional SMD setting and only logs generation,
% m1, and m2. The logger does not alter the optimization procedure.

projectRoot = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(projectRoot,'mean_trajectory_logs');
end
if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

addpath(fullfile(projectRoot,'BLEA_D'));
runNo = 1;
columns = {'generation','m1','m2'};
results = struct();

for smdNo = 1:12
    BI = getBLOPinfo('SMD',smdNo,5); % Total dimension 5 means upper 2 + lower 3.
    BI.UmaxFEs = 2500;
    BI.UmaxImprFEs = 350;
    BI.LmaxFEs = 250;
    BI.LmaxImprFEs = 25;
    BI.u_N = 50;
    BI.l_N = 50;
    BI.meanTrajectoryDiagnostic = struct('enabled',true);

    fileStem = sprintf('SMD%d_run%02d_meanTrajectory',smdNo,runNo);
    csvFile = fullfile(outputDir,[fileStem '.csv']);
    matFile = fullfile(outputDir,[fileStem '.mat']);
    if exist(csvFile,'file') || exist(matFile,'file')
        error('Diagnostic output already exists: %s',fileStem);
    end

    % This sets a reproducible initial state without consuming random values.
    rng(smdNo,'twister');
    ins = BLCMAES(BI);
    trajectory = ins.meanTrajectory;
    trajectoryTable = array2table(trajectory,'VariableNames',columns);
    writetable(trajectoryTable,csvFile);
    save(matFile,'trajectory','columns');

    results.(sprintf('smd%d',smdNo)) = struct( ...
        'trajectory',trajectory, 'csvFile',csvFile, 'matFile',matFile);
    fprintf('SMD%d: saved mean trajectory to\n  %s\n',smdNo,csvFile);
end
