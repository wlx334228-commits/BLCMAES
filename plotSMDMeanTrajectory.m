function plotSMDMeanTrajectory(logFile)
%PLOTSMDMEANTRAJECTORY Plot generation, m1, m2 from one trajectory log.

if nargin < 1 || isempty(logFile)
    error('Provide a CSV or MAT trajectory log file.');
end

[folder,~,extension] = fileparts(logFile);
if isempty(folder)
    logFile = fullfile(pwd,logFile);
end
if strcmpi(extension,'.mat')
    data = load(logFile,'trajectory','columns');
    trajectoryTable = array2table(data.trajectory,'VariableNames',data.columns);
elseif strcmpi(extension,'.csv')
    trajectoryTable = readtable(logFile);
else
    error('logFile must have a .csv or .mat extension.');
end

figure('Name','Upper mean trajectory');
plot(trajectoryTable.m1,trajectoryTable.m2,'-o','LineWidth',1.2, ...
    'MarkerSize',5);
hold on;
for i = 1:height(trajectoryTable)
    text(trajectoryTable.m1(i),trajectoryTable.m2(i), ...
        sprintf('  %d',trajectoryTable.generation(i)),'FontSize',8);
end
grid on;
axis equal;
xlabel('m_1');
ylabel('m_2');
title('Upper CMA-ES mean trajectory');
