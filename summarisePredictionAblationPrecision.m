function [summary,runs] = summarisePredictionAblationPrecision(outputRoot)
%SUMMARISEPREDICTIONABLATIONPRECISION Read saved runs; no optimisation rerun.
% It reports raw |F-F*| errors and the paper convention
% max(|F-F*|, tolerance), together with UFE/LFE median(IQR).

projectRoot = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(outputRoot)
    error('BLCMAES:OutputDirectory', ...
        'Pass a completed results_prediction_ablation directory.');
end
runCsv = fullfile(outputRoot,'ablation_runs.csv');
assert(exist(runCsv,'file') == 2, ...
    'Missing ablation_runs.csv in %s.',outputRoot);
runs = readtable(runCsv);
required = {'SMD','mode','run_no','UFE','LFE_total','mat_file'};
assert(all(ismember(required,runs.Properties.VariableNames)), ...
    'The run table does not have the required fields.');

runCount = height(runs);
runs.LF = NaN(runCount,1);
runs.upper_raw_error = NaN(runCount,1);
runs.lower_raw_error = NaN(runCount,1);
runs.upper_accuracy = NaN(runCount,1);
runs.lower_accuracy = NaN(runCount,1);
for i = 1:runCount
    matFile = char(string(runs.mat_file(i)));
    data = load(matFile,'ins','BI');
    runs.LF(i) = data.ins.LF;
    runs.upper_raw_error(i) = abs(data.ins.UF-data.BI.u_fopt);
    runs.lower_raw_error(i) = abs(data.ins.LF-data.BI.l_fopt);
    runs.upper_accuracy(i) = max(runs.upper_raw_error(i),data.BI.u_ftol);
    runs.lower_accuracy(i) = max(runs.lower_raw_error(i),data.BI.l_ftol);
end

problemNumbers = unique(runs.SMD,'stable')';
modes = unique(runs.mode,'stable')';
names = {'SMD','mode','runs', ...
    'upper_raw_error_median','upper_raw_error_IQR', ...
    'lower_raw_error_median','lower_raw_error_IQR', ...
    'upper_accuracy_median','upper_accuracy_IQR', ...
    'lower_accuracy_median','lower_accuracy_IQR', ...
    'UFE_median','UFE_IQR','LFE_median','LFE_IQR'};
rows = cell(0,numel(names));
for smdNo = problemNumbers
    for modeIndex = 1:numel(modes)
        mode = modes{modeIndex};
        subset = runs(runs.SMD == smdNo & strcmp(runs.mode,mode),:);
        [upperRaw,upperRawIQR] = medianIQR(subset.upper_raw_error);
        [lowerRaw,lowerRawIQR] = medianIQR(subset.lower_raw_error);
        [upperAccuracy,upperAccuracyIQR] = medianIQR(subset.upper_accuracy);
        [lowerAccuracy,lowerAccuracyIQR] = medianIQR(subset.lower_accuracy);
        [ufe,ufeIQR] = medianIQR(subset.UFE);
        [lfe,lfeIQR] = medianIQR(subset.LFE_total);
        rows(end+1,:) = {smdNo,mode,height(subset), ...
            upperRaw,upperRawIQR,lowerRaw,lowerRawIQR, ...
            upperAccuracy,upperAccuracyIQR,lowerAccuracy,lowerAccuracyIQR, ...
            ufe,ufeIQR,lfe,lfeIQR};
    end
end
summary = cell2table(rows,'VariableNames',names);
save(fullfile(outputRoot,'ablation_precision_summary.mat'),'summary','runs');
writetable(runs,fullfile(outputRoot,'ablation_precision_runs.csv'));
writetable(summary,fullfile(outputRoot,'ablation_precision_summary.csv'));
disp(summary);
end

function [middle,spread] = medianIQR(values)
values = sort(values(isfinite(values)));
if isempty(values), middle = NaN; spread = NaN; return; end
middle = percentile(values,50);
spread = percentile(values,75)-percentile(values,25);
end

function value = percentile(sortedValues,percentage)
position = 1 + (numel(sortedValues)-1)*percentage/100;
lower = floor(position); upper = ceil(position);
if lower == upper
    value = sortedValues(lower);
else
    value = sortedValues(lower) + (position-lower)* ...
        (sortedValues(upper)-sortedValues(lower));
end
end
