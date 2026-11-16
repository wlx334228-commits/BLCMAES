function selection = selectPredictionScaleDiagnosticEvents( ...
    states,samplingMode,diagnosticEveryNValidEvents,maxEventsPerPhase)
%SELECTPREDICTIONSCALEDIAGNOSTICEVENTS Deterministic, quality-blind choice.
% 'all' evaluates every valid event. 'interval' keeps every Nth valid event.
% 'uniform_per_phase' takes up to maxEventsPerPhase events from each real
% phase, at equally spaced phase ranks, separately in every independent run.
% Selection uses only generation order and completed-run phase labels; it
% never reads UF or any candidate-quality quantity.

if nargin < 2 || isempty(samplingMode), samplingMode = 'uniform_per_phase'; end
if nargin < 3 || isempty(diagnosticEveryNValidEvents)
    diagnosticEveryNValidEvents = 1;
end
if nargin < 4 || isempty(maxEventsPerPhase), maxEventsPerPhase = 3; end
samplingMode = normaliseSamplingMode(samplingMode);
validateattributes(diagnosticEveryNValidEvents,{'numeric'}, ...
    {'scalar','integer','positive'});
validateattributes(maxEventsPerPhase,{'numeric'}, ...
    {'scalar','integer','positive'});

template = struct('selected',false,'reason','not_selected', ...
    'sampling_mode',samplingMode,'stage_rank',NaN,'stage_event_count',NaN);
selection = repmat(template,numel(states),1);
if isempty(states), return; end

switch samplingMode
    case 'all'
        for index = 1:numel(states)
            selection(index).selected = true;
            selection(index).reason = 'all_valid_events';
            selection(index).stage_rank = stageRank(states,index);
            selection(index).stage_event_count = stageCount(states,states(index).stage);
        end
    case 'interval'
        for index = 1:numel(states)
            selection(index).stage_rank = stageRank(states,index);
            selection(index).stage_event_count = stageCount(states,states(index).stage);
            if mod(states(index).event_ordinal-1,diagnosticEveryNValidEvents) == 0
                selection(index).selected = true;
                selection(index).reason = sprintf('every_%d_valid_events', ...
                    diagnosticEveryNValidEvents);
            end
        end
    case 'uniform_per_phase'
        stageNames = {'early','middle','late'};
        for stageIndex = 1:numel(stageNames)
            indices = find(strcmp({states.stage},stageNames{stageIndex}));
            eventCount = numel(indices);
            if eventCount == 0, continue; end
            for rankIndex = 1:eventCount
                selection(indices(rankIndex)).stage_rank = rankIndex;
                selection(indices(rankIndex)).stage_event_count = eventCount;
            end
            take = min(maxEventsPerPhase,eventCount);
            % Select centres of equally sized phase segments.  Unlike using
            % objective values, this cannot favour good or bad generations.
            localIndices = round(((1:take)-0.5)*eventCount/take);
            localIndices = max(1,min(eventCount,localIndices));
            for localIndex = localIndices
                index = indices(localIndex);
                selection(index).selected = true;
                selection(index).reason = sprintf( ...
                    'uniform_phase_rank_%d_of_%d',localIndex,eventCount);
            end
        end
    otherwise
        error('BLCMAES:DiagnosticSamplingMode', ...
            'Unsupported diagnostic sampling mode: %s',samplingMode);
end
end

function count = stageCount(states,stage)
count = sum(strcmp({states.stage},stage));
end

function rank = stageRank(states,index)
rank = sum(strcmp({states(1:index).stage},states(index).stage));
end

function mode = normaliseSamplingMode(value)
mode = lower(char(value));
allowed = {'all','interval','uniform_per_phase'};
assert(ismember(mode,allowed), ...
    'samplingMode must be ''all'', ''interval'', or ''uniform_per_phase''.');
end
