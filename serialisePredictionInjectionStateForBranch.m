function serial = serialisePredictionInjectionStateForBranch(state)
%SERIALISEPREDICTIONINJECTIONSTATEFORBRANCH Copy mutable state by value.
% RandStream handles are represented by their type/seed/state triple so an
% offline branch cannot mutate the main run's private auxiliary stream.

serial = state;
serial.log = struct([]);
serial.auxiliaryStream = serialiseStream(fieldOrEmpty(state,'auxiliaryStream'));
serial.modelStream = serialiseStream(fieldOrEmpty(state,'modelStream'));
serial.branchStateSerialised = true;
end

function value = fieldOrEmpty(state,name)
if isfield(state,name), value = state.(name); else, value = []; end
end

function description = serialiseStream(stream)
description = struct('present',false,'type','','seed',NaN,'state',[]);
if isempty(stream), return; end
description.present = true;
description.type = stream.Type;
description.seed = stream.Seed;
description.state = stream.State;
end
