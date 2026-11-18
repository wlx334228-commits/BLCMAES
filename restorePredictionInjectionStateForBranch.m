function state = restorePredictionInjectionStateForBranch(serial)
%RESTOREPREDICTIONINJECTIONSTATEFORBRANCH Recreate branch-private streams.

state = serial;
assert(isfield(serial,'branchStateSerialised') && ...
    serial.branchStateSerialised, ...
    'BLCMAES:ExecutionControl','Resume injection state is not serialised.');
state.auxiliaryStream = restoreStream(serial.auxiliaryStream);
state.modelStream = restoreStream(serial.modelStream);
if isfield(state,'branchStateSerialised')
    state = rmfield(state,'branchStateSerialised');
end
end

function stream = restoreStream(description)
stream = [];
if isempty(description) || ~isfield(description,'present') || ...
        ~description.present
    return;
end
stream = RandStream(description.type,'Seed',description.seed);
stream.State = description.state;
end
