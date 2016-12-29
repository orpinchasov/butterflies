function [T, G, Ang, wake, rem, sws] = load_mouse_data(data_path, mouse_name, electrode_mapping, brain_region)
%LOAD_MOUSE_DATA Summary of this function goes here
%   Detailed explanation goes here

    strings = strsplit(mouse_name, '-');
    [name, id] = deal(strings{:});
    
    mouse = electrode_mapping(name);
    mouse_electrode_mapping = mouse(id);
    
    electrode_indices = find(mouse_electrode_mapping == brain_region);

    [T, G, ~, ~] = LoadCluRes([data_path mouse_name '\' mouse_name], electrode_indices);

    Ang = LoadAng([data_path 'AngFiles\' mouse_name '.ang']);

    wake = dlmread([data_path mouse_name '\' mouse_name '.states.WAKE']); %in sec;
    rem = dlmread([data_path mouse_name '\' mouse_name '.states.REM']); %in sec;
    sws = dlmread([data_path mouse_name '\' mouse_name '.states.SWS']); %in sec;
end

