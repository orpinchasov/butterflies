function [T, G, Map, Ang, Pos, wake, rem, sws] = load_mouse_data(data_path, mouse_name, varargin)
%LOAD_MOUSE_DATA Summary of this function goes here
%   Detailed explanation goes here

    strings = strsplit(mouse_name, '-');
    [name, id] = deal(strings{:});

    if nargin == 4
        electrode_mapping = varargin{1};
        brain_region = varargin{2};
        
        mouse = electrode_mapping(name);
        mouse_electrode_mapping = mouse(id);
        
        electrode_indices = find(mouse_electrode_mapping == brain_region);
        
        [T, G, Map, ~] = LoadCluRes([data_path mouse_name '\' mouse_name], electrode_indices);
    else
        % Load data from all electrodes
        [T, G, Map, ~] = LoadCluRes([data_path mouse_name '\' mouse_name]);
    end

    Ang = LoadAng([data_path 'AngFiles\' mouse_name '.ang']);
    Pos = LoadPos([data_path 'PosFiles\' mouse_name '.pos']);

    wake = dlmread([data_path mouse_name '\' mouse_name '.states.WAKE']); %in sec;
    rem = dlmread([data_path mouse_name '\' mouse_name '.states.REM']); %in sec;
    sws = dlmread([data_path mouse_name '\' mouse_name '.states.SWS']); %in sec;
end

