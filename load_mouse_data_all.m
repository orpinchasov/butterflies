% TODO: This function must be merged with 'load_mouse_data'

function [T, G, Ang, wake, rem, sws] = load_mouse_data_all(data_path, mouse_name)
%LOAD_MOUSE_DATA Summary of this function goes here
%   Detailed explanation goes here

    [T, G, ~, ~] = LoadCluRes([data_path mouse_name '\' mouse_name]);

    Ang = LoadAng([data_path 'AngFiles\' mouse_name '.ang']);

    wake = dlmread([data_path mouse_name '\' mouse_name '.states.WAKE']); %in sec;
    rem = dlmread([data_path mouse_name '\' mouse_name '.states.REM']); %in sec;
    sws = dlmread([data_path mouse_name '\' mouse_name '.states.SWS']); %in sec;
end

