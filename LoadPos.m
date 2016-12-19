function [Pos] = LoadPos(FileName);

Fp = fopen(FileName, 'r');

if Fp==-1
    error(['Could not open file ' FileName]);
end

Pos = fscanf(Fp, '%f %f %f %f');
fclose(Fp);