% imgName = "Mesona1.JPG";
% imgName = "Statue1.bmp";
imgName = "ours1.jpg";
name = strsplit(imgName, '1');
name = name(1);

path = strcat('../csv/', name, '/3Dpts_', name, '.csv');
f = fopen(path);
out = textscan(f, '%f%f%f', 'delimiter', ',');
fclose(f);
[x, y, z] = deal(out{:});
pts_3d = [x,y,z];

path = strcat('../csv/', name, '/Cam1_2Dpts_', name, '.csv');
f = fopen(path);
out = textscan(f, '%f%f', 'delimiter', ',');
fclose(f);
[x, y] = deal(out{:});
pts_2d = [x,y];

path = strcat('../csv/', name, '/Cam1_P_', name, '.csv');
f = fopen(path);
out = textscan(f, '%f%f%f%f', 'delimiter', ',');
fclose(f);
[p1, p2, p3, p4] = deal(out{:});
P = [p1, p2, p3, p4];


obj_main(pts_3d, pts_2d, P, imgName);
