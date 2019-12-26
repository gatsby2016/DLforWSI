% Read the whole region by Openslide
% Input: the svs filename and the level you want to read.
% Output: the rgb image.
function img = ReadRegion_Openslide(svsname, level)

    svs = openslide_open(svsname);
    [w, h] = openslide_get_level_dimensions(svs, level);
    img = openslide_read_region(svs, 0, 0, w, h, level); 
    img = img(:,:,2:end);
end
