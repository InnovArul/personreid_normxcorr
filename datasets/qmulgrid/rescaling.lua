require'lfs'
require 'image'

source = './underground_reid';
destination = './train';

--[[
   
   name: rescaleImages
   @param: path
   @return
   
]]--

function rescaleImages(sourcepath, destinationpath)
	--for all the files/folders in the directory
    for file in lfs.dir(sourcepath) do
    	-- if file/folder is not . or ..
        if file ~= "." and file ~= ".." then
        	-- take the current path
            local srcpath = sourcepath..'/'..file
            local destpath = destinationpath..'/'..file
            
            print ("\t processing "..srcpath)
            
            --check the attributes if the file is a directory
            local attr = lfs.attributes (srcpath)
            assert (type(attr) == "table")
            
            -- if file is a directory
            if attr.mode == "directory" then
            	-- create the directory in destination folder
            	paths.mkdir(destpath);
                rescaleImages(srcpath, destpath)
            else
            	--assume that the file is an image
            	--read the image, rescale it, write it to destpath
                extn = paths.extname(srcpath)
                if(extn == 'jpeg') then
                    img = image.load(srcpath)
                    imgRescaled = image.scale(img, 60, 160)
                    image.save(destpath, imgRescaled)
                end
            end
        end
    end
end

rescaleImages(source, destination)
print('processing completed!')
