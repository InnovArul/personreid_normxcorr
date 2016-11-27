require'lfs'
require 'image'
require 'pl'

source = './train'; 

-- set the random number seed
math.randomseed( os.time() )


--[[
   
   name: getRandomNumber
   @param
   @return a random number between lowe and upper, but without the number in exclude
   
]]--

function getRandomNumber(lower, upper, exclude)
    randNumber = math.random(lower, upper);
    while(randNumber == exclude) do
        randNumber = math.random(lower, upper);
    end
    return randNumber;
end

--[[
   
   name: rescaleImages
   @param: path
   @return
   
]]--

function augmentImages(sourcepath)
    
	--for all the files/folders in the directory
    for file in lfs.dir(sourcepath) do
    	-- if file/folder is not . or ..
        -- two level folder structure
        -- source
             -- identity1
                -- image1
                -- image2
             --identity2
                -- image1
                -- image2
                -- ...
            -- ...
            
        if file ~= "." and file ~= ".." then
        	-- take the current path
            local identitypath = sourcepath..'/'..file
            
            print ("\t processing "..identitypath)
            
            --check the attributes if the file is a directory
            local attr = lfs.attributes (identitypath)
            assert (type(attr) == "table")
            
            -- if file is a directory
            if attr.mode == "directory" then
                augmentImages(identitypath)
            else
            	--assume that the file is an image
            	--read the image, translate it, write it to aug-imgfilename-augmentcount.jpg
                img = image.load(identitypath)
                height = img:size(2)
                filename, extension = path.splitext(file)
                
                -- get 5 random translated images
                for index = 1, 5 do
                    imgSavePath = paths.concat(sourcepath, 'aug' .. '-' .. filename .. '-' .. index  .. extension);
                    range = 0.05 * height;
                    randomWidth = getRandomNumber(-range, range)
                    randomHeight = getRandomNumber(-range, range)
                    
                    -- get the random width & random height to translate
                    imgTranslated = image.translate(img, randomWidth, randomHeight)
                    image.save(imgSavePath, imgTranslated)
                    print('saved image with translation : ' .. randomWidth .. 'x' .. randomHeight)
                    --io.read()
                end
                
                -- save the flipped image
                imgFlipped = image.hflip(img)
                imgSavePath = paths.concat(sourcepath, 'flip' .. '-' .. file);
                image.save(imgSavePath, imgFlipped)
                print('saved image with flip')
                --io.read()
            end
        end
    end
end

augmentImages(source)
