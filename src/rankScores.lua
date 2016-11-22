--[[
   rankScores.lua
   
   Copyright 2015 Arulkumar <arul.csecit@ymail.com>
      
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
   MA 02110-1301, USA.
   
]]--

matio = require 'matio';
logger = require 'log';
dofile 'utilities.lua'

function rankScores(scoresFile, correctMatchesFile)
    -- read scores 
    if(scoresFile == nil) then
      scoresFile = paths.concat(opt.save, 'scores.mat')
    end
    
    scores = matio.load(scoresFile)
    scores = scores.x;
    --print(scores:size())
    totalGalleryImages = scores:size(2)

    -- read the correct matches
    if(correctMatchesFile == nil) then
      correctMatchesFile = paths.concat(opt.save, 'correctMatches.mat')
    end
    
    correctMatches = matio.load(correctMatchesFile)
    correctMatches = correctMatches.x;
    totalProbeImages = correctMatches:size(1)
    
    logger.trace("total probe images found in result: " .. totalProbeImages);

    -- do the ranking for CMC (cumulative matching characteristics)
    topNs = torch.linspace(1, totalGalleryImages, totalGalleryImages);
    topNs[1] = 1;
    countTopN = #topNs;

    --buffer to hold top indices
    topIndicesBuffer = {};
    
    logger.info('WITH SAME SCORE ==> ')
    cmc = torch.Tensor(countTopN):fill(0);
    previousPassCount = 0;

    for topn = 1, countTopN[1] do
        currentTopN = topNs[topn];
     --   print(currentTopN)
        passCount = 0;
        if(previousPassCount >= totalProbeImages) then
            passCount = previousPassCount;
        else
            for index1 = 1, totalProbeImages do
                --select scores of current row
                currentScores = scores[{{index1}, {}, {}}];
                currentScores = currentScores[1];
                
                determinedScores = torch.Tensor(currentScores:size(1), 1);
                determinedScores:copy(currentScores[{{}, {1}}]);
                
                --sort the scores
                dummy, indices = torch.sort(determinedScores:transpose(1,2), true)
                
                --select top 'topn' indices
                topNIndices, topNrawIndices = getTopNIndices(indices, currentTopN, currentScores[{{}, {3}}])
                
                if(currentTopN == 50) then
                  topIndicesBuffer[index1] = indices[1];
                end
                --[[]if(index1 %5 == 0) then
                    print('currentindex: ' .. index1)
                    print(topNIndices)
                    io.read()
                end   --]]         
                
                --check if the current index1 exists in topN indices
                isExists = torch.sum(topNIndices:eq(correctMatches[index1][1]))
                
                if(isExists ~= 0) then
                    passCount = passCount + 1;
                end
            end --end for
        end
        print(passCount)
        
        cmc[topn] = passCount / totalProbeImages;
        previousPassCount = passCount;
        
        logger.info(topNs[topn] .. ' ==> ' .. cmc[topn] * 100 .. '%')
    end
    
    --[[
    --NOTE: logger.info('WITH DIFFERENT SCORE ==> ')
    cmc = torch.Tensor(countTopN):fill(0);

    for topn = 1, countTopN[1] do
        currentTopN = topNs[topn];
     --   print(currentTopN)
        passCount = 0;
        for index1 = 1, totalProbeImages do
            --select scores of current row
            currentScores = scores[{{index1}, {}, {}}];
            currentScores = currentScores[1];
            
            determinedScores = torch.Tensor(currentScores:size(1), 1);
            determinedScores:copy(currentScores[{{}, {2}}]);
            
            --sort the scores
            dummy, indices = torch.sort(determinedScores:transpose(1,2), false)
            
            --select top 'topn' indices
            topNIndices, topNrawIndices = getTopNIndices(indices, currentTopN, currentScores[{{}, {3}}])
            
            --check if the current index1 exists in topN indices
            isExists = torch.sum(topNIndices:eq(correctMatches[index1][1]))
            
            if(isExists ~= 0) then
                passCount = passCount + 1;
            end
        end
        --NOTE: print(passCount)
        
        cmc[topn] = passCount / totalProbeImages;
        --NOTE: logger.info(topNs[topn] .. ' ==> ' .. cmc[topn] * 100 .. '%')
    end    
    
    --NOTE: logger.info('WITH DIFFERENCE IN SCORES ==> ')
    cmc = torch.Tensor(countTopN):fill(0);

    for topn = 1, countTopN[1] do
        currentTopN = topNs[topn];
     --   print(currentTopN)
        passCount = 0;
        for index1 = 1, totalProbeImages do
            --select scores of current row
            currentScores = scores[{{index1}, {}, {}}];
            currentScores = currentScores[1];
            
            determinedScores = torch.Tensor(currentScores:size(1), 1);
            determinedScores = currentScores[{{}, {1}}] - currentScores[{{}, {2}}];
            
            --sort the scores
            dummy, indices = torch.sort(determinedScores:transpose(1,2), true)
            
            --select top 'topn' indices
            topNIndices, topNrawIndices = getTopNIndices(indices, currentTopN, currentScores[{{}, {3}}])
            
            --check if the current index1 exists in topN indices
            isExists = torch.sum(topNIndices:eq(correctMatches[index1][1]))
            
            if(isExists ~= 0) then
                passCount = passCount + 1;
            end
        end
        --NOTE:print(passCount)
        
        cmc[topn] = passCount / totalProbeImages;
        --NOTE:logger.info(topNs[topn] .. ' ==> ' .. cmc[topn] * 100 .. '%')
    end    
    --]]
    
    return cmc, topIndicesBuffer;
end

--rankScores()
