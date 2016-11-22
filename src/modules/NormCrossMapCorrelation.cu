extern "C"
{
#include <lualib.h>
#include <lauxlib.h>
#include <lua.h>
}

//#include "THCUNN.h"
#include "common.h"
#include "stdio.h"
#include "math.h"
#include <THC/THC.h> 
#include <THC/THCApply.cuh>

#define ELEMENT(array, i, j, k, limitHeight, limitWidth)  (array[((i) * ((limitHeight) * (limitWidth))) + ((j) * (limitWidth)) + (k)])
#define CHECKIN  printf("At line number : %d : %s \n", __LINE__, __FILE__)

// Define this to turn on error checking
#define CUDA_ERROR_CHECK
//#define DEBUG_LEVEL 1
#define MALLOC_LIMIT  2047*1024*1024

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define MAX_THREADS 144

extern "C" {
    
bool NormCrossMapCorrelation_IsMallocSet = false;


/**
 * API to call Cuda APIs safely
 * @param err
 * @param file
 * @param line
 */
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

/**
 * API to check the last returned cuda error
 * @param file
 * @param line
 */
inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

/**
 * API to report the memory usage of the GPU
 */
static void reportMemStatus() {

	// show memory usage of GPU
	size_t free_byte;
	size_t total_byte;
	size_t malloc_byte;

	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

	if (cudaSuccess != cuda_status) {
		printf("Error: cudaMemGetInfo fails, %s \n",
				cudaGetErrorString(cuda_status));
		return;
	}

	cuda_status = cudaDeviceGetLimit(&malloc_byte, cudaLimitMallocHeapSize);
	if (cudaSuccess != cuda_status) {
			printf("Error: cudaDeviceGetLimit fails, %s \n",
					cudaGetErrorString(cuda_status));
			return;
	}

	double free_db = (double) free_byte;
	double total_db = (double) total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB, malloc limit = %f MB\n",
			used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
			total_db / 1024.0 / 1024.0, malloc_byte / 1024.0 / 1024.0);

}

/**
 * API to set the malloc limit of GPU
 */
static void setMallocLimit() {

	// cudaDeviceSetLimit can be called only once to set the malloc limit
	// this if loop is to prevent multiple calls of cudaDeviceSetLimit
	if(!NormCrossMapCorrelation_IsMallocSet)
	{
		cudaError_t	cuda_status = cudaDeviceSetLimit(cudaLimitMallocHeapSize, MALLOC_LIMIT);
		if (cudaSuccess != cuda_status) {
				printf("Error: cudaDeviceSetLimit fails, %s \n",
						cudaGetErrorString(cuda_status));
				return;
		}

		NormCrossMapCorrelation_IsMallocSet = true;
	}
}


__device__
void NCMC_PrintArray(float* array, int volume, int rows, int columns)
{
 
    for(int volumeIndex = 0; volumeIndex < volume; volumeIndex++)
    {
        //#pragma unroll 5
        for(int rowIndex = 0; rowIndex < rows; rowIndex++)
        {
            //#pragma unroll 10
            for(int columnIndex = 0; columnIndex < columns; columnIndex++)
            {
                printf("%f ", ELEMENT(array, volumeIndex, rowIndex, columnIndex, rows, columns));
            }
            printf("\n");
        }  
         printf("*************\n"); 
    } 
}

__device__
void NCMC_setArrayContentsToZero(float* array, int numRows, int numColumns)
{
    for(int rowIndex = 0; rowIndex < numRows; rowIndex++)
    {
        for(int columnIndex = 0; columnIndex < numColumns; columnIndex++)
        {
            ELEMENT(array, 0, rowIndex, columnIndex, numRows, numColumns) = 0;
        }
    }

}

__device__ void NCMC_getPatchWithMidPoint(float* input, int layerNumber, int rowNumber, int columnNumber, int numLayers, int mapHeight, int mapWidth, int patchwidth, float* focusPatch)
{
    int halfPatchWidth = patchwidth / 2;

    #pragma unroll 5
    for(int rowIndex = -halfPatchWidth; rowIndex <= halfPatchWidth; rowIndex++)
    {
        #pragma unroll 5
        for(int columnIndex = -halfPatchWidth; columnIndex <= halfPatchWidth; columnIndex++)
        {
            int currentMapRowIndex = rowIndex + rowNumber;
            int currentMapColumnIndex = columnIndex + columnNumber;
            
            int focusPatchRowIndex = rowIndex + halfPatchWidth;
            int focusPatchColumnIndex = columnIndex + halfPatchWidth;
                      
            ELEMENT(focusPatch, 0, focusPatchRowIndex, focusPatchColumnIndex, patchwidth, patchwidth) = 0;
            
            if(currentMapRowIndex >= 0 && currentMapRowIndex < mapHeight &&
                currentMapColumnIndex >= 0 && currentMapColumnIndex < mapWidth)
            {
                ELEMENT(focusPatch, 0, focusPatchRowIndex, focusPatchColumnIndex, patchwidth, patchwidth) = 
                        ELEMENT(input, layerNumber, currentMapRowIndex, currentMapColumnIndex, mapHeight, mapWidth);
            }
        }
    }
}


__device__
void NCMC_getMeanAndStd(float* meanMaps, float* stdMaps, int mapHeight, int mapWidth, int patchwidth, int layerNumber, int rowNumber, int columnNumber, float* mean, float* std)
{
	*mean = ELEMENT(meanMaps, layerNumber, rowNumber + 2, columnNumber, mapHeight + 4, mapWidth);
	*std = ELEMENT(stdMaps, layerNumber, rowNumber + 2, columnNumber, mapHeight + 4, mapWidth);
}

__device__ float NCMC_correlatePatch(float* input, int layerNumber, int rowNumber, int columnNumber, int numLayers, int mapHeight, int mapWidth, int patchwidth, int verticalWidth, float* focusPatch, float correlatorMean, float correlatorStd, float* meanMaps, float* stdMaps)
{
    int halfPatchWidth = patchwidth / 2;
    float correlatedValue = 0;
    int N = (patchwidth * patchwidth);
    
    float correlateeMean = 0, correlateeStd = 0;
    
    //get the bottom layer patch's mean and std
    NCMC_getMeanAndStd(meanMaps, stdMaps, mapHeight, mapWidth, patchwidth, layerNumber, rowNumber, columnNumber, &correlateeMean, &correlateeStd);

    //calculate normalization constant
    float normConstant = 1 / ((N-1) * (correlateeStd * correlatorStd));
      
    #pragma unroll 5
    for(int rowIndex = -halfPatchWidth; rowIndex <= halfPatchWidth; rowIndex++)
    {
        #pragma unroll 5
        for(int columnIndex = -halfPatchWidth; columnIndex <= halfPatchWidth; columnIndex++)
        {
			float correlatorVal = 0, correlateeVal = 0;
			
            int currentMapRowIndex = rowIndex + rowNumber;
            int currentMapColumnIndex = columnIndex + columnNumber;
            
            int focusPatchRowIndex = rowIndex + halfPatchWidth;
            int focusPatchColumnIndex = columnIndex + halfPatchWidth;
            
            correlatorVal = ELEMENT(focusPatch, 0, focusPatchRowIndex, focusPatchColumnIndex, patchwidth, patchwidth);
            
            if(currentMapRowIndex >= 0 && currentMapRowIndex < mapHeight &&
                currentMapColumnIndex >= 0 && currentMapColumnIndex < mapWidth)
            {
                 correlateeVal = ELEMENT(input, layerNumber, currentMapRowIndex, currentMapColumnIndex, mapHeight, mapWidth);
            }
            
			correlatedValue += (((correlatorVal - correlatorMean) * (correlateeVal - correlateeMean)) * normConstant);
        }
    }
    
    return correlatedValue;
}

__device__ 
void NCMC_ConstrainedCorrelation(float* input, float* focusPatch, int layerNumber, int rowNumber, int columnNumber, int numLayers, int mapHeight, int mapWidth, float* output, int patchwidth, int verticalWidth, float* meanMaps, float* stdMaps)
{
    //here outputRowNumber is dummy parameter now. since 37x12x5 =~ 2000 threads are not allowed to spawn in CUDA
    int halfVerticalWidth = verticalWidth / 2;
    int halflayerspoint = numLayers / 2;
    float correlatorMean = 0, correlatorStd = 0;
    
    //get the correlator mean and std
    NCMC_getMeanAndStd(meanMaps, stdMaps, mapHeight, mapWidth, patchwidth, layerNumber - halflayerspoint, rowNumber, columnNumber, &correlatorMean, &correlatorStd);

    for(int rowIndex = 0; rowIndex < verticalWidth; rowIndex++)
    { 
        int currentRowIndex = rowNumber - halfVerticalWidth + rowIndex;
                
        //in this particular row, for each column element
        //#pragma unroll 12
        for(int columnIndex = 0; columnIndex < mapWidth; columnIndex++)
        {
            float correlatedValue = NCMC_correlatePatch(input, layerNumber, currentRowIndex, columnIndex, numLayers, mapHeight, mapWidth, patchwidth, verticalWidth, focusPatch, correlatorMean, correlatorStd, meanMaps, stdMaps);

            //store the correlatedvalue in output
            ELEMENT(output, (layerNumber - halflayerspoint) * verticalWidth * mapWidth + rowIndex * mapWidth + columnIndex, rowNumber, columnNumber, mapHeight, mapWidth) = correlatedValue;

        }
    }

}


__global__
void NCMC_calcNormCrossMapCorrelation(float* input, int numLayers, int mapHeight, int mapWidth, float* output, int patchwidth, int verticalWidth, float* meanMaps, float* stdMaps)
{
    //calculate the layer index, row index, column index    
    int blockNumber = blockIdx.x;
    int elementIndex = threadIdx.x;
    
    //determine source input row number , column number
    int layerNumber = blockNumber;
    int inputRowNumber = elementIndex / mapWidth;
    int inputColumnNumber = elementIndex % mapWidth;

   // if(!(layerNumber == 0 && rowNumber == 0 && columnNumber == 0)) return;
    int halflayerspoint = numLayers / 2;

    // get the particular focus patch
    float focusPatch[25];
    NCMC_setArrayContentsToZero(focusPatch, patchwidth, patchwidth);
    
    NCMC_getPatchWithMidPoint(input, layerNumber, inputRowNumber, inputColumnNumber, numLayers, mapHeight, mapWidth, patchwidth, focusPatch);

    // for each element of its neighborhood rows, calculate the patch correlation
    // for this particular element, calculate the constrained crosspatch correlation
    // and save it in 'output' buffer
    //float* input, float* focusPatch, int layerNumber, int rowNumber, int columnNumber, int numLayers, int mapHeight, int mapWidth, float* output, int patchwidth, int verticalWidth
    NCMC_ConstrainedCorrelation(input, focusPatch, layerNumber + halflayerspoint, inputRowNumber, inputColumnNumber, numLayers, mapHeight, mapWidth, output, patchwidth, verticalWidth, meanMaps, stdMaps);

}

__global__
void NCMC_calcMeanAndStdMaps(float* input, int numLayers, int mapHeight, int mapWidth, int patchwidth, int verticalWidth, float* meanMaps, float* stdMaps) 
{
    //calculate the layer index, row index, column index    
    int blockNumber = blockIdx.x;
    int elementIndex = threadIdx.x;
    
    //determine source input row number , column number
    int layerNumber = blockNumber;
    int inputRowNumber = elementIndex / mapWidth;
    int inputColumnNumber = elementIndex % mapWidth;
    int halfVericalWidth = verticalWidth / 2;

    float focusPatch[25];
    NCMC_setArrayContentsToZero(focusPatch, patchwidth, patchwidth);
    
    NCMC_getPatchWithMidPoint(input, layerNumber, inputRowNumber - halfVericalWidth, inputColumnNumber, numLayers, mapHeight, mapWidth, patchwidth, focusPatch);
    
    float sum = 0;
    
    // calculate the mean and standard deviation
    for(int index = 0; index < patchwidth * patchwidth; index++)
    {
		sum += focusPatch[index];
	}
	
	// calculate the mean
	//mapHeight + 4 , since the 2 rows above and 2 rows below are included in correlation
    ELEMENT(meanMaps, layerNumber, inputRowNumber, inputColumnNumber, mapHeight + 4, mapWidth) = sum / (patchwidth * patchwidth);
	float mean = ELEMENT(meanMaps, layerNumber, inputRowNumber, inputColumnNumber, mapHeight + 4, mapWidth);
	float variance = 0;
	
    // calculate the mean and standard deviation
    for(int index = 0; index < patchwidth * patchwidth; index++)
    {
		variance += ((focusPatch[index] - mean) * (focusPatch[index] - mean));
	}
	
    //mapHeight + 4 , since the 2 rows above and 2 rows below are included in correlation + eps value previous: 1e-6
    float std = sqrt(variance / ((patchwidth * patchwidth) - 1)) + 1e-2;
	ELEMENT(stdMaps, layerNumber, inputRowNumber, inputColumnNumber, mapHeight + 4, mapWidth) = std;
}

void updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output, int patchwidth, int verticalWidth, THCudaTensor *meanMaps, THCudaTensor *stdMaps) {

	THCUNN_assertSameGPU(state, 2, input, output);

	THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

	int numLayers = input->size[0];
	int mapHeight = input->size[1];
	int mapWidth = input->size[2];
	
	//get c-pointer address
	float* inputContents = THCudaTensor_data(state, input);

    //set malloc limit to a higher limit
	setMallocLimit();

	//resize output
    int outputNumLayers = (numLayers / 2) * verticalWidth * mapWidth;
    int outputMapHeight =  mapHeight;
    int outputMapWidth = mapWidth;

	// assign memory for mean, std maps
	THCudaTensor_resize3d(state, meanMaps, numLayers, mapHeight + 4, mapWidth);
	THCudaTensor_fill(state, meanMaps, 0);
	THCudaTensor_resize3d(state, stdMaps, numLayers, mapHeight + 4, mapWidth);
	THCudaTensor_fill(state, stdMaps, 0);
	
	//get c-pointer address
	float* meanMapsPtr = THCudaTensor_data(state, meanMaps);
	float* stdMapsPtr = THCudaTensor_data(state, stdMaps);
	
	int totalBlocks = numLayers;
	int totalThreads = (mapHeight + 4) * mapWidth;
	
	NCMC_calcMeanAndStdMaps<<<totalBlocks, totalThreads>>>(inputContents, numLayers, mapHeight, mapWidth, patchwidth, verticalWidth, meanMapsPtr, stdMapsPtr);
		
	THCudaTensor_resize3d(state, output, outputNumLayers, outputMapHeight, outputMapWidth);
    THCudaTensor_fill(state, output, 0);

	//get c-pointer address
	float* outputPtr = THCudaTensor_data(state, output);

	totalBlocks = (numLayers / 2);
	totalThreads = mapHeight * mapWidth;

    NCMC_calcNormCrossMapCorrelation<<<totalBlocks, totalThreads>>>(inputContents, numLayers, mapHeight, mapWidth, outputPtr, patchwidth, verticalWidth, meanMapsPtr, stdMapsPtr);

}

__device__
float NCMC_getCorrelationValue(float* output, int layerIndex, int mapHeight, int mapWidth, int verticalWidth, int relativeCorrelateeRowIndex, int relativeCorrelateeColumnIndex, int currentMapRowIndex, int currentMapColumnIndex)
{
	float correlationValue = 0;
	correlationValue = ELEMENT(output, (layerIndex * verticalWidth * mapWidth) + (relativeCorrelateeRowIndex * mapWidth) + relativeCorrelateeColumnIndex, 
                                currentMapRowIndex, currentMapColumnIndex, mapHeight, mapWidth);
    
	return correlationValue;
}

__device__
void NCMC_retrieveOutputGradient(float* gradOutput, float* currentOutputGradient, int layerNumber, int currentMapRowIndex, int currentMapColumnIndex, int mapHeight, int mapWidth, int patchwidth, int verticalWidth)
{
	int startLayerNumber = layerNumber * verticalWidth * mapWidth;
    #pragma unroll 5
    for(int rowIndex = 0; rowIndex < verticalWidth; rowIndex++)
    {
        #pragma unroll 12
        for(int columnIndex = 0; columnIndex < mapWidth; columnIndex++)
        {
        	int currentLayerNumber = startLayerNumber + (rowIndex * mapWidth) + columnIndex;
            ELEMENT(currentOutputGradient, 0, rowIndex, columnIndex, verticalWidth, mapWidth) = ELEMENT(gradOutput, currentLayerNumber, currentMapRowIndex, currentMapColumnIndex, mapHeight, mapWidth);
        }
        
    }
}

__device__
float NCMC_retrieveAndMultiplyOutputGradWithCorrelator(float* gradOutput, float* currentCorrelator, int layerNumber, int currentMapRowIndex, int currentMapColumnIndex, int mapHeight, int mapWidth, int patchwidth, int verticalWidth)
{
	int startLayerNumber = layerNumber * verticalWidth * mapWidth;
	float inputGradient = 0;

    #pragma unroll 5
    for(int rowIndex = 0; rowIndex < verticalWidth; rowIndex++)
    {
        #pragma unroll 12
        for(int columnIndex = 0; columnIndex < mapWidth; columnIndex++)
        {
        	int currentLayerNumber = startLayerNumber + (rowIndex * mapWidth) + columnIndex;
            inputGradient += ELEMENT(currentCorrelator, 0, rowIndex, columnIndex, verticalWidth, mapWidth) *
            		ELEMENT(gradOutput, currentLayerNumber, currentMapRowIndex, currentMapColumnIndex, mapHeight, mapWidth);
        }

    }

    return inputGradient;
}

__device__
void NCMC_gradientWrtnput(float* input, float* output, float* currentGradInput, int currentMapRowIndex, int currentMapColumnIndex, int layerNumber, int wrtRowNumber, int wrtColumnNumber, int numLayers, int mapHeight, int mapWidth, int patchwidth, int verticalWidth, float* meanMaps, float* stdMaps)
{
    int halfPatchWidth = patchwidth / 2;
    int halfVerticalWidth = verticalWidth / 2;
    int halflayerspoint = numLayers / 2;
    
    // columnIndex = column of the current focus patch's mid pixel
    // wrtColumnNumber = the static pixel point whose gradient we are finding out
    // halfPatchWidth = half of the patch width
    //  0 0 0 0 0
    //  0 0 0 0 0
    //  0 0 x 0 0 
    //  0 0 0 0 0
    //  0 0 0 0 0
    // example : 1 - 3 + 2 = 0
    //           5 - 3 + 2 = 4    
    
    int numberOfZeroColumns = currentMapColumnIndex - wrtColumnNumber;
    
    // gradient w.r.t., x_i = (((y_i - y_bar) / sigma_y) - (corr(x,y) * (x_i - x_bar) / sigma_x)) / ((N-1) * sigma_x)
    float N = patchwidth * patchwidth;
    float x_i = ELEMENT(input, layerNumber - halflayerspoint, wrtRowNumber, wrtColumnNumber, mapHeight, mapWidth);
    float sigma_x = 0, x_bar = 0;
    NCMC_getMeanAndStd(meanMaps, stdMaps, mapHeight, mapWidth, patchwidth, layerNumber - halflayerspoint, currentMapRowIndex, currentMapColumnIndex, &x_bar, &sigma_x);
    float gradX = (x_i - x_bar) / sigma_x;
    float outsideConstant = 1 / ((N-1) * sigma_x);
    int corrTorTeeRowDiff = currentMapRowIndex - wrtRowNumber;
    
    #pragma unroll 5
    for(int rowIndex = 0; rowIndex < verticalWidth; rowIndex++)
    {
        #pragma unroll 12
        for(int columnIndex = 0; columnIndex < mapWidth; columnIndex++)
        {
            int currentRowIndex = wrtRowNumber - halfVerticalWidth + rowIndex;
            int currentColumnIndex = columnIndex - numberOfZeroColumns;
            float y_i = 0, sigma_y = 0, y_bar = 0;
            
            if(currentRowIndex >= 0 && currentRowIndex < mapHeight && currentColumnIndex >= 0 && currentColumnIndex < mapWidth) 
            {
				// here layerNumber > halflayerspoint (i.e., bottom layer)
            	y_i = ELEMENT(input, layerNumber, currentRowIndex, currentColumnIndex, mapHeight, mapWidth);
            }
            
            NCMC_getMeanAndStd(meanMaps, stdMaps, mapHeight, mapWidth, patchwidth, layerNumber, currentRowIndex + corrTorTeeRowDiff, columnIndex, &y_bar, &sigma_y);
            
            float corrValue = NCMC_getCorrelationValue(output, layerNumber - halflayerspoint, mapHeight, mapWidth, verticalWidth, 
												currentRowIndex + corrTorTeeRowDiff - currentMapRowIndex + halfPatchWidth, columnIndex, 
												currentMapRowIndex, currentMapColumnIndex);
            //currentRowIndex + corrTorTeeRowDiff gives the row w.r.t. which correlation is carried out
            //currentRowIndex + corrTorTeeRowDiff - currentMapRowIndex  gives the difference between the wrt-correlation-row and actual midpoint of patch
            //currentRowIndex + corrTorTeeRowDiff - currentMapRowIndex + halfPatchWidth gives any of 0,1,2,3,4
            
            ELEMENT(currentGradInput, 0, rowIndex, columnIndex, verticalWidth, mapWidth) = (((y_i - y_bar) / sigma_y) - (corrValue * gradX)) * outsideConstant;
        }        
    }
    
}

__device__
void NCMC_getGradientForTopLayer(float* input, float* output, int layerNumber, int rowNumber, int columnNumber, int numLayers, int mapHeight, int mapWidth, float* gradOutput, float* gradInput, int patchwidth, int verticalWidth, float* meanMaps, float* stdMaps)
{
    int halfPatchWidth = patchwidth / 2;
    int halflayerspoint = numLayers / 2;
    float inputGradient = 0;
    
    //float* currentCorrelator = new float[verticalWidth * mapWidth];
    float currentCorrelator[60];

    //column index is placed first, as it gives advantage of retrieving output-gradient only once per column
    #pragma unroll 9
    for(int columnIndex = -halfPatchWidth; columnIndex <= halfPatchWidth; columnIndex++)
    {
        // calculate current column number
        int currentMapColumnIndex = columnNumber + columnIndex;

    	#pragma unroll 9
    	for(int rowIndex = -halfPatchWidth; rowIndex <= halfPatchWidth; rowIndex++)
        {
            int currentMapRowIndex = rowNumber + rowIndex;
            
           //if the row and column index is valid, get the corresponding gradient
            if(currentMapRowIndex >= 0 && currentMapRowIndex < mapHeight and
                currentMapColumnIndex >= 0 and currentMapColumnIndex < mapWidth)
            {
                NCMC_setArrayContentsToZero(currentCorrelator, verticalWidth, mapWidth);
                NCMC_gradientWrtnput(input, output, currentCorrelator, currentMapRowIndex, currentMapColumnIndex, 
                                    layerNumber + halflayerspoint, rowNumber, columnNumber, numLayers, mapHeight, mapWidth,
                                    patchwidth, verticalWidth, meanMaps, stdMaps);
                
                //retrieve the output gradient for currentMapRowIndex, currentMapColumnIndex
            	inputGradient += NCMC_retrieveAndMultiplyOutputGradWithCorrelator(gradOutput, currentCorrelator, layerNumber, currentMapRowIndex, currentMapColumnIndex, mapHeight, mapWidth, patchwidth, verticalWidth);
                
            }
        }
    }
    
    ELEMENT(gradInput, layerNumber, rowNumber, columnNumber, mapHeight, mapWidth) = inputGradient;

}


__device__
float NCMC_retrieveGradientWrtCorrelateeInput(float* input, float* output, float* currentGradientForBottom, int layerNumber, int currentRowIndex, 
												int currentColumnIndex, int rawRowIndex, int rowNumber, int columnNumber, int numLayers, int mapHeight,
												int mapWidth, int patchwidth, int verticalWidth, float* meanMaps, float* stdMaps)
{
	float focusPatch[25];
    NCMC_setArrayContentsToZero(focusPatch, patchwidth, patchwidth);
    int halfPatchWidth = patchwidth / 2;
    int halflayerspoint = numLayers / 2;
    
    NCMC_getPatchWithMidPoint(input, layerNumber, currentRowIndex, currentColumnIndex, numLayers, mapHeight, mapWidth, patchwidth, focusPatch);
    float x_bar = 0, sigma_x = 0;
    NCMC_getMeanAndStd(meanMaps, stdMaps, mapHeight, mapWidth, patchwidth, layerNumber, currentRowIndex, currentColumnIndex, &x_bar, &sigma_x);
    
    //determine start index
    int startIndex = 0;
    if(rawRowIndex < 0)
    {
        startIndex = -rawRowIndex;
    }
    else 
    {
        startIndex = 0;
    }
                
    //determine end index
    int endIndex = 0;
    if(rawRowIndex < 0)
    {
        endIndex = patchwidth - 1;
    }
    else{
        endIndex = patchwidth - rawRowIndex - 1;
    }
     
    float inputGradient = 0;
    float N = (patchwidth * patchwidth);
            
    #pragma unroll 5
    for(int reverseRowIndex = startIndex; reverseRowIndex <= endIndex; reverseRowIndex++)
    {
        int focusPatchRowIndex = endIndex - (reverseRowIndex - startIndex);
        
		#pragma unroll 5
        for(int reverseColumnIndex = 0; reverseColumnIndex < patchwidth; reverseColumnIndex++)
        {
            int focusPatchColumnIndex = patchwidth - reverseColumnIndex - 1;
            int inputGradientColumnNumber = (columnNumber + reverseColumnIndex - halfPatchWidth);
            
            if(inputGradientColumnNumber >= 0 && inputGradientColumnNumber < mapWidth) {
				// get the bottom layer's patchmidpoint
				// the rowNumber is the row at which the influential grad elements of focusPatch is there right now.
				// focusPatchRowIndex contains one of (0 1 2 3 4)
				// simply, if focusPatchRowIndex = 2, then bottomCentreRowIndex = rowNumber. go by this intuition				
                int bottomCentreRowIndex = rowNumber - (focusPatchRowIndex - halfPatchWidth);
                float corrValue = NCMC_getCorrelationValue(output, layerNumber, mapHeight, mapWidth, verticalWidth, 
                                                        // currentRowIndex = -4 to 4 row numbers surrounding bottom layer element
                                                        bottomCentreRowIndex - currentRowIndex + halfPatchWidth, inputGradientColumnNumber,
															currentRowIndex, currentColumnIndex);
                
                float x_i = ELEMENT(focusPatch, 0, focusPatchRowIndex, focusPatchColumnIndex, patchwidth, patchwidth);
                float gradX = (x_i - x_bar) / sigma_x;
                
                // determine y related values
                float y_i = ELEMENT(input, layerNumber + halflayerspoint, rowNumber, columnNumber, mapHeight, mapWidth);
                float y_bar = 0, sigma_y = 0;
                NCMC_getMeanAndStd(meanMaps, stdMaps, mapHeight, mapWidth, patchwidth, layerNumber + halflayerspoint, 
                                    bottomCentreRowIndex, inputGradientColumnNumber, &y_bar, &sigma_y);
                
                float outsideConstant = 1/((N-1) * sigma_y);
                float currentGradInput = (gradX - (corrValue * (y_i - y_bar) / sigma_y)) * outsideConstant;
				inputGradient += currentGradInput * ELEMENT(currentGradientForBottom, 0, reverseRowIndex, inputGradientColumnNumber, verticalWidth, mapWidth);
            }
        }
    }    
    
    //delete[] focusPatch;
    return inputGradient;
}

__device__
void NCMC_getGradientForBottomLayer(float* input, float* output, int layerNumber, int rowNumber, int columnNumber, int numLayers, int mapHeight, int mapWidth, float* gradOutput, float* gradInput, int patchwidth, int verticalWidth, float* meanMaps, float* stdMaps)
{
    int halfPatchWidth = patchwidth / 2;
    int halfVerticalWidth = verticalWidth / 2;
    int halflayerspoint = numLayers / 2;
    float inputGradient = 0;
    //float* currentOutputGradient = new float[verticalWidth * mapWidth];
    float currentOutputGradient[5 * 12];
  
    #pragma unroll 9
    for(int rowIndex = -halfVerticalWidth * 2; rowIndex <= halfVerticalWidth * 2; rowIndex++)
    {
        #pragma unroll 12
        for(int columnIndex = 0; columnIndex < mapWidth; columnIndex++)
        {
            int currentRowIndex = rowNumber + rowIndex;
            int currentColumnIndex = columnIndex; 
            
           //if the row and column index is valid, get the corresponding gradient
            if(currentRowIndex >= 0 && currentRowIndex < mapHeight &&
                currentColumnIndex >= 0 && currentColumnIndex < mapWidth) 
            {
                NCMC_setArrayContentsToZero(currentOutputGradient, verticalWidth, mapWidth);
                
                //retrieve the output gradient for currentMapRowIndex, currentMapColumnIndex
                NCMC_retrieveOutputGradient(gradOutput, currentOutputGradient, layerNumber, currentRowIndex, currentColumnIndex, mapHeight, mapWidth, patchwidth, verticalWidth);
                
                // retrieve the gradient with respect to the input for current respective element
                //currentRowIndex, currentColumnIndex - to get the focus patch with corresponding mid point
                inputGradient += NCMC_retrieveGradientWrtCorrelateeInput(input, output, currentOutputGradient, layerNumber, currentRowIndex, 
																		currentColumnIndex, rowIndex, rowNumber, columnNumber, numLayers, mapHeight, 
																		mapWidth, patchwidth, verticalWidth, meanMaps, stdMaps);
            }
        }
    }
    
    //delete[] currentOutputGradient;

    ELEMENT(gradInput, layerNumber + halflayerspoint, rowNumber, columnNumber, mapHeight, mapWidth) = inputGradient;
}

__global__
void NCMC_calcNormCrossMapCorrelationGradInput(float* input, float* output, int numLayers, int mapHeight, int mapWidth, float* gradOutput, float* gradInput, int patchwidth, int verticalWidth, float* meanMaps, float* stdMaps)
{
    //calculate the layer index, row index, column index
    int layerNumber = blockIdx.x;
    int elementIndex = threadIdx.x;
    int halflayerspoint = numLayers / 2;
    
    int rowNumber = elementIndex / mapWidth;
    int columnNumber = elementIndex % mapWidth;
    
    //if(!(layerNumber == 2 && rowNumber == 3 && columnNumber == 0)) return;
    
    if(layerNumber < halflayerspoint) {
        NCMC_getGradientForTopLayer(input, output, layerNumber, rowNumber, columnNumber, numLayers, mapHeight, mapWidth, gradOutput, gradInput, patchwidth, verticalWidth, meanMaps, stdMaps);
    }
    else
    {
        layerNumber = layerNumber - halflayerspoint;
        NCMC_getGradientForBottomLayer(input, output, layerNumber, rowNumber, columnNumber, numLayers, mapHeight, mapWidth, gradOutput, gradInput, patchwidth, verticalWidth, meanMaps, stdMaps);
    }

}

void updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *output, THCudaTensor *gradOutput, THCudaTensor *gradInput, int patchwidth, int verticalWidth, THCudaTensor *meanMaps, THCudaTensor *stdMaps) {

	//determine gradInput sizes
	int numLayers = input->size[0];
	int mapHeight = input->size[1];
	int mapWidth = input->size[2];

	//resize gradInputs
	THCudaTensor_resize3d(state, gradInput, numLayers, mapHeight, mapWidth);
    THCudaTensor_fill(state, gradInput, 0);

	//get elementary datatype pointers
	float* inputContents = THCudaTensor_data(state, input);
	float* outputContents = THCudaTensor_data(state, output);
	float* gradOutputPtr = THCudaTensor_data(state, gradOutput);
	float* gradInputPtr = THCudaTensor_data(state, gradInput);
	float* meanMapsPtr = THCudaTensor_data(state, meanMaps);
	float* stdMapsPtr = THCudaTensor_data(state, stdMaps);

    NCMC_calcNormCrossMapCorrelationGradInput<<<numLayers, mapHeight * mapWidth>>>(inputContents, outputContents, numLayers, mapHeight, mapWidth, gradOutputPtr, gradInputPtr, patchwidth, verticalWidth, meanMapsPtr, stdMapsPtr);
    CudaCheckError();
}

}
