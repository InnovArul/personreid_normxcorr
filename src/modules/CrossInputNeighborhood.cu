extern "C"
{
#include <lualib.h>
#include <lauxlib.h>
#include <lua.h>
}

//#include "THCUNN.h"
#include "common.h"
#include "stdio.h"
#include <THC/THC.h> 
#include <THC/THCApply.cuh>

#define ELEMENT(array, i, j, k)  array[((i) * ((rowsCount) * (columnsCount))) + ((j) * (columnsCount)) + (k)]

extern "C" {
    
__global__ void _calcNeighborhoodDifference(const float* input, int layersCount, int rowsCount, int columnsCount, float* output1, float* output2) {

	// get the current layer and element indices
	int currentLayerIndex = blockIdx.x;
	int currentElementIndex = threadIdx.x;

	//printf("element(%d) = %f, element(%d) = %f\n", currentElementIndex, input[currentElementIndex], 12 + currentElementIndex, input[12 + currentElementIndex]);

	// get the correct row and column indices
	int halfLayersCount = layersCount / 2;
	int currentRow = currentElementIndex / columnsCount;
	int currentColumn = currentElementIndex % columnsCount;

	// get the elements that are in focus now
	float focusElementInA = ELEMENT(input, currentLayerIndex, currentRow, currentColumn);
	float focusElementInB = ELEMENT(input, currentLayerIndex + halfLayersCount, currentRow, currentColumn);

	// calculate the neighborhood differences
	for(int xIndex = -2 ; xIndex <= 2; xIndex++) {
		for(int yIndex = -2 ; yIndex <= 2; yIndex++) {
			// A - B calculation
			int currentXIndex = currentRow + xIndex;
			int currentYIndex = currentColumn + yIndex;
			float elementInA = 0, elementInB = 0;

			int outputLayerIndex = (currentLayerIndex * 25)+ (xIndex + 2) * 5 + (yIndex + 2);

			if(currentXIndex >= 0 && currentXIndex < rowsCount && currentYIndex >= 0 && currentYIndex < columnsCount)
			{
				elementInA = ELEMENT(input, currentLayerIndex, currentXIndex, currentYIndex);
				elementInB = ELEMENT(input, currentLayerIndex + halfLayersCount, currentXIndex, currentYIndex);
				//printf("element(%d, %d, %d) = %f, element(%d, %d, %d) = %f\n", currentLayerIndex, currentXIndex, currentYIndex, elementInA, currentLayerIndex + halfLayersCount, currentXIndex, currentYIndex, elementInB);
				//printf("\n", );
			}

			float neighborhoodDiffAtoB = focusElementInA - elementInB;
			float neighborhoodDiffBtoA = focusElementInB - elementInA;

			// place the element in correct position
			ELEMENT(output1, outputLayerIndex, currentRow, currentColumn) = neighborhoodDiffAtoB;
			ELEMENT(output2, outputLayerIndex, currentRow, currentColumn) = neighborhoodDiffBtoA;
		}
	}
}

void updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output1, THCudaTensor *output2) {
    THCUNN_assertSameGPU(state, 3, input, output1, output2);

	int inputLayers = input->size[0];
	int inputWidth = input->size[1];
	int inputHeight = input->size[2];

	float* inputContents = THCudaTensor_data(state, input);

	//resize output
	//for each layer, 25 neighbors (5 x 5) of each pixel
	THCudaTensor_resize3d(state, output1, (inputLayers / 2) * 25, inputWidth, inputHeight);
	THCudaTensor_resize3d(state, output2, (inputLayers / 2) * 25, inputWidth, inputHeight);
	float* outputPtr1 = THCudaTensor_data(state, output1);
	float* outputPtr2 = THCudaTensor_data(state, output2);

	_calcNeighborhoodDifference<<<inputLayers/2, inputWidth * inputHeight>>>(inputContents, inputLayers, inputWidth, inputHeight, outputPtr1, outputPtr2);

}


__global__ void _calcGradInput(const float* input, int layersCount, int rowsCount, int columnsCount, float* gradOutput1, float* gradOutput2, float* gradInput)
{

	// get the current layer and element indices
	int currentLayerIndex = blockIdx.x;
	int currentElementIndex = threadIdx.x;

	//printf("element(%d) = %f, element(%d) = %f\n", currentElementIndex, input[currentElementIndex], 12 + currentElementIndex, input[12 + currentElementIndex]);

	// get the correct row and column indices
	int halfLayersCount = layersCount / 2;
	int currentRow = currentElementIndex / columnsCount;
	int currentColumn = currentElementIndex % columnsCount;

	// get the elements that are in focus now
	float gradForElementInA = 0;
	float gradForElementInB = 0;

	// calculate the neighborhood differences
	for(int xIndex = -2 ; xIndex <= 2; xIndex++) {
		for(int yIndex = -2 ; yIndex <= 2; yIndex++) {
			int outputLayerIndex = (currentLayerIndex * 25)+ (xIndex + 2) * 5 + (yIndex + 2);

			// add positive gradients (independent of currentXIndex, currentYIndex), only dependent on outputLayerIndex
			gradForElementInA += ELEMENT(gradOutput1, outputLayerIndex, currentRow, currentColumn);
			gradForElementInB += ELEMENT(gradOutput2, outputLayerIndex, currentRow, currentColumn);

			// subtract gradients from gradOutput2 (or gradOutput1)
			// to find the correct gradOutput element of current layer, refer below

			//0  0  0  0  0
			//0  0  0  0  0
			//0  0  x  0  0
			//10 9  8  7  6
			//5  4  3  2  1

			int currentNegGradXIndex = currentRow - xIndex;
			int currentNegGradYIndex = currentColumn - yIndex;

			if(currentNegGradXIndex >= 0 && currentNegGradXIndex < rowsCount && currentNegGradYIndex >= 0 && currentNegGradYIndex < columnsCount)
			{
				gradForElementInA -= ELEMENT(gradOutput2, outputLayerIndex, currentNegGradXIndex, currentNegGradYIndex);
				gradForElementInB -= ELEMENT(gradOutput1, outputLayerIndex, currentNegGradXIndex, currentNegGradYIndex);
			}
		}
	}

	ELEMENT(gradInput, currentLayerIndex, currentRow, currentColumn) = gradForElementInA;
	ELEMENT(gradInput, currentLayerIndex + halfLayersCount, currentRow, currentColumn) = gradForElementInB;
}


void updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput1, THCudaTensor *gradOutput2, THCudaTensor *gradInput) {

	//determine gradInput sizes
	int layersCount = input->size[0];
	int rowsCount = input->size[1];
	int columnsCount = input->size[2];

	//resize gradInputs
	THCudaTensor_resize3d(state, gradInput, layersCount, rowsCount, columnsCount);

	//get elementary datatype pointers
	float* inputContents = THCudaTensor_data(state, input);
	float* gradOutputPtr1 = THCudaTensor_data(state, gradOutput1);
	float* gradOutputPtr2 = THCudaTensor_data(state, gradOutput2);
	float* gradInputPtr = THCudaTensor_data(state, gradInput);

	//calculate gradient of final output with respect to each input element
	_calcGradInput<<<layersCount/2, rowsCount * columnsCount>>>(inputContents, layersCount, rowsCount, columnsCount, gradOutputPtr1, gradOutputPtr2, gradInputPtr);

}

}
