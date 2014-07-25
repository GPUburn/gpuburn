#include <stdio.h>

// includes, project
//#include <sdkHelper.h>  // helper for shared that are common to CUDA SDK samples
//#include <shrQATest.h>  // This is for automated testing output (--qatest)
//#include <shrUtils.h>
#include <helper_cuda.h>

#ifndef _RELIABILITY_GPU_H_
#define _RELIABILITY_GPU_H_

const int VERBOSE=1;							// For debugging purposes 

/***************************************************************
 ********************  GPU PARAMETERS  *************************
 ***************************************************************/
__device__ unsigned int *sm_cartography;		// Cartography of trusted and untrusted multiprocessor 
__device__ unsigned int *cu_cartography;		// Cartography of trusted and untrusted CU 
__device__ uint3 ori_blockdim; 					// Original blockdim 
__device__ uint3 ori_griddim;					// Original griddim 

__device__ unsigned int blkIdx=0;				// New BlockId 
__device__ unsigned int leftover=0;				// Variable which indicate how block have completed
__shared__ uint3 sBlockIdx;						// BlockIdx
__shared__ unsigned int idx3D;					// Variable used to go above the N° of launched blocks 
												// and keep track of when the work is over. 
/*********************************************************************
 **************************** CARTOGRAPHY  ***************************  
 *********************************************************************/
#include "reliability_cpu.h"

/*********************************************************************
 ****************************  Functions   ***************************  
 *********************************************************************/
Cartography::Cartography(unsigned int devID=0){
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, devID);
		nb_of_sm 			= deviceProp.multiProcessorCount;
		nb_of_cu_per_sm		= _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor); 
		maxThreadperBlock	= deviceProp.maxThreadsPerBlock;
		max_grid_size.x		= deviceProp.maxGridSize[0];
		max_grid_size.y		= deviceProp.maxGridSize[1];
		max_grid_size.z		= deviceProp.maxGridSize[2];
				
		// 1) Memory allocation		
		cpu_carto_of_sm 	= (unsigned int *)malloc(nb_of_sm * sizeof(unsigned int));
		cpu_carto_of_cu 	= (unsigned int *)malloc(nb_of_sm * nb_of_cu_per_sm * sizeof(unsigned int));
		nb_of_corrupted_cu 	= (unsigned int *)malloc(nb_of_sm * sizeof(unsigned int));

		myCudaErrors(cudaMalloc( (void **)&gpu_carto_of_sm, nb_of_sm * sizeof(unsigned int)));
		myCudaErrors(cudaMalloc( (void **)&gpu_carto_of_cu, nb_of_sm * nb_of_cu_per_sm * sizeof(unsigned int)));

		// 2) Setting parameters on GPU
		memset(nb_of_corrupted_cu, 0, nb_of_sm * sizeof(unsigned int));
		nb_of_corrupted_sm = 0;
		for(int i=0; i<nb_of_sm; i++) 
			cpu_carto_of_sm[i]=1;
		for(int i=0; i<nb_of_sm * nb_of_cu_per_sm; i++) 
			cpu_carto_of_cu[i]=1;
		
		// 3) Transfert cartography to the GPU
		myCudaErrors( cudaMemcpy( gpu_carto_of_sm, cpu_carto_of_sm, 
			nb_of_sm * sizeof(unsigned int), cudaMemcpyHostToDevice ));	
		myCudaErrors( cudaMemcpy( gpu_carto_of_cu, cpu_carto_of_cu, 
			nb_of_sm * nb_of_cu_per_sm * sizeof(unsigned int), cudaMemcpyHostToDevice ));	
					
		myCudaErrors( cudaMemcpyToSymbol( sm_cartography, &gpu_carto_of_sm, 
			sizeof(unsigned int *), 0, cudaMemcpyHostToDevice ));						
		myCudaErrors( cudaMemcpyToSymbol( cu_cartography, &gpu_carto_of_cu, 
			sizeof(unsigned int *), 0, cudaMemcpyHostToDevice ));	

		reset_blk_index();	
		cudaDeviceSynchronize();
			
		if(VERBOSE){
			fprintf(stderr,"CARTOGRAPHY CREATION: \n");
			fprintf(stderr,"- nb_of_sm: %d\n", nb_of_sm);
			fprintf(stderr,"- nb_of_cu_per_sm: %d\n", nb_of_cu_per_sm);		 
			fprintf(stderr,"- maxThreadperBlock: %d \n", maxThreadperBlock);
			fprintf(stderr,"- max_grid_size (%d,%d,%d) \n", max_grid_size.x, max_grid_size.y, max_grid_size.z);		 
		}			
	};


Cartography::~Cartography(void){	
	if(VERBOSE)
		fprintf(stderr,"Release cartography\n");

	cudaFree(gpu_carto_of_sm);
	cudaFree(gpu_carto_of_cu);		
	free(nb_of_corrupted_cu);
	free(cpu_carto_of_sm);
	free(cpu_carto_of_cu);
};

void Cartography::reset_blk_index(void){
	unsigned int tmp=0;
	myCudaErrors( cudaMemcpyToSymbol( blkIdx, &tmp, sizeof(unsigned int), 
			0, cudaMemcpyHostToDevice ));							

}

/*
 * Functions for the interface with the GPU execution
 */
// Set execution parameters (griddim & blockdim)
void Cartography::set_exec_parameter(dim3 griddim, dim3 blockdim){
		unsigned long long int how_many_blocks;
		ori_how_many_blocks=griddim.x*griddim.y*griddim.z ;
		
		// Determine how many blocks to launch.
//		how_many_blocks	= nb_of_sm*ceil((double)ori_how_many_blocks / (double)(nb_of_sm - nb_of_corrupted_sm));
		how_many_blocks	= ori_how_many_blocks;
		if (nb_of_corrupted_sm > 0) 
			how_many_blocks	+= nb_of_sm;

		if (how_many_blocks > max_grid_size.x * max_grid_size.y * max_grid_size.z){
				// TODO: launch the kernel again with the extra missing blocks
		} 
		
		// assign as many block in x,y & z dimension
		if 	(how_many_blocks < max_grid_size.x){
			tmp_grid.x = (unsigned int)how_many_blocks;
		}else {	
			tmp_grid.x = max_grid_size.x;
			how_many_blocks = ceil(how_many_blocks/max_grid_size.x);

			if 	(how_many_blocks < max_grid_size.y){
				tmp_grid.y = (unsigned int)how_many_blocks;
			}else {	
				tmp_grid.y = max_grid_size.y;
				tmp_grid.z = ceil(how_many_blocks/max_grid_size.y);
			}
		}
				
		// Nothing to do for block .... yet
		tmp_threads = blockdim;

		// transfert data to GPU
		myCudaErrors( cudaMemcpyToSymbol( ori_blockdim, &blockdim, sizeof(dim3), 
			0, cudaMemcpyHostToDevice ));							
		myCudaErrors( cudaMemcpyToSymbol( ori_griddim, &griddim, sizeof(dim3), 
			0, cudaMemcpyHostToDevice));							
		myCudaErrors( cudaMemcpyToSymbol( leftover, &ori_how_many_blocks, sizeof(unsigned int), 
			0, cudaMemcpyHostToDevice));							
		
#ifndef WITH_KERNEL_RELIABILITY
	// If we don't want to use reliability then set original parameters for the execution of the kernel
	tmp_grid  = griddim;
	tmp_threads =blockdim;
#endif	
		if(VERBOSE){
			fprintf(stderr,"GRID SIZE: \n");
			fprintf(stderr,"- original (%d,%d,%d) \n", griddim.x, griddim.y, griddim.z);		 
			fprintf(stderr,"- modified (%d,%d,%d) \n", tmp_grid.x, tmp_grid.y, tmp_grid.z);
			fprintf(stderr,"BLOCK SIZE: \n");
			fprintf(stderr,"- original (%d,%d,%d) \n", blockdim.x, blockdim.y, blockdim.z);		 
			fprintf(stderr,"- modified (%d,%d,%d) \n", tmp_threads.x, tmp_threads.y, tmp_threads.z);
		}		
} 

/*********************************************************************
 ************************** SM  Functions   **************************  
 *********************************************************************/
// Input: array of SM set to 0 if the corresponding SM is corrupted and 1 if it is valid
// Action: 
// - Compute the number of corrupted SM 
// - Transfert cartography of valid SM to the GPU
void Cartography::set_sm_cartography(unsigned int *array_of_sm){
	memcpy(cpu_carto_of_sm, array_of_sm, nb_of_sm * sizeof(unsigned int));	
	myCudaErrors( cudaMemcpy( gpu_carto_of_sm, cpu_carto_of_sm, 
		nb_of_sm * sizeof(unsigned int), cudaMemcpyHostToDevice ));		

	set_nb_of_corrupted_sm();						
}

void Cartography::set_nb_of_corrupted_sm(){
	nb_of_corrupted_sm = 0;
	for(int i=0; i< nb_of_sm; i++)
		if (cpu_carto_of_sm[i] == 0)
			nb_of_corrupted_sm++;

	if (nb_of_corrupted_sm >= nb_of_sm){
		fprintf(stderr,"ERROR: no SM left to perform computation \n");
		exit(-1);
	} 
}

void Cartography::set_corrupted_sm(unsigned int idx){
	if (idx < nb_of_sm){
		cpu_carto_of_sm[idx]=0;
		myCudaErrors( cudaMemcpy( gpu_carto_of_sm, cpu_carto_of_sm, 
			nb_of_sm * sizeof(unsigned int), cudaMemcpyHostToDevice ));	
		set_nb_of_corrupted_sm();
	}else {
		fprintf(stderr," #### WARNING: SM N°: %d is not a valid number ###\n", idx);
	}
}

int Cartography::check_nb_of_block_launched(void){
#ifdef WITH_KERNEL_RELIABILITY	
	unsigned int cpu_blkIdx;
	cudaDeviceSynchronize();
	myCudaErrors( cudaMemcpyFromSymbol(&cpu_blkIdx , blkIdx, sizeof(unsigned int), 
			0, cudaMemcpyDeviceToHost));

	if(VERBOSE)
		fprintf(stderr,"CHECK: Number of Blocks executed on valid SM: %u (wanted: %u)\n", cpu_blkIdx, ori_how_many_blocks);
	
	if (cpu_blkIdx >=ori_how_many_blocks)
		return 1;
	else{
		fprintf(stderr,"WARNING !!! Number of Blocks executed on valid SM is less than the expected one (%u block executed for %u wanted)\n", cpu_blkIdx, ori_how_many_blocks);
		return 0;
	}
#else
return 1;	
#endif
}

// Input: array of CU for each SM set to 0 if the correcuonding CU is corrupted and 1 if it is valid
// Action: 
// - Compute the number of corrupted CU for each SM 
// - Transfert cartography of valid CU for each SM to the GPU
void Cartography::set_cu_cartography(unsigned int *array_of_cu){	
	for(unsigned int i=0; i<=nb_of_sm; i++){	
		nb_of_corrupted_cu[i] = 0;
		for(unsigned int j=0; i<=nb_of_cu_per_sm; j++)
			if (*(array_of_cu + i*nb_of_sm + j)==0) nb_of_corrupted_cu[i]++;
	}
	
	memcpy(cpu_carto_of_cu, array_of_cu, nb_of_sm * nb_of_cu_per_sm * sizeof(unsigned int));
	myCudaErrors( cudaMemcpy( gpu_carto_of_cu, cpu_carto_of_cu, 	
		nb_of_sm * nb_of_cu_per_sm * sizeof(unsigned int), cudaMemcpyHostToDevice ));	
}


void Cartography::print_sm_cartography(void){
	printf("Cartography of SM (%d corrupted):\n", nb_of_corrupted_sm);
	for(int i=0; i<nb_of_sm; i++)
		printf(" %u", cpu_carto_of_sm[i]);
	printf("\n\n");
}

/***************************************************************
 ********************  GPU PARAMETERS  *************************
 ***************************************************************/
// Each valid SM has to acquire a new Block Index
// WARNING: the number of launched block has to be less than 2^64
__device__ uint3 get_ordered_blockId(){
	unsigned int MaxVal;
	unsigned int private_idx3D;

	if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)){
		unsigned int oldval=atomicAdd(&blkIdx, 0);
		MaxVal = ori_griddim.x * ori_griddim.y * ori_griddim.z;
		// We allow 4 times more block to be launched 
		// this variable has to be adjusted !!!
		idx3D = atomicInc(&blkIdx, 4*MaxVal);
		private_idx3D = idx3D % MaxVal;
		
		sBlockIdx.x = private_idx3D % ori_griddim.x;
		private_idx3D /= ori_griddim.x;
		sBlockIdx.y = private_idx3D % ori_griddim.y;
		private_idx3D /= ori_griddim.y;
		sBlockIdx.z = private_idx3D % ori_griddim.z;	
	}
	
	__syncthreads();
	return sBlockIdx;
}


__device__ void __active_waiting(){
	
	if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)){
		// use atomic to go through the atomic pipeline
		while(atomicAdd(&blkIdx, 0) < ori_griddim.x * ori_griddim.y * ori_griddim.z){
		}
	}
	__syncthreads();
}

/***************************************************************
 ************ PREPROCESSING FOR GPU PARAMETERS  ****************
 ***************************************************************/

#ifdef WITH_KERNEL_RELIABILITY
	// Preprocessing statement to alter block Index
	#define blockIdx sBlockIdx
	// Preprocessing statement to alter block and grid dimension
	#define blockDim ori_blockdim
	#define gridDim ori_griddim


	// Preprocessing statement to put at the beginning of each global kernel
	#define __RELIABILITY_BEGIN \
		unsigned int which_sm;\
		asm volatile("mov.u32 %0, %smid;":"=r"(which_sm));\
		if (sm_cartography[which_sm]){ \
			get_ordered_blockId();\
			if (idx3D < ori_griddim.x * ori_griddim.y * ori_griddim.z){
		
			/*
				if(threadIdx.x==0)\
				printf(" blockId (%d,%d,%d) executed on SM N° %d \n", blockIdx.x, blockIdx.y, blockIdx.z, which_sm);
	*/

	// Preprocessing statement to put at the end of each global kernel
	// corrupted SM will wait for others to complete 
	#define __RELIABILITY_END \
		}}else {\
			__active_waiting();\
		}
#else
	#define __RELIABILITY_BEGIN 
	#define __RELIABILITY_END 
#endif

#endif

