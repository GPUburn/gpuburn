#ifndef _RELIABILITY_CPU_H_
#define _RELIABILITY_CPU_H_

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define myCudaErrors(err)  __myCudaErrors (err, __FILE__, __LINE__)

inline void __myCudaErrors(cudaError err, const char *file, const int line ){
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}

/*********************************************************************
 **************************** CARTOGRAPHY  ***************************  
 *********************************************************************/
struct Cartography{
	// Information related to the considered GPU
	unsigned int nb_of_sm;				// Nb of streaming multiprocessor
	unsigned int nb_of_cu_per_sm;		// Nb of compute unit per SM
	unsigned int maxThreadperBlock;		// Maximum number of thread per block
	unsigned int nb_of_corrupted_sm;	// Nb of corrupted SM
	unsigned int* nb_of_corrupted_cu;	// Table of Nb of corrupted CU per SM
	
	unsigned int* cpu_carto_of_sm; 		// Cartography of corrupted SM (1: valid; 0:invalid)
	unsigned int* cpu_carto_of_cu;		// Cartography of corrupted CU (1: valid; 0:invalid)

	unsigned int* gpu_carto_of_sm; 		// gpu pointer for the cartography of corrupted SM 
	unsigned int* gpu_carto_of_cu;		// gpu pointer for the cartography of corrupted CU
	
	dim3 max_grid_size;

	// Information related to the execution parameter
	// New index for blocks and threads
	dim3 tmp_grid;						// New dimension of the grid to avoid corrupted SM
	dim3 tmp_threads; 					// New dimension of the block to avoid corrupted CU
	unsigned int ori_how_many_blocks;	// Original number of launched block
	
	Cartography(unsigned int);
	~Cartography(void);
	
	void reset_blk_index(void);					// Reset our BlockIndex on GPU
	void set_corrupted_sm(unsigned int idx);	// Set a specific idx SM to state "corrupted"
	void set_nb_of_corrupted_sm(void);			// Compute number of corrupted SM
	int check_nb_of_block_launched(void);		// Functions to test if every block has been taken in charge by a valid block

	void set_exec_parameter(dim3 griddim, dim3 blockdim);	// Compute new dimension to take into account corrupted SM
	void set_sm_cartography(unsigned int *array_of_sm);		// Define a map of corrupted SM ('array_of_sm' has to be of size 'nb_of_sm')
	void set_cu_cartography(unsigned int *array_of_cu);

	void print_sm_cartography(void);	
	void print_cu_cartography(void);	
};

/*********************************************************************
 ************************* TIMING FUNCTIONS  *************************  
 *********************************************************************/
#define BEGIN_TIMING \
    StopWatchInterface *__reliability_timer = 0;\
    sdkCreateTimer( &__reliability_timer );\
    sdkStartTimer( &__reliability_timer );

#define END_TIMING \
	cudaDeviceSynchronize(); \
    sdkStopTimer( &__reliability_timer );\
    fprintf(stderr, "### RELIABILITY EXECUTION TIME: %f (ms)\n", sdkGetTimerValue( &__reliability_timer ) );\
    sdkDeleteTimer( &__reliability_timer );
#endif