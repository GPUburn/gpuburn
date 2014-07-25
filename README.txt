How to use:


1) Locate calls for your kernel you want to use reliability and surrond it as follow
 which is for example:
 
  testKernel<<< grid, threads, mem_size >>>( d_idata, d_odata);
 

2) Modify the file which contains calls to CUDA kernels as follow
	#include "reliability_cpu.h"					// Add functions related to reliability
	...
	/////////////////////////////////////////
	// BEGIN MODIFICATIONS FOR RELIABILITY //
	/////////////////////////////////////////
	Cartography carto(devID);						// Create a cartography of compute unit for the device N° devID
	carto.set_corrupted_sm(7);						// Set which sm are corrupted. This correspond to the one you want to avoid
	carto.set_corrupted_sm(6);
	carto.set_corrupted_sm(5);
	carto.set_corrupted_sm(4);
	carto.set_corrupted_sm(3);
	carto.set_corrupted_sm(2);
	carto.set_corrupted_sm(1);
	carto.print_sm_cartography();					// Eventually print out the cartography of valid/invalid SM
	carto.set_exec_parameter(grid, threads);		// Recompute grid and threads parameters according to the number of invalid SM
	
	BEGIN_TIMING
	// Execute the kernel with new grid and thread parameters as follow
    testKernel<<< carto.tmp_griddim, carto.tmp_blockdim, mem_size >>>( d_idata, d_odata);
    carto.reset_blk_index();
    END_TIMING
    
    // check if everything went OK, meaning every block were executed on valid SM
    carto.check_nb_of_block_launched();
	/////////////////////////////////////////
	//  END MODIFICATIONS FOR RELIABILITY  //
	/////////////////////////////////////////


2) Modify the file which includes CUDA Kernels as follows
	#define WITH_KERNEL_RELIABILITY 				// To enable reliability check on GPU
	#include "reliability_gpu.h"
	...

	__global__ void testKernel( float* g_idata, float* g_odata) 
	{
	  	__RELIABILITY_BEGIN

		// Do something here

  		__RELIABILITY_END
	}
