/*
 * Class of functions for testing a specific GPU
 *
 */

#ifndef CUDA_STRESSTEST
#define CUDA_STRESSTEST

//#define PROFILING // Define to see the time the kernel takes
#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS // needed for exceptions

#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <ctime> 
#include <sys/time.h>



////////////////////////////////////////////////////////////////////////////////////////////////////
// Name of known Accelerator
////////////////////////////////////////////////////////////////////////////////////////////////////
#define NB_NVIDIA 3
static const char* nvidia_name[NB_NVIDIA] =
{"GeForce", "Tesla", "Quadro"};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Name of kernel to test
////////////////////////////////////////////////////////////////////////////////////////////////////
#define NB_KERNEL 3

static const char* KernelNames[NB_KERNEL] =
{
    "MADKernel",
    "ShMemKernel",
    "REGREGKernel"
};


#define SEPARATOR       "----------------------------------------------------------------------\n"

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define myCudaErrors(err)  __myCudaErrors (err, __FILE__, __LINE__)

inline void __myCudaErrors(cudaError err, const char *file, const int line ){
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);        
    }
}



class StressTest {
private:
	int devID;									// Selected Device
	int kernelID;								// Selected Kernel
	bool	allkernel;							// test all kernel

	int multiProcessorCount;					// Number of Multiprocessor
	int cudaCore;								// Number of CUDA Cores

	int h_iter;									// Number of iteration
  	int vLen, vSize; 							// Size of vector
  	int *h_vecA;								// Pointer to resulting host vector 
  	float *h_vecB, *h_vecC, *h_vecD;			// Pointer to host vector
  	int *d_vecA;								// Pointer to resulting device vector   	
  	float *d_vecB, *d_vecC, *d_vecD;			// Pointer to device vector

  	time_t tbegin;								// Starting time
  	int TimeLimit;								// Time to execute the test
  	
  	int maxtemp;								// Maximum temperature of the device
  	int timewait;								// Time to wait between 2 execution
  												// Used to regulate the maximum temperature
  	
  	std::ostream *output_file;					// Outputfile for the results	

	// Setup the data for the kernel execution
	// seed: seed for the random number generation 
	inline void Init_Data_and_Argument(int seed);

	// Test if there is any error to repport
	// kernelidx: kernel tested
  	inline void CheckError(uint kernelidx);
	
public:

	//Parse the command line to set up the parameters
	void parse_command_line(int argc, char **argv);

	// Initialize the platform, device, and execution parameters
	StressTest();
	~StressTest();
	void DeviceInit();
	
	// Initialize the test environnement
	void InitTest();

	// Core of the test, perform the execution on the device
	void ExecuteTest();
	void ExecKernel(int kID);
};

#endif
