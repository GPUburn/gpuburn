// CUDA runtime
#include <cuda_runtime.h>

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS // needed for exceptions


#include "stress_test.h"

////////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char **argv)
{
	// Create an OpenCL C++ environment
	StressTest ST;
	// parse command line option
	ST.parse_command_line(argc, argv);
	
	// Initialize the device
	ST.DeviceInit();
	
	// Initialize kernel execution
	ST.InitTest( );

	// Start the test on the device
	ST.ExecuteTest( );
}
