#include "includes\cudapp"
#include <iostream>

union myarray_t
{
	struct
	{
		float a;
		float b;
	};
	float raw[ 2 ];
};

int main()
{
	// Basic initialization
	//
	cuda::initialize( 0 );

	// Device properties wrapped for lazy access
	//
	printf( "Device count: %d\n", cuda::device::count.get() );

	// Easly allocate shared memory
	//
	cuda::resource<myarray_t> myarray = cuda::make_resource<myarray_t>( { 1, 2 } );

	cuda::resource<float> carray( 2 );
	carray[ 0 ] = 10.0f;
	carray[ 1 ] = 5.0f;
	carray >> cuda::gpu;

	cuda::resource<myarray_t> output;

	// Simple to run kernels with built-in work distribution
	//
	myarray_t* d_a = !myarray;
	float* d_b = !carray;
	myarray_t* d_o = !output;
	cuda::run<dimt<2>>( [ = ] __device__ ()
	{
		auto idx = cuda::work_idx();
		printf( "[%d] Hello from GPU!\n", idx.x );
		d_o->raw[ idx.x ] = d_a->raw[ idx.x ] * d_b[ idx.x ];
	} );
	
	cuda::device::sync(); // So that printfs are processed
	output << cuda::gpu;

	for ( float f : output->raw ) printf( "%f ", f );
	printf( "\n" );
	return 0;
}