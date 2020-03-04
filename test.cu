#include "includes\cudapp"
#include <iostream>

union myarray_t
{
	struct 
	{ 
		float a, b; 
	};
	float raw[ 2 ];
};

int main()
{
	// Device properties and routines are neatly wrapped
	//
	cuda::initialize( cuda::device::count - 1 );

	// Modern exception handling
	//
	/*try
	{
		float* ptr = nullptr;
		
		cuda::run<dimt<1>>( [ = ] __device__() 
		{ 
			*ptr = 0; 
		} );

		cuda::device::sync();
	}
	catch ( std::exception ex )
	{
		std::cout << "Caught exception: " << ex.what();
		exit( 0 );
	}*/

	// Easily allocate shared memory
	//
	cuda::resource<myarray_t> output;
	cuda::resource<myarray_t> myarray = cuda::make_resource<myarray_t>( { 1.0f, 2.0f } );

	cuda::resource<float> carray( 2 );
	carray[ 0 ] = 10.0f;
	carray[ 1 ] = 5.0f;
	carray >> cuda::gpu;

	// Simple to run kernels with built-in work distribution
	//
	float* d_a = !carray;
	myarray_t* d_b = !myarray;
	myarray_t* d_o = !output;

	cuda::run<dimt<2>>( [ = ] __device__()
	{
		auto idx = cuda::work_idx();
		printf( "[%d] Hello from GPU!\n", idx.x );
		d_o->raw[ idx.x ] = d_a[ idx.x ] * d_b->raw[ idx.x ];
	} );
	
	output << cuda::gpu;
	printf( "[%f, %f] * [%f, %f] = [%f, %f]\n", carray[ 0 ], carray[ 1 ], myarray->a, myarray->b, output->a, output->b );

	cuda::device::sync();
	return 0;
}