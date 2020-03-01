#include "includes\cudapp"
#include <iostream>

int main()
{
	cuda::initialize( 1 );


	printf( "Device count: %d\n", cuda::device::count.get() );

	cuda::run<dimt<4, 2>>( [ ] __device__ ()
	{
		printf( "[%d] Hello from GPU %d!\n", cuda::work_idx().y, cuda::work_idx().x );
	} );

	cuda::device::sync();

	return 0;
}