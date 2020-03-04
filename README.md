# cuda++
cuda++ is an easy-to-use wrapper for the CUDA C API.

### 1) Samples
There's no documentation yet, but see below for some examples of how cuda++ makes your life easier. You can also browse the source code which is easy to understand and neatly commented.

```C++
// Device properties and routines are neatly wrapped
//
cuda::initialize( cuda::device::count - 1 );
```

```C++
// Modern exception handling
//
try
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
}
```

```C++
// Easily allocate shared memory
//
cuda::resource<myarray_t> output;
cuda::resource<myarray_t> myarray = cuda::make_resource<myarray_t>( { 1.0f, 2.0f } );

cuda::resource<float> carray( 2 );
carray[ 0 ] = 10.0f;
carray[ 1 ] = 5.0f;
carray >> cuda::gpu;
```

```C++
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
```
	
### 2) Requirements
`--extended-lambda` must be added as an additional option to the CUDA C/C++ compiler.