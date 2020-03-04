#pragma once
#include "shared.cuh"

namespace cuda
{
	// Thanks to:
	// https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h

	// Floating point atomicity
	static __device__ __forceinline float atomic_add( float* address, float val )
	{
		unsigned int* ptr = ( unsigned int* ) address;
		unsigned int old, newint, ret = *ptr;
		do
		{
			old = ret;
			newint = __float_as_int( __int_as_float( old ) + val );
		}
		while ( ( ret = atomicCAS( ptr, old, newint ) ) != old );

		return __int_as_float( ret );
	}

	static __device__ __forceinline float atomic_min( float* address, float val )
	{
		int ret = __float_as_int( *address );
		while ( val < __int_as_float( ret ) )
		{
			int old = ret;
			if ( ( ret = atomicCAS( ( int* ) address, old, __float_as_int( val ) ) ) == old )
				break;
		}
		return __int_as_float( ret );
	}

	static __device__ __forceinline float atomic_max( float* address, float val )
	{
		int ret = __float_as_int( *address );
		while ( val > __int_as_float( ret ) )
		{
			int old = ret;
			if ( ( ret = atomicCAS( ( int* ) address, old, __float_as_int( val ) ) ) == old )
				break;
		}
		return __int_as_float( ret );
	}
};