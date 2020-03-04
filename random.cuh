#pragma once
#include "shared.cuh"

namespace cuda
{
	struct random_generator_t
	{
		curandState _state;

		// Constructor wrapping curand_init
		__device__ random_generator_t( uint64_t seed, uint64_t subsequence = 0, uint64_t offset = 0 )
		{
			curand_init( seed, subsequence, offset, &_state );
		}
		
		// Simple casting to original types
		__device__ operator curandState*() { return &_state; }
		__device__ operator curandState&() { return _state; }

		// Uniform random generation
		__device__ float uniform() { return curand_uniform( *this ); }
	};
};