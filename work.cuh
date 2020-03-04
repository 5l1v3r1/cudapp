#pragma once
#include "shared.cuh"

// Template wrapper for dim3
template<unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1>
struct dimt
{
	constexpr static unsigned int x = _x, y = _y, z = _z;
	constexpr static unsigned int magnitude = _x * _y * _z;

	__host__ __device__ static dim3 instantiate() { return { _x, _y, _z }; }
	__host__ __device__ operator dim3() { return { _x, _y, _z }; }

	using linear = dimt<magnitude, 1, 1>;
	__host__ __device__ constexpr operator unsigned int() { return magnitude; }
};

namespace cuda
{
	// Work balancing
	//
	constexpr size_t max_threads = 256;

	template<typename work_dims>
	struct work_balancer
	{
		template<bool grid, int dimension_index>
		static constexpr size_t __helper()
		{
			// Try to shift multipliers from grid to blocks WITHOUT adding no-op work
			size_t xg = work_dims::x, yg = work_dims::y, zg = work_dims::z;
			size_t xb = 1, yb = 1, zb = 1;
			while ( zg >= 2 && !( zg & 1 ) && ( xb * yb * zb < CUPP_WORK_BALANCER_MAX_THREADS ) ) zg >>= 1, zb <<= 1;
			while ( yg >= 2 && !( yg & 1 ) && ( xb * yb * zb < CUPP_WORK_BALANCER_MAX_THREADS ) ) yg >>= 1, yb <<= 1;
			while ( xg >= 2 && !( xg & 1 ) && ( xb * yb * zb < CUPP_WORK_BALANCER_MAX_THREADS ) ) xg >>= 1, xb <<= 1;

			// Return the values
			switch ( dimension_index )
			{
				case 0: return grid ? xg : xb;
				case 1: return grid ? yg : yb;
				case 2: return grid ? zg : zb;
			}
		}

		using grid_size = dimt<__helper<true, 0>(), __helper<true, 1>(), __helper<true, 2>()>;
		using block_size = dimt<__helper<false, 0>(), __helper<false, 1>(), __helper<false, 2>()>;

#if CUPP_WORK_BALANCER_ASSERT_IDEAL != 0
		static_assert( work_dims::linear::x <= 4096 || block_size::linear::x >= 256, "Non-ideal work distribution." );
#endif
	};

	// Thread / Block / Work indexing
	//
	static __device__ __forceinline const dim3& thread_idx() { return threadIdx; }
	static __device__ __forceinline const dim3& thread_lim() { return blockDim; }
	static __device__ __forceinline const dim3& block_idx() { return blockIdx; }
	static __device__ __forceinline const dim3& block_lim() { return gridDim; }

	template<bool thread_locality = true>
	static __device__ __forceinline dim3 work_idx()
	{
		return
		{
			thread_locality ? ( threadIdx.x + blockIdx.x * blockDim.x ) : ( threadIdx.x * gridDim.x + blockIdx.x ),
			thread_locality ? ( threadIdx.y + blockIdx.y * blockDim.y ) : ( threadIdx.y * gridDim.y + blockIdx.y ),
			thread_locality ? ( threadIdx.z + blockIdx.z * blockDim.z ) : ( threadIdx.z * gridDim.z + blockIdx.z ),
		};
	}

	// Syncronizes all threads
	static __device__ __forceinline void sync_threads() { __syncthreads(); }

	// Lambda wrapper
	template<typename lambda>
	__global__ void lambda_kernel( lambda fn ) { fn(); }

	// Runs the given kernel
	template<typename function_t, typename... args_t> 
	void run( function_t* fn, const dim3& grid, const dim3& block, args_t&&... args ) 
	{ 
		fn<<<grid, block>>>( args... ); CU_WRAP( cudaPeekAtLastError() ); 
	}
	
	// Runs the given lambda
	template<typename lambda> 
	void run( lambda l, const dim3& grid = { 1 }, const dim3& block = { 1 } ) 
	{ 
		run( &lambda_kernel<lambda>, grid, block, l ); 
	}

	// Runs the given kernel, automatically distributing work
	template<typename work_dims, typename function_t, typename... args_t> 
	void run( function_t* fn, args_t&&... args ) 
	{ 
		run( fn, work_balancer<work_dims>::grid_size::instantiate(), work_balancer<work_dims>::block_size::instantiate(), args... ); 
	}
	
	// Runs the given lambda, automatically distributing work
	template<typename work_dims, typename lambda> void run( lambda l ) 
	{ 
		run<work_dims>( &lambda_kernel<lambda>, l ); 
	}
};