#pragma once
#include "shared.cuh"
#include "work.cuh"

namespace cuda
{
	struct stream_t
	{
		cudaStream_t _v;

		// Creates a handle for default stream
		stream_t() { _v = nullptr; }

		// Creates a new stream
		stream_t( int flags ) { CU_WRAP( cudaStreamCreateWithFlags( &_v, flags ) ); }
		
		// Destroys the current stream (if not default stream)
		~stream_t() { if( _v ) CU_WRAP( cudaStreamDestroy( _v ) ); }

		stream_t( const stream_t& ) = delete;

		// Async memory operations
		void memset_d( void* dst, int value, size_t size ) { CU_WRAP( cudaMemsetAsync( dst, value, size, _v ) ); }
		void memcpy_d2d( void* dst, const void* src, size_t size ) { CU_WRAP( cudaMemcpyAsync( dst, src, size, cudaMemcpyDeviceToDevice, _v ) ); }
		void memcpy_h2d( void* dst, const void* src, size_t size ) { CU_WRAP( cudaMemcpyAsync( dst, src, size, cudaMemcpyHostToDevice, _v ) ); }
		void memcpy_d2h( void* dst, const void* src, size_t size ) { CU_WRAP( cudaMemcpyAsync( dst, src, size, cudaMemcpyDeviceToHost, _v ) ); }

		// Syncronizes with the stream
		void sync() { CU_WRAP( cudaStreamSynchronize( _v ) ); }

		// Casting back to cudaStream_t
		operator cudaStream_t() { return _v; }

		// Runs the given kernel
		template<typename function_t, typename... args_t> 
		void run( function_t* fn, dim3 grid, dim3 block, args_t&&... args ) 
		{ 
			fn<<<grid, block, 0, _v >>>( args... ); CU_WRAP( cudaPeekAtLastError() ); 
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
		template<typename work_dims, typename lambda> 
		void run( lambda l ) 
		{ 
			run<work_dims>( &lambda_kernel<lambda>, l ); 
		}
	};

	// Default stream
	stream_t default_stream = {};
};