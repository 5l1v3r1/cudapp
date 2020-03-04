#pragma once
// Load the default configuration
#include "default_config.hpp"

// Makes intellisense recognize CUDA routines
#ifdef __INTELLISENSE__
	#define __CUDACC__
#endif

// Include CUDA headers
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// Makes intellisense not lose it's mind
#ifdef __INTELLISENSE__
	#undef __location__
	#define __location__(...)
#endif

// Some other headers we need
#include <iostream>
#include <utility>

// Directly wrap all CUDA C API that our sub-components use
namespace cuda
{
	// CUDA error handling
	static void throw_if( cudaError_t code, const char* file, int line )
	{
		if ( code != cudaSuccess )
		{
			char buffer[ 1024 ];
			sprintf( buffer, "CUDA reported error '%s' at %s:%d\n", cudaGetErrorString( code ), file, line );
			fprintf( stderr, buffer );
			throw buffer;
		}
	}
	#define CU_WRAP( ... ) cuda::throw_if( (__VA_ARGS__), __FILE__, __LINE__ )

	// Implementation details of GET/SET wrappers
	namespace detail
	{
		template<cudaLimit e>
		struct limit_instance_t
		{
			size_t get() { size_t out; CU_WRAP( cudaDeviceGetLimit( &out, e ) ); return out; }
			void set( size_t v ) { CU_WRAP( cudaDeviceSetLimit( e, v ) ); }
			operator size_t() { return get(); }
			auto operator=( size_t v ) { set( v ); return *this; }
		};

		template<typename t, cudaError_t( *getter )( t* ), cudaError_t( *setter )( t )>
		struct cuda_fn_getset_wrapper_t
		{
			t get() { t out; CU_WRAP( getter( &out ) ); return out; }
			void set( t v ) { CU_WRAP( setter( v ) ); }
			operator t() { return get(); }
			auto operator=( t v ) { set( v ); return *this; }
		};

		template<typename t, cudaError_t( *getter )( t* )>
		struct cuda_fn_get_wrapper_t
		{
			t get() { t out; CU_WRAP( getter( &out ) ); return out; }
			operator t() { return get(); }
		};
	};

	// Device related API
	namespace device
	{
		// Number of compute-capable devices.
		static detail::cuda_fn_get_wrapper_t<int, cudaGetDeviceCount> count;

		// Which device is currently being used.
		static detail::cuda_fn_getset_wrapper_t<int, cudaGetDevice, cudaSetDevice> index;

		namespace limits
		{
			// GPU thread stack size
			static detail::limit_instance_t<cudaLimitStackSize> stack_size;

			// GPU printf FIFO size
			static detail::limit_instance_t<cudaLimitPrintfFifoSize> printf_fifo_size;

			// GPU malloc heap size
			static detail::limit_instance_t<cudaLimitMallocHeapSize> malloc_heap_size;

			// GPU device runtime synchronize depth
			static detail::limit_instance_t<cudaLimitDevRuntimeSyncDepth> dev_runtime_sync_depth;

			// GPU device runtime pending launch count
			static detail::limit_instance_t<cudaLimitDevRuntimePendingLaunchCount> dev_runtime_pending_launch_count;

			// A value between 0 and 128 that indicates the maximum fetch granularity of L2 (in Bytes). This is a hint
			static detail::limit_instance_t<cudaLimitMaxL2FetchGranularity> l2_fetch_granularity;
		};

		// Preferred cache configuration for the current device
		static detail::cuda_fn_getset_wrapper_t<cudaFuncCache, cudaDeviceGetCacheConfig, cudaDeviceSetCacheConfig> func_cache_config;

		// Shared memory configuration for the current device
		static detail::cuda_fn_getset_wrapper_t<cudaSharedMemConfig, cudaDeviceGetSharedMemConfig, cudaDeviceSetSharedMemConfig> shared_memory_config;

		// Flags for the current device.
		static detail::cuda_fn_getset_wrapper_t<unsigned int, cudaGetDeviceFlags, cudaSetDeviceFlags> flags;

		// Wait for compute device to finish.
		static void sync() { CU_WRAP( cudaDeviceSynchronize() ); }

		// Destroy all allocations and reset all state on the current device in the current process.
		static void reset() { CU_WRAP( cudaDeviceReset() ); }

		// Returns numerical values that correspond to the least and greatest stream priorities.
		static std::pair<int, int> get_stream_priority_range()
		{
			std::pair<int, int> out;
			CU_WRAP( cudaDeviceGetStreamPriorityRange( &out.first, &out.second ) );
			return out;
		}

		// Returns information about the compute-device
		static cudaDeviceProp get_properties( int idx = index )
		{
			cudaDeviceProp out;
			CU_WRAP( cudaGetDeviceProperties( &out, idx ) );
			return out;
		}

		// 	Returns information about the device.
		static int get_attribute( cudaDeviceAttr attr, int idx = index )
		{
			int out;
			CU_WRAP( cudaDeviceGetAttribute( &out, attr, idx ) );
			return out;
		}
	};

	static void initialize( int index = 0 )
	{
		device::index = index;
		device::reset();
		CU_WRAP( cudaFree( nullptr ) );
	}
};