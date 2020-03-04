#pragma once

// Maximum number of threads work balancer uses
//
#ifndef CUPP_WORK_BALANCER_MAX_THREADS
	#define CUPP_WORK_BALANCER_MAX_THREADS 256
#endif

// If set to 1, work balancer will raise a static assertaion failure when the distribution is not ideal
//
#ifndef CUPP_WORK_BALANCER_ASSERT_IDEAL
	#define CUPP_WORK_BALANCER_ASSERT_IDEAL 0
#endif