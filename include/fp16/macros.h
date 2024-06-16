#pragma once
#ifndef FP16_MACROS_H
#define FP16_MACROS_H


#ifndef FP16_USE_FP16_TYPE
	#if defined(__clang__)
		#if defined(__F16C__) || defined(__aarch64__)
			#define FP16_USE_FP16_TYPE 1
		#endif
	#endif
	#if !defined(FP16_USE_FP16_TYPE)
		#define FP16_USE_FP16_TYPE 0
	#endif  // !defined(FP16_USE_FP16_TYPE)
#endif  // !defined(FP16_USE_FP16_TYPE)

#endif /* FP16_MACROS_H */
