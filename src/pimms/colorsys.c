#if !(defined IS_CPU)
#   error "IS_CPU not defined."
#endif
#if !(defined IS_UINT)
#   error "IS_UINT not defined."
#endif
#if !(defined BITS)
#   error "BITS not defined."
#endif
#if !(defined NITEMS)
#   error "NITEMS not defined."
#endif
#if !(defined COUNT)
#   error "COUNT not defined."
#endif
/* type-independent functions */

#if IS_UINT
#if BITS == 8
/* uint8 functions */
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void rgb_to_hsl(
			 __global const unsigned char *r,
			 __global const unsigned char *g,
			 __global const unsigned char *b,
			 __global volatile unsigned char *h,
			 __global volatile unsigned char *s,
			 __global volatile unsigned char *l) {
  unsigned long idx    = (IS_CPU) ? get_global_id(0) * COUNT : get_global_id(0);
  unsigned long stride = (IS_CPU) ? 1 : get_global_size(0);
  half rf, gf, bf, hf, sf, lf;
  for (uint n = 0; n < COUNT; n++, idx+=stride) {
    rf = (half) r[idx]/255.0;
    gf = (half) r[idx]/255.0;
    bf = (half) r[idx]/255.0;
    //TODO:
  }
}
#elif BITS == 16
/* uint16 functions */

#elif BITS == 32
/* uint32 functions */

#else
/* uint64 functions */

#endif
#else
#if BITS == 16
/* float16 (half-precision) functions */

#elif BITS == 32
/* float32 (single-precision) functions */

#else
/* float64 (double-precision) functions */

#endif
#endif
