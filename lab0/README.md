# Vectorization

## Part 1

**The resulting vector utilization**
vector width | vector utilization
:-----------:|:-------------------:
2 |77.7%
4 |71.3%
8 |67.9%
16 |66.3%

**Q1-1:** Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?

**Answer:**

1. The vector utilization decreases when the vector width increases.
2. For reason that when the vector width increases, the number of data that can fill the whole vector decreases, so the vector utilization decreases.

## Part2

**Q2-1:** Fix the code to make sure it uses aligned moves for the best performance.

**Answer：**
Because clang has used packed AVX2 instructions to add 32 bytes at a time, so that I change the `__builtin_assume_aligned `'s size to 32 bytes.
The source code is as follows:

```cpp!
void test1(float* __restrict a, float* __restrict b, float* __restrict c, int N) {
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 32);
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);

  fasttime_t time1 = gettime();
  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
  fasttime_t time2 = gettime();

  double elapsedf = tdiff(time1, time2);
  std::cout << "Elapsed execution time of the loop in test1():\n"
    << elapsedf << "sec (N: " << N << ", I: " << I << ")\n";
}
```

Generated assembly code (partial):

```asm!
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovaps	(%rbx,%rcx,4), %ymm0
	vmovaps	32(%rbx,%rcx,4), %ymm1
	vmovaps	64(%rbx,%rcx,4), %ymm2
	vmovaps	96(%rbx,%rcx,4), %ymm3
	vaddps	(%r15,%rcx,4), %ymm0, %ymm0
	vaddps	32(%r15,%rcx,4), %ymm1, %ymm1
	vaddps	64(%r15,%rcx,4), %ymm2, %ymm2
	vaddps	96(%r15,%rcx,4), %ymm3, %ymm3
	vmovaps	%ymm0, (%r14,%rcx,4)
	vmovaps	%ymm1, 32(%r14,%rcx,4)
	vmovaps	%ymm2, 64(%r14,%rcx,4)
	vmovaps	%ymm3, 96(%r14,%rcx,4)
	addq	$32, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	jne	.LBB0_3
```

---

**Recording:**
Each case is executed 10 times and average
the case number | elapsed execution time
:----------:|:---------------------:
case1 |8.35sec
case2 |2.15sec
case3 |2.15sec

**Q2-2:** What speedup does the vectorized code achieve over the unvectorized code?

**Answer：**

1. Reduce the number of cycles.
2. Utilizing data locality.

---

What additional speedup does using -mavx2 give (AVX2=1 in the Makefile)?

**Skip**

---

What can you infer about the bit width of the default vector registers on the PP machines?

**Answer：** 128bit

---

What about the bit width of the AVX2 vector registers.
**Skip**

---

**Q2-3:** Provide a theory for why the compiler is generating dramatically different assembly.

**Answer：** The compiler performs loop unroll before triggering vectorization during vectorization optimization. The decision to vectorize code is made based on code behavior pattern matching. The modified code can be vectorized because it exactly matches the behavior pattern of the generated assembly instructions, but the behavior pattern of the code before the modification is different so it cannot be vectorized.
The assembly code generated after the modification is as follows：

```asm=
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movaps	(%r15,%rcx,4), %xmm0
	movaps	16(%r15,%rcx,4), %xmm1
	maxps	(%rbx,%rcx,4), %xmm0
	maxps	16(%rbx,%rcx,4), %xmm1
	movaps	%xmm0, (%r14,%rcx,4)
	movaps	%xmm1, 16(%r14,%rcx,4)
	movaps	32(%r15,%rcx,4), %xmm0
	movaps	48(%r15,%rcx,4), %xmm1
	maxps	32(%rbx,%rcx,4), %xmm0
	maxps	48(%rbx,%rcx,4), %xmm1
	movaps	%xmm0, 32(%r14,%rcx,4)
	movaps	%xmm1, 48(%r14,%rcx,4)
	addq	$16, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	jne	.LBB0_3
```

Lines 3-8 of the assembly code behavior exactly match the source code below

```cpp=
 if (b[j] > a[j]) c[j] = b[j];
      else c[j] = a[j];
```
