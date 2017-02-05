#pragma once

/*******************************************************************
*   KfNN.h
*   KfNN
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Oct 13, 2016
*******************************************************************/
//
// Fastest CPU implementation of a brute-force
// matcher for 128-float descriptors such as SIFT
// in 2NN mode, i.e., a match is returned if the best
// match between a query vector and a training vector
// is more than a certain threshold ratio
// better than the second-best match.
// AVX2 and all the SSEs are used to accelerate
// computation.
//
// Check out the CUDA version, CUDAKfNN, for significantly
// more speed.
//
// KfNN supports both raw floats and packed (as uint8_t).
// Just set the 'packed' boolean flag accordingly in the
// demo. For CPU packed is faster; for CUDA it's slower.
// 
// Float descriptors are slow. Check out my K2NN and
// CUDAK2NN projects
// for much faster binary description matching. Use a
// good binary descriptor such as LATCH where possible.
// 
// All functionality is contained in the file KfNN.h.
// 'main.cpp' is simply a sample test harness
// with example usage and performance testing.
//

#pragma once

#include <algorithm>
#include <future>
#include <immintrin.h>
#include <vector>

struct Match {
	int q, t;

	Match() {}
	Match(const int _q, const int _t) : q(_q), t(_t) {}
};

void _bruteMatch_float(const void* __restrict ts, const int tcount, const void* const __restrict qs, std::vector<int>& match_idxs, const float quadrance_thresh, const int start, const int count) {
	const float* __restrict tset = reinterpret_cast<const float* __restrict>(ts);
	const float* __restrict qset = reinterpret_cast<const float* __restrict>(qs);
	for (int q = start; q < start + count; ++q) {
		const int qp = q << 7;
		int best_i = -1;
		float best_v = 10000000.0f;
		float second_v = 20000000.0f;

		for (int t = 0, tp = 0; t < tcount; ++t, tp += 128) {
			__m256 q1 = _mm256_load_ps(qset + qp);
			__m256 q2 = _mm256_load_ps(qset + qp + 8);
			__m256 q3 = _mm256_load_ps(qset + qp + 16);
			__m256 q4 = _mm256_load_ps(qset + qp + 24);
			__m256 q5 = _mm256_load_ps(qset + qp + 32);
			__m256 q6 = _mm256_load_ps(qset + qp + 40);
			__m256 q7 = _mm256_load_ps(qset + qp + 48);
			__m256 q8 = _mm256_load_ps(qset + qp + 56);
			q1 = _mm256_sub_ps(q1, _mm256_load_ps(tset + tp));
			q1 = _mm256_mul_ps(q1, q1);
			q2 = _mm256_sub_ps(q2, _mm256_load_ps(tset + tp + 8));
			q2 = _mm256_mul_ps(q2, q2);
			q1 = _mm256_add_ps(q1, q2);
			q3 = _mm256_sub_ps(q3, _mm256_load_ps(tset + tp + 16));
			q3 = _mm256_mul_ps(q3, q3);
			q4 = _mm256_sub_ps(q4, _mm256_load_ps(tset + tp + 24));
			q4 = _mm256_mul_ps(q4, q4);
			q3 = _mm256_add_ps(q3, q4);
			q1 = _mm256_add_ps(q1, q3);
			q5 = _mm256_sub_ps(q5, _mm256_load_ps(tset + tp + 32));
			q5 = _mm256_mul_ps(q5, q5);
			q6 = _mm256_sub_ps(q6, _mm256_load_ps(tset + tp + 40));
			q6 = _mm256_mul_ps(q6, q6);
			q5 = _mm256_add_ps(q5, q6);
			q7 = _mm256_sub_ps(q7, _mm256_load_ps(tset + tp + 48));
			q7 = _mm256_mul_ps(q7, q7);
			q8 = _mm256_sub_ps(q8, _mm256_load_ps(tset + tp + 56));
			q8 = _mm256_mul_ps(q8, q8);
			q7 = _mm256_add_ps(q7, q8);
			q5 = _mm256_add_ps(q5, q7);
			__m256 res1 = _mm256_add_ps(q1, q5);

			q1 = _mm256_load_ps(qset + qp + 64);
			q2 = _mm256_load_ps(qset + qp + 72);
			q3 = _mm256_load_ps(qset + qp + 80);
			q4 = _mm256_load_ps(qset + qp + 88);
			q5 = _mm256_load_ps(qset + qp + 96);
			q6 = _mm256_load_ps(qset + qp + 104);
			q7 = _mm256_load_ps(qset + qp + 112);
			q8 = _mm256_load_ps(qset + qp + 120);
			q1 = _mm256_sub_ps(q1, _mm256_load_ps(tset + tp + 64));
			q1 = _mm256_mul_ps(q1, q1);
			q2 = _mm256_sub_ps(q2, _mm256_load_ps(tset + tp + 72));
			q2 = _mm256_mul_ps(q2, q2);
			q1 = _mm256_add_ps(q1, q2);
			q3 = _mm256_sub_ps(q3, _mm256_load_ps(tset + tp + 80));
			q3 = _mm256_mul_ps(q3, q3);
			q4 = _mm256_sub_ps(q4, _mm256_load_ps(tset + tp + 88));
			q4 = _mm256_mul_ps(q4, q4);
			q3 = _mm256_add_ps(q3, q4);
			q1 = _mm256_add_ps(q1, q3);
			q5 = _mm256_sub_ps(q5, _mm256_load_ps(tset + tp + 96));
			q5 = _mm256_mul_ps(q5, q5);
			q6 = _mm256_sub_ps(q6, _mm256_load_ps(tset + tp + 104));
			q6 = _mm256_mul_ps(q6, q6);
			q5 = _mm256_add_ps(q5, q6);
			q7 = _mm256_sub_ps(q7, _mm256_load_ps(tset + tp + 112));
			q7 = _mm256_mul_ps(q7, q7);
			q8 = _mm256_sub_ps(q8, _mm256_load_ps(tset + tp + 120));
			q8 = _mm256_mul_ps(q8, q8);
			q7 = _mm256_add_ps(q7, q8);
			q5 = _mm256_add_ps(q5, q7);
			res1 = _mm256_add_ps(res1, _mm256_add_ps(q1, q5));

			__m128 x = _mm_add_ps(_mm256_extractf128_ps(res1, 1), _mm256_castps256_ps128(res1));
			__m128 shuf = _mm_movehdup_ps(x);
			__m128 sums = _mm_add_ps(x, shuf);
			shuf = _mm_movehl_ps(shuf, sums);
			const float score = _mm_cvtss_f32(_mm_add_ss(sums, shuf));

			if (score < second_v) second_v = score;
			if (score < best_v) {
				second_v = best_v;
				best_v = score;
				best_i = t;
			}
		}

		match_idxs[q] = (best_v > quadrance_thresh * second_v) ? -1 : best_i;
	}
}

 void _bruteMatch_uint8_t(const void* __restrict ts, const int tcount, const void* const __restrict qs, std::vector<int>& match_idxs, const float quadrance_thresh, const int start, const int count) {
	 const uint8_t* __restrict tset = reinterpret_cast<const uint8_t* __restrict>(ts);
	 const uint8_t* __restrict qset = reinterpret_cast<const uint8_t* __restrict>(qs);
	 for (int q = start; q < start + count; ++q) {
		const int qp = q << 7;
		int best_i = -1;
		int best_v = 100000000;
		int second_v = 200000000;

		for (int t = 0, tp = 0; t < tcount; ++t, tp += 128) {
			__m256i q1 = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(qset + qp)));
			__m256i q2 = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(qset + qp + 16)));
			__m256i q3 = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(qset + qp + 32)));
			__m256i q4 = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(qset + qp + 48)));
			__m256i q5 = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(qset + qp + 64)));
			__m256i q6 = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(qset + qp + 80)));
			__m256i q7 = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(qset + qp + 96)));
			__m256i q8 = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(qset + qp + 112)));

			q1 = _mm256_sub_epi16(q1, _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(tset + tp))));
			q1 = _mm256_madd_epi16(q1, q1);
			q2 = _mm256_sub_epi16(q2, _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(tset + tp + 16))));
			q2 = _mm256_madd_epi16(q2, q2);
			q3 = _mm256_sub_epi16(q3, _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(tset + tp + 32))));
			q3 = _mm256_madd_epi16(q3, q3);
			q4 = _mm256_sub_epi16(q4, _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(tset + tp + 48))));
			q4 = _mm256_madd_epi16(q4, q4);
			q5 = _mm256_sub_epi16(q5, _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(tset + tp + 64))));
			q5 = _mm256_madd_epi16(q5, q5);
			q6 = _mm256_sub_epi16(q6, _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(tset + tp + 80))));
			q6 = _mm256_madd_epi16(q6, q6);
			q7 = _mm256_sub_epi16(q7, _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(tset + tp + 96))));
			q7 = _mm256_madd_epi16(q7, q7);
			q8 = _mm256_sub_epi16(q8, _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)(tset + tp + 112))));
			q8 = _mm256_madd_epi16(q8, q8);

			q1 = _mm256_add_epi32(q1, q2);
			q3 = _mm256_add_epi32(q3, q4);
			q5 = _mm256_add_epi32(q5, q6);
			q7 = _mm256_add_epi32(q7, q8);

			q1 = _mm256_add_epi32(q1, q3);
			q5 = _mm256_add_epi32(q5, q7);

			q1 = _mm256_add_epi32(q1, q5);

			__m128i x = _mm_add_epi32(_mm256_extractf128_si256(q1, 1), _mm256_castsi256_si128(q1));
			x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 14));
			x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 1));
			const int score = _mm_cvtsi128_si32(x);

			if (score < second_v) second_v = score;
			if (score < best_v) {
				second_v = best_v;
				best_v = score;
				best_i = t;
			}
		}

		if (static_cast<float>(best_v) > quadrance_thresh * static_cast<float>(second_v)) best_i = -1;
		match_idxs[q] = best_i;
	}
}

template<bool packed_as_uint8_t>
void bruteMatch(std::vector<Match>& matches, const void* const __restrict tset, const int tcount, const void* const __restrict qset, const int qcount, const float threshold) {
	std::vector<int> match_idxs(qcount, -1);
	const int hw_concur = static_cast<int>(std::thread::hardware_concurrency());
	std::future<void>* const __restrict fut = new std::future<void>[hw_concur];
	const int stride = (qcount - 1) / hw_concur + 1;
	const float quadrance_thresh = threshold * threshold;
	int i = 0;
	int start = 0;
	for (; i < std::min(qcount - 1, hw_concur - 1); ++i, start += stride) {
		fut[i] = std::async(std::launch::async, packed_as_uint8_t ? _bruteMatch_uint8_t : _bruteMatch_float, tset, tcount, qset, std::ref(match_idxs), quadrance_thresh, start, stride);
	}
	fut[i] = std::async(std::launch::async, packed_as_uint8_t ? _bruteMatch_uint8_t : _bruteMatch_float, tset, tcount, qset, std::ref(match_idxs), quadrance_thresh, start, qcount - start);
	for (int j = 0; j <= i; ++j) fut[j].wait();

	delete[] fut;

	matches.clear();
	for (int q = 0; q < qcount; ++q) {
		if (match_idxs[q] != -1) {
			matches.emplace_back(q, match_idxs[q]);
		}
	}
}
