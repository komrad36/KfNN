/*******************************************************************
*   main.cpp
*   KfNN
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Oct 21, 2016
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

#include "KfNN.h"

#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace std::chrono;

int main() {
	// ------------- Configuration ------------
	constexpr int warmups = 0;
	constexpr int runs = 1;
	constexpr int size = 10000;
	constexpr float threshold = 0.98f;
	constexpr bool pack = true;
	// --------------------------------


	// ------------- Generation of Random Data ------------
	// obviously, this is not representative of real data;
	// it doesn't matter for brute-force matching
	std::cout << std::endl << "Generating random test data..." << std::endl;
	std::mt19937 gen(std::mt19937::default_seed);
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);
	float* fqvecs = reinterpret_cast<float*>(malloc(128 * sizeof(float) * size));
	float* ftvecs = reinterpret_cast<float*>(malloc(128 * sizeof(float) * size));
	uint8_t* uqvecs = reinterpret_cast<uint8_t*>(malloc(128 * sizeof(uint8_t) * size));
	uint8_t* utvecs = reinterpret_cast<uint8_t*>(malloc(128 * sizeof(uint8_t) * size));
	for (int i = 0; i < 128 * size; ++i) {
		fqvecs[i] = dis(gen);
		uqvecs[i] = static_cast<uint8_t>(255.0f*fqvecs[i] + 0.5f);
		ftvecs[i] = dis(gen);
		utvecs[i] = static_cast<uint8_t>(255.0f*ftvecs[i] + 0.5f);
	}
	// --------------------------------

	std::vector<Match> matches;
	std::cout << "Warming up..." << std::endl;
	for (int i = 0; i < warmups; ++i) bruteMatch<pack>(matches, pack ? reinterpret_cast<void*>(utvecs) : reinterpret_cast<void*>(ftvecs), size, pack ? reinterpret_cast<void*>(uqvecs) : reinterpret_cast<void*>(fqvecs), size, threshold);
	std::cout << "Testing..." << std::endl;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) bruteMatch<pack>(matches, pack ? reinterpret_cast<void*>(utvecs) : reinterpret_cast<void*>(ftvecs), size, pack ? reinterpret_cast<void*>(uqvecs) : reinterpret_cast<void*>(fqvecs), size, threshold);
	high_resolution_clock::time_point end = high_resolution_clock::now();

	double total = 0.0;
	for (auto& m : matches) {
		for (int i = 0; i < 128; ++i) {
			total += static_cast<double>(ftvecs[(m.t << 7) + i]) + static_cast<double>(fqvecs[(m.q << 7) + i]);
		}
	}
	std::cout.precision(17);
	std::cout << "Checksum: " << total << std::endl;
	std::cout.precision(-1);
	//if (total != 358851.69540632586) {
	//	for (int i = 0; i < 5; ++i) std::cout << "ERROR!" << std::endl;
	//}

	const double sec = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(runs);
	std::cout << std::endl << "Brute force KfNN found " << matches.size() << " matches in " << sec * 1e3 << " ms" << std::endl;
	std::cout << "Throughput: " << static_cast<double>(size)*static_cast<double>(size) / sec * 1e-6 << " million comparisons/second." << std::endl << std::endl;
}
