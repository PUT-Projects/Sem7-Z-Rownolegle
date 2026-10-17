#include <stdio.h>
#include <time.h>
#include <omp.h>

const long long NUM_STEPS = 1000000000;
double step;

const short CORES = 8;
const short THREADS = 16;

typedef struct
{
	const char* name;      // opis wariantu (do wypisywania)
	double pi;
	double cpu_time;       // czas procesora [s]
	double wall_time;      // czas zegarowy [s]
} Result;

// Obliczenie PI sekwencyjnie
double compute_pi_sequential(void)
{
	double sum = 0.0;
	step = 1. / (double)NUM_STEPS;

	for (long long i = 0; i < NUM_STEPS; i++)
	{
		double x = (i + .5) * step;
		sum = sum + 4.0 / (1. + x * x);
	}
	return sum * step;
}

// Obliczenie PI równolegle (atomic) na zadanej liczbie wątków
double compute_pi_parallel(int num_threads)
{
	double sum = 0.0;
	step = 1. / (double)NUM_STEPS;

	#pragma omp parallel num_threads(num_threads)
	{
		#pragma omp for reduction(+:sum)
		for (long long i = 0; i < NUM_STEPS; i++)
		{
			double x = (i + .5) * step;
			sum += 4.0 / (1. + x * x);
		}
	}
	return sum * step;
}

// Wspólna obudowa pomiaru czasu i wypisania wyniku.
// num_threads <= 0 oznacza wariant sekwencyjny.
Result run_and_measure(const char* name, int num_threads)
{
	Result r;
	r.name = name;

	double wall_start = omp_get_wtime();
	clock_t cpu_start = clock();

	r.pi = (num_threads > 0) ? compute_pi_parallel(num_threads) : compute_pi_sequential();

	clock_t cpu_stop = clock();
	double wall_end = omp_get_wtime();

	r.cpu_time = (double)(cpu_stop - cpu_start) / CLOCKS_PER_SEC;
	r.wall_time = wall_end - wall_start;

	printf("%15.12f Wartosc liczby PI %s \n", r.pi, r.name);
	return r;
}

void print_times(const Result* results, int count)
{
	for (int i = 0; i < count; i++)
		printf("Czas procesorów przetwarzania %s  %f sekund \n", results[i].name, results[i].cpu_time);

	for (int i = 0; i < count; i++)
		printf("Czas trwania obliczen %s - wallclock %f sekund \n", results[i].name, results[i].wall_time);
}

void print_speedups(const Result* seq, const Result* results, int count)
{
	for (int i = 0; i < count; i++)
		printf("Przyspieszenie %s %5.3f \n", results[i].name, seq->wall_time / results[i].wall_time);
}

int main(int argc, char* argv[])
{
	Result results[3];

	results[0] = run_and_measure("sekwencyjnie", 0);
	results[1] = run_and_measure("rownolegle (fizyczne)", CORES);
	results[2] = run_and_measure("rownolegle (logiczne)", THREADS);

	print_times(results, 3);
	print_speedups(&results[0], &results[1], 2);

	return 0;
}