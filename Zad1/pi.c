#include <stdio.h>
#include <time.h>
#include <omp.h>

const long long NUM_STEPS = 100000000;
double step;

const short CORES = 8;
const short THREADS = 16;


int main(int argc, char* argv[])
{
	clock_t sequential_pi_start, sequential_pi_stop,parallel_logical_start,parallel_logical_stop,parallel_phisical_start,parallel_phisical_stop;

	double seq_start_wtime, seq_end_wtime, parallel_logical_start_wtime, parallel_logical_end_wtime, parallel_phisical_start_wtime, parallel_phisical_end_wtime;
//volatile
	double x, pi, sum=0.0;
	int i;
//SEKWENCYJNIE
	seq_start_wtime = omp_get_wtime();
	sequential_pi_start = clock();

	step = 1./(double)NUM_STEPS;

	for (i=0; i<NUM_STEPS; i++)
	{
		x = (i + .5)*step;
		sum = sum + 4.0/(1.+ x*x);
	}

	pi = sum*step;

	sequential_pi_stop = clock();
	seq_end_wtime = omp_get_wtime();
	printf("%15.12f Wartosc liczby PI sekwencyjnie \n", pi);


//RÓWNOLEGLE FIZYCZNE

	parallel_phisical_start_wtime = omp_get_wtime();
	parallel_phisical_start = clock();
	sum = 0.0;
	step = 1. / (double)NUM_STEPS;
	#pragma omp parallel num_threads(CORES)
	#pragma omp for
	for (i = 0; i < NUM_STEPS; i++)
	{
		double x = (i + .5) * step;
		#pragma omp atomic
		sum = sum + 4.0 / (1. + x * x);
	}



	pi = sum * step;

	parallel_phisical_stop = clock();
	parallel_phisical_end_wtime = omp_get_wtime();

	printf("%15.12f Wartosc liczby PI rownolegle (fizyczne) \n",pi);

	// RÓWNOLGŁE LOGICZNE

	parallel_logical_start_wtime = omp_get_wtime();
	parallel_logical_start = clock();

	sum = 0.0;
	step = 1. / (double)NUM_STEPS;
	#pragma omp parallel num_threads(THREADS)
	#pragma omp for
	for (i = 0; i < NUM_STEPS; i++)
	{
		double x = (i + .5) * step;
		#pragma omp atomic
		sum = sum + 4.0 / (1. + x * x);
	}

	pi = sum * step;

	parallel_logical_stop = clock();
	parallel_logical_end_wtime = omp_get_wtime();

	printf("%15.12f Wartosc liczby PI rownolegle (logiczne) \n",pi);


	printf("Czas procesorów przetwarzania sekwencyjnego  %f sekund \n", ((double)(sequential_pi_stop - sequential_pi_start)/CLOCKS_PER_SEC));

	printf("Czas procesorów przetwarzania równoleglego (fizyczne)  %f sekund \n", ((double)(parallel_phisical_stop - parallel_phisical_start)/CLOCKS_PER_SEC));
	printf("Czas procesorów przetwarzania równoleglego (logiczne)  %f sekund \n", ((double)(parallel_logical_stop - parallel_logical_start)/CLOCKS_PER_SEC));

	printf("Czas trwania obliczen sekwencyjnych - wallclock %f sekund \n",  seq_end_wtime-seq_start_wtime);
	printf("Czas trwania obliczen rownoleglych (fizyczne) - wallclock %f sekund \n", parallel_phisical_end_wtime-parallel_phisical_start_wtime);
	printf("Czas trwania obliczen rownoleglych (logiczne) - wallclock %f sekund \n", parallel_logical_end_wtime-parallel_logical_start_wtime);

	printf("Przyspieszenie (rdzenie fizyczne) %5.3f \n", (seq_end_wtime - seq_start_wtime) / (parallel_phisical_end_wtime - parallel_phisical_start_wtime));
	printf("Przyspieszenie (rdzenie logiczne) %5.3f \n", (seq_end_wtime - seq_start_wtime) / (parallel_logical_end_wtime - parallel_logical_start_wtime));
	return 0;
}
