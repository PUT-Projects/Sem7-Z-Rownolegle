#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define ARRAY_WORDS 64
#define PAIRS       (ARRAY_WORDS - 1)
#define ALIGNMENT   128                 /* wyrównanie początku tablicy do brzegu linii pp */

const long long NUM_STEPS = 20000000;   /* na jedną parę słów; zmniejsz/zwiększ wg potrzeb */

/* Jedno obliczenie PI na parze słów tab[k], tab[k+1]; zwraca czas [s] */
double run_pair(volatile double* tab, int k, double* pi)
{
	double step = 1. / (double)NUM_STEPS;
	tab[k] = 0.0;
	tab[k + 1] = 0.0;

	double start = omp_get_wtime();

	#pragma omp parallel num_threads(2)
	{
		int slot = k + omp_get_thread_num();   /* wątek 0 -> tab[k], wątek 1 -> tab[k+1] */

		#pragma omp for
		for (long long i = 0; i < NUM_STEPS; i++)
		{
			double x = (i + .5) * step;
			tab[slot] = tab[slot] + 4.0 / (1. + x * x);   /* volatile wymusza odczyt+zapis do pamięci */
		}
	}

	double end = omp_get_wtime();
	*pi = (tab[k] + tab[k + 1]) * step;
	return end - start;
}

/* Wyznacza długość linii z okresu powtarzania się krótkich czasów */
void estimate_line_size(const double* t, int count)
{
	double tmin = t[0], tmax = t[0];
	for (int i = 1; i < count; i++)
	{
		if (t[i] < tmin) tmin = t[i];
		if (t[i] > tmax) tmax = t[i];
	}

	if (tmax < 1.3 * tmin)
	{
		printf("\nBrak wyraznej roznicy czasow (min %f, max %f) - eksperyment sie nie powiodl.\n", tmin, tmax);
		return;
	}

	double threshold = tmin * 1.5;   /* progiem jest 1.5 * tmin */
	int first = -1, last = -1, fast = 0;
	for (int i = 0; i < count; i++)
	{
		if (t[i] < threshold)
		{
			if (first < 0) first = i;
			last = i;
			fast++;
		}
	}

	if (fast < 2)
	{
		printf("\nZa malo krotkich iteracji, aby wyznaczyc okres - zwieksz ARRAY_WORDS.\n");
		return;
	}

	double period = (double)(last - first) / (fast - 1);   /* iteracji na jedna linie */
	printf("\nKrotkie iteracje: %d, okres: %.2f iteracji\n", fast, period);
	printf("Szacowana dlugosc linii pamieci podrecznej: %.0f bajtow\n", period * sizeof(double));
}

int main(void)
{
	/* ciągły obszar + ręczne wyrównanie początku do ALIGNMENT bajtów */
	void* raw = malloc(ARRAY_WORDS * sizeof(double) + ALIGNMENT);
	volatile double* tab = (volatile double*)(((uintptr_t)raw + ALIGNMENT - 1) & ~(uintptr_t)(ALIGNMENT - 1));

	double times[PAIRS];
	double pi;

	for (int k = 0; k < PAIRS; k++)
	{
		times[k] = run_pair(tab, k, &pi);
		printf("iteracja %2d (tab[%2d], tab[%2d])  PI = %.12f  czas = %f s\n", k, k, k + 1, pi, times[k]);
	}

	estimate_line_size(times, PAIRS);

	free(raw);
	return 0;
}