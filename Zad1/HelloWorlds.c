#include <stdio.h>
#include <omp.h>

int main() {
		int i;
		#pragma omp parallel num_threads(8)
		{
			#pragma omp for schedule(guided, 1) nowait
			for(i = 0; i < 10; i++) {
				int thread_num = omp_get_thread_num();
				printf("Hello World Iter: %d from thread %d\n", i, thread_num);
			}
			//Czekamy na pozostałe wątki, bariera pamięci. Zapis zmienianych wartości
			printf("GoodBye World\n");
		}
		printf("i=%d\n", i);
}