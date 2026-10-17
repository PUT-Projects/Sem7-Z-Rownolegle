#include <stdio.h>
#include <omp.h>

int main() {

		#pragma omp parallel
		{
			int thread_num = omp_get_thread_num();
			printf("Hello World from thread %d\n", thread_num);
			// int i;
			// for(i=0;i<6;i++)
			// 	printf("Iter:%d\n",i);

			// printf("GoodBye World\n");
		}
}