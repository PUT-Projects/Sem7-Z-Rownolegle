#include <stdio.h>
#include <omp.h>

int main() {
		#pragma omp parallel num_threads(4)
		{
			int i;
			for(i=0;i<6;i++)
				printf("Iter:%d\n",i);

			printf("GoodBye World\n");
		}
}