#include <stdio.h>

#define N 4
#define B 2
#define MIN(a,b) ((a) < (b) ? (a) : (b))

int main()
{
	int jj, kk, i, j, k,r;
	int x[4][4] = { {0,0,0,0},{0,0,0,0} ,{0,0,0,0} ,{0,0,0,0} };
	int y[4][4] = { {1,2,3,4},{5,6,7,8} ,{9,10,11,12} ,{13,14,15,16} };
	int z[4][4] = { {0,1,0,0},{1,0,0,0} ,{0,0,0,1} ,{0,0,1,0} };

	for (jj = 0; jj < N; jj += B)
	{
		for (kk = 0; kk < N; kk += B)
		{
			for (i = 0; i < N; i += 1)
			{
				for (j = jj; j < MIN((jj + B), N); j++)
				{
					r = 0;
					for (k = kk; k < MIN((kk + B), N); k++)
					{
						r = r + y[i][k] * z[k][j];
					}
					x[i][j] = x[i][j] + r;
				}
			}
		}
	}

	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			printf("%d ", x[i][j]);
		}
		printf("\n");
	}

	return 0;
}