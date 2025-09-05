#include <stdio.h>

int main()
{
	int arr[5] = { 50,20,60,10,55 };
	int i,j,l1=-1,l2=-1;

	for (i = 0; i < 5; i++)
	{
		if (arr[i] > l1)
		{
			l2 = l1; 
			l1 = arr[i];
		}
		else if (arr[i] > l2 && arr[i] != l1)
		{
			l2 = arr[i];
		}
	}
	printf("%d", l2);
	return 0;
}