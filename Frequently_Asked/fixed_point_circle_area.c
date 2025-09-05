#include <stdio.h>

#define PI 25736 //Q2.13 //3.14159

int main()
{
	float radius = 4.67;
	int area_fx;
	float area,area_q;
	short radius_fx = (short)((float)((4.67)*(((int)1) << 12)) + 0.5); //Q3.12
	int temp = PI * radius_fx; //Q2.13 * Q3.12 -> Q5.25
	short temp_sh = temp >> 15; //Q5.10
	area_fx = temp_sh * radius_fx; //Q5.10 * Q3.12 -> Q8.22
	area_q = (area_fx / ((float)(((int)1) << 22)));
	area = (3.14159 * 4.67 * 4.67);
    printf("area float = %f, area fixed = %f, error = %f", area, area_q, (area - area_q));
}

