#include <stdio.h>

int countSetBits(unsigned int n) {
    int count = 0;
    while (n) {
        count += n & 1;   // Check if last bit is 1
        n >>= 1;          // Shift right by 1
    }
    return count;
}

// Optional: Using Brian Kernighan's algorithm (faster)
int countSetBitsKernighan(unsigned int n) {
    int count = 0;
    while (n) {
        n = n & (n - 1); // Clears the least significant set bit
        count++;
    }
    return count;
}

int main() {
    unsigned int num;
    printf("Enter a number: ");
    scanf("%u", &num);

    int count1 = countSetBits(num);
    printf("Number of set bits (method 1) = %d\n", count1);

    int count2 = countSetBitsKernighan(num);
    printf("Number of set bits (method 2) = %d\n", count2);

    return 0;
}
