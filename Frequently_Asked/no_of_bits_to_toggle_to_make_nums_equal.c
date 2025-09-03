#include <stdio.h>

// Function to count set bits (Brian Kernighan's algorithm)
int countSetBits(unsigned int n) {
    int count = 0;
    while (n) {
        n = n & (n - 1); // clears the least significant set bit
        count++;
    }
    return count;
}

int main() {
    unsigned int a, b;
    printf("Enter two numbers: ");
    scanf("%u %u", &a, &b);

    unsigned int xor = a ^ b;
    int toggles = countSetBits(xor);

    printf("Number of bits to toggle to make %u and %u equal: %d\n", a, b, toggles);

    return 0;
}
