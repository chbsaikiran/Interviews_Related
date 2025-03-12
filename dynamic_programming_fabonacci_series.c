#include <stdio.h>

#define MAX 100

// Initialize memoization array
int memo[MAX];

// Recursive Fibonacci with memoization (DP)
int fibonacci(int n) {
    if (n <= 1)
        return n;

    // Return cached result if already computed
    if (memo[n] != -1)
        return memo[n];

    // Otherwise, compute and cache the result
    memo[n] = fibonacci(n - 1) + fibonacci(n - 2);

    return memo[n];
}

int main() {
    int n = 10; // Example: 10th Fibonacci number

    // Initialize memo array
    for (int i = 0; i <= n; i++)
        memo[i] = -1;

    printf("Fibonacci number at position %d is %d\n", n, fibonacci(n));

    return 0;
}
