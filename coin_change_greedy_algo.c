#include <stdio.h>

// Greedy algorithm for coin change problem
void coinChange(int coins[], int numCoins, int amount) {
    int count[numCoins];

    // Initialize count array
    for (int i = 0; i < numCoins; i++)
        count[i] = 0;

    // Greedy algorithm: start from largest coin
    for (int i = 0; i < numCoins; i++) {
        while (amount >= coins[i]) {
            amount -= coins[i];
            count[i]++;
        }
    }

    // Display the results
    printf("Coin change solution:\n");
    for (int i = 0; i < numCoins; i++) {
        if (count[i] > 0)
            printf("%d coin(s) of %d\n", count[i], coins[i]);
    }
}

int main() {
    int coins[] = {25, 10, 5, 1};  // Coins in descending order
    int numCoins = sizeof(coins) / sizeof(coins[0]);
    int amount = 87;

    printf("Amount: %d\n", amount);
    coinChange(coins, numCoins, amount);

    return 0;
}
