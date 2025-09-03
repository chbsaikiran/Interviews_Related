#include <stdio.h>

#define N 3

int main() {
    int matrix[N][N] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Transpose in-place
    for(int i = 0; i < N; i++) {
        for(int j = i + 1; j < N; j++) {
            int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }

    // Print transposed matrix
    printf("Transposed Matrix:\n");
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++)
            printf("%d ", matrix[i][j]);
        printf("\n");
    }

    return 0;
}
