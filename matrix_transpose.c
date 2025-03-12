#include <stdio.h>

// Function to transpose matrix using pointers
void transpose(int *mat, int *res, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            *(res + j * rows + i) = *(mat + i * cols + j);
        }
    }
}

// Function to print matrix
void printMatrix(int *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", *(mat + i * cols + j));
        }
        printf("\n");
    }
}

int main() {
    int rows = 3, cols = 3;
    int mat[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int res[3][3];

    printf("Original Matrix:\n");
    printMatrix((int *)mat, rows, cols);

    transpose((int *)mat, (int *)res, rows, cols);

    printf("\nTransposed Matrix:\n");
    printMatrix((int *)res, cols, rows);

    return 0;
}