#include <stdio.h>
#include <stdlib.h>

// Function deliberately causing a memory leak
void memory_leak_demo() {
    int *leak_ptr = (int*)malloc(sizeof(int) * 100);
    if (!leak_ptr) {
        printf("Memory allocation failed\n");
        return;
    }
    printf("Memory allocated but not freed!\n");
    // Intentionally forgetting to free(leak_ptr);
}

// Corrected function without memory leak
void memory_leak_corrected() {
    int *ptr = (int*)malloc(sizeof(int) * 100);
    if (!ptr) {
        printf("Memory allocation failed\n");
        return;
    }
    printf("Memory allocated and will be freed!\n");
    free(ptr);
}

int main() {
    memory_leak_demo();       // Call leaking function
    memory_leak_corrected();  // Call corrected function

    return 0;
}
