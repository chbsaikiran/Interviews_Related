#include <stdio.h>

// Function with matching signature
int* myFunction(char* str1, char* str2) {
    static int result = 42;
    printf("str1 = %s, str2 = %s\n", str1, str2);
    return &result;
}

int main() {
    // Declare function pointer
    int* (*funcPtr)(char*, char*);

    // Assign function to pointer
    funcPtr = myFunction;

    // Call function through pointer
    int* ptr = funcPtr("Hello", "World");
    printf("Returned value = %d\n", *ptr);

    return 0;
}
