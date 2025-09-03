#include <stdio.h>
#include <string.h>

int main() {
    char str[] = "Hello world, welcome to C programming";
    char word[] = "welcome";

    char *token = strtok(str, " ,"); // split by space or comma
    int found = 0;

    while (token != NULL) {
        if (strcmp(token, word) == 0) {
            found = 1;
            break;
        }
        token = strtok(NULL, " ,");
    }

    if (found)
        printf("Word \"%s\" found in the string.\n", word);
    else
        printf("Word \"%s\" not found.\n", word);

    return 0;
}
