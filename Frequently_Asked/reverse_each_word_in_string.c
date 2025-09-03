#include <stdio.h>
#include <string.h>

// Function to reverse a part of string from index start to end
void reverseWord(char *str, int start, int end) {
    while (start < end) {
        char temp = str[start];
        str[start] = str[end];
        str[end] = temp;
        start++;
        end--;
    }
}

// Function to reverse each word in the string
void reverseWordsInString(char *str) {
    int start = 0, end = 0;
    int len = strlen(str);

    while (end <= len) {
        if (str[end] == ' ' || str[end] == '\0') {
            reverseWord(str, start, end - 1);
            start = end + 1;
        }
        end++;
    }
}

int main() {
    char str[] = "Hello world in C";
    printf("Original string: %s\n", str);

    reverseWordsInString(str);
    printf("Reversed words: %s\n", str);

    return 0;
}
