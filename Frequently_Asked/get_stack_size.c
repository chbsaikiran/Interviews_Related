#define STACK_SIZE 65536
char stack[STACK_SIZE];   // simulate stack

int main() {
    // fill stack with 0xDA
    memset(stack, 0xDA, STACK_SIZE);

    // call your functions here

    // check stack usage
    int used = 0;
    for(int i = 0; i < STACK_SIZE; i++) {
        if(stack[i] != (char)0xDA) used++;
    }
    printf("Approximate stack usage: %d bytes\n", used);

    return 0;
}
